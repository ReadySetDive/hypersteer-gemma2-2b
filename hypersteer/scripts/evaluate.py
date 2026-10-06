import asyncio
import datetime
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx
import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from openai import AsyncOpenAI

import wandb
from hypersteer import LMJudgeEvaluator, PerplexityEvaluator, WinRateEvaluator
from hypersteer.data.utils import load_dataset_for_inference
from hypersteer.utils.configs import (
    ExperimentConfig,
    config_to_pydantic,
)
from hypersteer.utils.constants import (
    EVAL_STATE_FILE,
    STEERING_EXCLUDE_MODELS,
)
from hypersteer.utils.dry_run import patch_client
from hypersteer.utils.helpers import (
    dump_json,
    get_logger,
    load_state,
    process_jsonl_file,
)
from hypersteer.utils.language_models import LanguageModel
from hypersteer.utils.plot_utils import (
    plot_metrics,
    plot_win_rates,
)

logger = get_logger(__name__)


def find_run_id(
    run_name: str,
    project: str,
    entity: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
) -> str | None:
    """
    Find a wandb run ID given the run name.

    Args:
        run_name: The name of the run to find
        project: The wandb project name
        entity: The wandb entity (username or team name)
        group: Optional group to filter by
        tags: Optional list of tags to filter by

    Returns:
        The run ID if found, None otherwise
    """
    api = wandb.Api()

    # Build filter string
    filters = {"display_name": run_name}
    if group:
        filters["group"] = group
    if tags:
        filters["tags"] = {"$in": tags}

    # Query the API
    runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)

    # Return the first matching run ID
    for run in runs:
        return run.id

    return None


def data_generator(data_dir, mode, winrate_split_ratio=None, infer_run="inference"):
    """
    Generator function to read data files and yield data subsets by group_id.
    Pre-loads data in chunks to reduce I/O bottlenecks.

    Args:
        data_dir (str): Path to the data directory.
        mode (str): Mode of operation ('latent' or 'steering').

    Yields:
        (group_id, df_subset): A tuple containing the group_id and subset DataFrame.
    """
    # Pre-load and organize data by concept_id
    concept_data = {}
    if mode == "latent":
        df = pd.read_parquet(os.path.join(data_dir, infer_run, "latent_data.parquet"))
    elif "steering" in mode or mode == "winrate":
        df = pd.read_parquet(os.path.join(data_dir, infer_run, "steering_data.parquet"))
    # Group by concept_id and store in dictionary
    for concept_id, group in df.groupby("concept_id"):
        if concept_id not in concept_data:
            concept_data[concept_id] = []
        concept_data[concept_id].append(group)

    # Yield concatenated data for each concept_id
    for concept_id in sorted(concept_data.keys()):
        if len(concept_data[concept_id]) > 1:
            df_subset = pd.concat(concept_data[concept_id])
        else:
            df_subset = concept_data[concept_id][0]
        if winrate_split_ratio is not None:
            n_input_ids = df_subset["input_id"].max() + 1
            n_steering_ids = n_input_ids - round(n_input_ids * winrate_split_ratio)
            if mode == "steering":
                df_subset = df_subset[df_subset["input_id"] < n_steering_ids]
            elif mode == "steering_test" or mode == "winrate":
                df_subset = df_subset[df_subset["input_id"] >= n_steering_ids]
        yield (concept_id, df_subset)


def get_best_factors(aggregated_results):
    best_factors = {}
    for result in aggregated_results:
        best_factors[result["concept_id"]] = {}
        for method, scores in result["results"]["LMJudgeEvaluator"].items():
            best_factors[result["concept_id"]][method] = scores["factor"][
                np.argmax(scores["lm_judge_rating"])
            ]
    return best_factors


def winrate_data_generator(data_dir, aggregated_results, winrate_split_ratio):
    best_factors = get_best_factors(aggregated_results)
    df_generator = data_generator(
        data_dir, mode="winrate", winrate_split_ratio=winrate_split_ratio
    )
    for concept_id, current_df in df_generator:
        # if concept_id >= start_concept_id: # TODO: uncomment this when we fix our pipeline
        concept_best_dfs = {}
        for method, factor in best_factors[concept_id].items():
            include_columns = [
                "concept_id",
                "input_concept",
                "input_id",
                "original_prompt",
                "steered_input",
                "factor",
                f"{method}_steered_generation",
            ]
            method_df = current_df[include_columns]
            method_best_df = method_df[method_df["factor"] == factor]
            concept_best_dfs[method] = method_best_df.copy()
            concept_best_df = method_best_df[
                [
                    "concept_id",
                    "input_concept",
                    "input_id",
                    "original_prompt",
                    "steered_input",
                ]
            ].copy()
        for method in best_factors[concept_id].keys():
            # Use merge instead of direct assignment to ensure proper alignment
            concept_best_df = concept_best_df.merge(
                concept_best_dfs[method][
                    [
                        "concept_id",
                        "input_concept",
                        "input_id",
                        f"{method}_steered_generation",
                    ]
                ],
                on=["concept_id", "input_concept", "input_id"],
                how="left",
            )
        yield (concept_id, concept_best_df)


def save_results(
    dump_dir,
    state,
    concept_id,
    partition,
    eval_results,
    eval_df=None,
    eval_run="evaluate",
):
    """
    Save the results dictionary to a .jsonl file.
    Each line in the file represents one concept_id's evaluation results.
    """
    # handle training df first
    dump_dir = Path(dump_dir) / eval_run
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{EVAL_STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    # Define the output file path for JSON Lines
    result_path = Path(dump_dir) / f"{partition}.jsonl"
    result_entry = {"concept_id": int(concept_id), "results": eval_results}
    with open(result_path, "a") as f:
        f.write(json.dumps(result_entry) + "\n")

    # save the steering ratings for each example
    if eval_df is not None:
        sorted_evaluator_names = sorted(list(eval_df.keys()))
        sorted_model_names = sorted(list(eval_df[sorted_evaluator_names[0]].keys()))
        if len(sorted_evaluator_names) == 0:
            return
        current_df = eval_df[sorted_evaluator_names[0]][sorted_model_names[0]].copy()

        for evaluator_name in sorted_evaluator_names:
            if evaluator_name == "PerplexityEvaluator":
                continue
            if evaluator_name == "WinRateEvaluator":
                continue
            for model_name in sorted_model_names:
                if (
                    evaluator_name == sorted_evaluator_names[0]
                    and model_name == sorted_model_names[0]
                ):
                    continue

                current_df[f"{model_name}_{evaluator_name}"] = eval_df[evaluator_name][
                    model_name
                ][f"{model_name}_{evaluator_name}"]

                current_df[
                    f"{model_name}_{evaluator_name}_relevance_concept_ratings"
                ] = eval_df[evaluator_name][model_name][
                    f"{model_name}_{evaluator_name}_relevance_concept_ratings"
                ]
                current_df[
                    f"{model_name}_{evaluator_name}_relevance_concept_completions"
                ] = eval_df[evaluator_name][model_name][
                    f"{model_name}_{evaluator_name}_relevance_concept_completions"
                ]

                current_df[
                    f"{model_name}_{evaluator_name}_relevance_instruction_ratings"
                ] = eval_df[evaluator_name][model_name][
                    f"{model_name}_{evaluator_name}_relevance_instruction_ratings"
                ]
                current_df[
                    f"{model_name}_{evaluator_name}_relevance_instruction_completions"
                ] = eval_df[evaluator_name][model_name][
                    f"{model_name}_{evaluator_name}_relevance_instruction_completions"
                ]

                current_df[f"{model_name}_{evaluator_name}_fluency_ratings"] = eval_df[
                    evaluator_name
                ][model_name][f"{model_name}_{evaluator_name}_fluency_ratings"]
                current_df[f"{model_name}_{evaluator_name}_fluency_completions"] = (
                    eval_df[evaluator_name][model_name][
                        f"{model_name}_{evaluator_name}_fluency_completions"
                    ]
                )

        df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
        if os.path.exists(df_path):
            existing_df = pd.read_parquet(df_path)
            combined_df = pd.concat([existing_df, current_df], ignore_index=True)
        else:
            combined_df = current_df
        combined_df.to_parquet(df_path, index=False)


def plot_steering(
    aggregated_results, dump_dir, report_to=[], wandb_name=None, mode=None
):
    try:
        configs = [
            {
                "evaluator_name": "PerplexityEvaluator",
                "metric_name": "perplexity",
                "y_label": "Perplexity",
                "use_log_scale": False,
            },
            {
                "evaluator_name": "LMJudgeEvaluator",
                "metric_name": "relevance_concept_ratings",
                "y_label": "Concept",
                "use_log_scale": False,
            },
            {
                "evaluator_name": "LMJudgeEvaluator",
                "metric_name": "relevance_instruction_ratings",
                "y_label": "Instruct",
                "use_log_scale": False,
            },
            {
                "evaluator_name": "LMJudgeEvaluator",
                "metric_name": "fluency_ratings",
                "y_label": "Fluency",
                "use_log_scale": False,
            },
            {
                "evaluator_name": "LMJudgeEvaluator",
                "metric_name": "lm_judge_rating",
                "y_label": "Aggregated",
                "use_log_scale": False,
            },
        ]
        plot_metrics(
            jsonl_data=aggregated_results,
            configs=configs,
            write_to_path=dump_dir,
            report_to=report_to,
            wandb_name=wandb_name,
            mode=mode,
        )
    except Exception as e:
        logger.warning(f"Failed to plot: {e}")


def process_mask_sparsity_metrics(
    steering_df, dump_dir, eval_run, args: ExperimentConfig
):
    """
    Calculate and log mask sparsity (L1 norm) metrics and plots for all columns ending with '_sparsity'.
    """
    import matplotlib.pyplot as plt

    sparsity_metrics = {}
    sparsity_cols = [col for col in steering_df.columns if col.endswith("_sparsity")]
    if not sparsity_cols:
        logger.warning(
            "No columns ending with '_sparsity' found in steering_data.parquet."
        )
        return
    for col in sparsity_cols:
        model_name = col.replace("_sparsity", "")

        def reduce_sparsity(x):
            # Handle cases where sparsity might be stored as list/ndarray per example
            if isinstance(x, list) or isinstance(x, np.ndarray):
                return np.mean(x)
            return x

        reduced_sparsity = steering_df[col].dropna().map(reduce_sparsity)
        avg_sparsity = reduced_sparsity.mean()
        std_sparsity = reduced_sparsity.std()
        sparsity_metrics[f"eval_mean_mask_sparsity/{model_name}"] = avg_sparsity
        sparsity_metrics[f"eval_std_mask_sparsity/{model_name}"] = std_sparsity
        logger.warning(
            f"Average mask sparsity for {model_name}: {avg_sparsity:.4f} (std: {std_sparsity:.4f})"
        )
        # --- Save histogram plot to disk ---
        plt.figure(figsize=(8, 5))
        plt.hist(reduced_sparsity, bins=30, color="skyblue", edgecolor="black")
        plt.title(
            f"Mask Sparsity for {model_name}\nMean: {avg_sparsity:.4f}, Std: {std_sparsity:.4f}"
        )
        plt.xlabel("Per-example mask sparsity")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        # Optionally, add text box with stats
        plt.gca().text(
            0.98,
            0.95,
            f"Mean: {avg_sparsity:.4f}\nStd: {std_sparsity:.4f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        plot_path = Path(dump_dir) / eval_run / f"mask_sparsity_{model_name}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.warning(f"Saved mask sparsity histogram to {plot_path}")
        # Log histogram to wandb if enabled
        if args.evaluate.report_to == "wandb" and wandb.run:
            wandb.log(
                {
                    f"eval/mask_sparsity_hist/{model_name}": wandb.Histogram(
                        reduced_sparsity.values
                    )
                }
            )
        # Group by factor and plot boxplot/violinplot
        if "factor" in steering_df.columns:
            import seaborn as sns

            factor_vals = steering_df["factor"]
            # Align factor and sparsity values (dropna alignment)
            factor_aligned = factor_vals[reduced_sparsity.index]
            plot_df = pd.DataFrame(
                {
                    "factor": factor_aligned,
                    "sparsity": reduced_sparsity.values,
                }
            )
            plt.figure(figsize=(8, 5))
            sns.boxplot(x="factor", y="sparsity", data=plot_df, color="skyblue")
            plt.title(f"Mask Sparsity by Steering Factor for {model_name}")
            plt.xlabel("Steering Factor")
            plt.ylabel("Mask Sparsity")
            plt.grid(True, alpha=0.3)
            boxplot_path = (
                Path(dump_dir) / eval_run / f"mask_sparsity_boxplot_{model_name}.png"
            )
            plt.tight_layout()
            plt.savefig(boxplot_path)
            plt.close()
            logger.warning(f"Saved mask sparsity boxplot to {boxplot_path}")
            # Log boxplot to wandb
            if args.evaluate.report_to == "wandb" and wandb.run:
                wandb.log(
                    {
                        f"eval/mask_sparsity_boxplot/{model_name}": wandb.Image(
                            str(boxplot_path)
                        )
                    }
                )
    # Log to wandb if enabled
    if args.evaluate.report_to == "wandb" and wandb.run:
        wandb.log(sparsity_metrics)
        logger.warning("Logged average mask sparsity to wandb.")


def eval_steering_single_task(args_tuple):
    """Helper function to evaluate a single concept-model-evaluator combination"""
    (
        concept_id,
        current_df,
        evaluator_name,
        model_name,
        dump_dir,
        lm_model,
        winrate_baseline,
        lm_caches,
    ) = args_tuple

    # Create LanguageModel instance within the worker process
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=int(
                    os.environ.get("OPENAI_MAX_KEEPALIVE_CONNECTIONS", 100)
                ),
                max_connections=int(os.environ.get("OPENAI_MAX_CONNECTIONS", 1000)),
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )
    if int(os.environ.get("OPENAI_DRY_RUN", "1")) == 1:
        client = patch_client(client)
    lm_model = LanguageModel(
        lm_model,
        client,
        dump_dir=dump_dir,
        use_cache=True,
        cache_level="prompt",
        cache_tag="evaluate",
        master_data_dir="hypersteer/data",
        temperature=0.7,
    )
    # overwrite cache if any.
    if bool(lm_caches):
        lm_model.cache_in_mem = lm_caches

    try:
        # Map evaluator names to classes
        evaluator_classes = {
            "LMJudgeEvaluator": LMJudgeEvaluator,
            "PerplexityEvaluator": PerplexityEvaluator,
            "WinRateEvaluator": WinRateEvaluator,
        }
        evaluator_class = evaluator_classes[evaluator_name]
        evaluator = evaluator_class(
            model_name,
            dump_dir=dump_dir,
            concept_id=concept_id,
            lm_model=lm_model,
            winrate_baseline=winrate_baseline,
        )
        eval_result = evaluator.compute_metrics(current_df)
        return (
            concept_id,
            evaluator.__str__(),
            model_name.__str__(),
            eval_result,
            lm_model.stats.get_report(),
            None if bool(lm_caches) else lm_model.cache_in_mem,
            current_df,
        )
    except Exception as e:
        logger.error(
            f"Error evaluating concept_id {concept_id}, model {model_name}, evaluator {evaluator_name}: {e}"
        )
        raise e
    finally:
        # Properly close both the HTTP client and async client
        async def cleanup():
            await client.close()

        asyncio.run(cleanup())


def eval_steering(
    args: ExperimentConfig,
    select_concept_ids=None,
    return_results=False,
    eval_run="evaluate",
    infer_run="inference",
):
    """
    Evaluate steering performance using multi-processing for all tasks
    """
    dump_dir = args.dump_dir

    # Initialize data generator
    df_generator = data_generator(
        args.dump_dir,
        mode="steering",
        winrate_split_ratio=args.evaluate.winrate_split_ratio,
        infer_run=infer_run,
    )

    # Collect all data from the generator into a list
    concept_data_list = list(df_generator)

    # Load previous state if exists
    state = (
        load_state(args.dump_dir, mode="steering", eval_run=eval_run)
        if not getattr(args, "ignore_steering_state", False)
        else None
    )
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept_id: {start_concept_id}")

    if select_concept_ids is not None:
        start_concept_id = select_concept_ids[0]

    # Determine which models to evaluate
    models_to_evaluate = []
    if args.evaluate.models:
        # Use the models list if provided
        models_to_evaluate = args.evaluate.models
    else:
        # Fall back to single model_name for backward compatibility
        models_to_evaluate = [args.model.model_name]

    logger.info(f"Evaluating models: {models_to_evaluate}")

    # Create all evaluation tasks - flattened for maximum parallelization
    all_tasks = []
    for model_name in models_to_evaluate:
        if model_name not in STEERING_EXCLUDE_MODELS:
            model_tasks = [
                (
                    concept_id,
                    current_df,
                    evaluator_name,
                    model_name,
                    args.dump_dir,
                    args.evaluate.lm_model,
                    args.evaluate.winrate_baseline,
                    {},
                )
                for concept_id, current_df in concept_data_list
                if concept_id >= start_concept_id
                and (select_concept_ids is None or concept_id in select_concept_ids)
                for evaluator_name in args.evaluate.steering_evaluators
            ]
            all_tasks.extend(model_tasks)
        else:
            logger.warning(
                f"Model {model_name} is excluded from steering evaluation, skipping"
            )

    # Group results by concept_id
    all_results = {}

    # Run all evaluations with process pool
    logger.warning(
        f"Number of workers: {args.evaluate.num_of_workers}; Number of CPUs: {multiprocessing.cpu_count()}"
    )
    if (
        not hasattr(args.evaluate, "num_of_workers")
        or args.evaluate.num_of_workers is None
    ):
        args.evaluate.num_of_workers = max(1, multiprocessing.cpu_count() - 1)
    lm_reports = []
    eval_dfs = {}
    lm_caches = {}
    with ProcessPoolExecutor(max_workers=args.evaluate.num_of_workers) as executor:
        for (
            concept_id,
            evaluator_str,
            model_str,
            result,
            lm_report,
            lm_cache,
            current_df,
        ) in executor.map(eval_steering_single_task, all_tasks):
            if concept_id not in all_results:
                all_results[concept_id] = {}
                eval_dfs[concept_id] = {}
            if evaluator_str not in all_results[concept_id]:
                all_results[concept_id][evaluator_str] = {}
                eval_dfs[concept_id][evaluator_str] = {}
            all_results[concept_id][evaluator_str][model_str] = result
            if (
                "raw_relevance_concept_ratings" in result
                or "raw_relevance_instruction_ratings" in result
                or "raw_fluency_ratings" in result
                or "raw_aggregated_ratings" in result
            ):
                current_df[f"{model_str}_{evaluator_str}_relevance_concept_ratings"] = (
                    result["raw_relevance_concept_ratings"]
                )
                current_df[
                    f"{model_str}_{evaluator_str}_relevance_instruction_ratings"
                ] = result["raw_relevance_instruction_ratings"]
                current_df[f"{model_str}_{evaluator_str}_fluency_ratings"] = result[
                    "raw_fluency_ratings"
                ]
                current_df[f"{model_str}_{evaluator_str}"] = result[
                    "raw_aggregated_ratings"
                ]
                current_df[
                    f"{model_str}_{evaluator_str}_relevance_concept_completions"
                ] = result["relevance_concept_completions"]
                current_df[
                    f"{model_str}_{evaluator_str}_relevance_instruction_completions"
                ] = result["relevance_instruction_completions"]
                current_df[f"{model_str}_{evaluator_str}_fluency_completions"] = result[
                    "fluency_completions"
                ]
                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()
            else:
                eval_dfs[concept_id][evaluator_str][model_str] = current_df.copy()
            lm_reports += [lm_report]
            lm_caches.update(lm_cache)
            logger.warning(
                f"Completed task for concept_id: {concept_id}, model: {model_str}, evaluator: {evaluator_str}"
            )

    # Batch save all results
    current_aggregated_results = []
    for concept_id, eval_results in sorted(all_results.items()):
        save_results(
            dump_dir,
            {"concept_id": concept_id + 1},
            concept_id,
            "steering",
            eval_results,
            eval_dfs[concept_id],
            eval_run=eval_run,
        )
        current_aggregated_results.append(
            {"concept_id": concept_id + 1, "results": eval_results}
        )

    # Reload for plotting and optional winrate
    try:
        aggregated_results = process_jsonl_file(
            load_jsonl(os.path.join(Path(dump_dir) / eval_run / "steering.jsonl"))
        )
    except Exception as e:
        logger.warning(f"Failed to load steering.jsonl: {e}. Aborting evaluation.")
        return

    # Calculate and log average mask sparsity
    steering_data_path = Path(dump_dir) / infer_run / "steering_data.parquet"
    if steering_data_path.exists():
        steering_df = pd.read_parquet(steering_data_path)
        sparsity_cols = [
            col for col in steering_df.columns if col.endswith("_sparsity")
        ]
        if sparsity_cols:
            process_mask_sparsity_metrics(steering_df, dump_dir, eval_run, args)
        else:
            logger.warning(
                "No columns ending with '_sparsity' found in steering_data.parquet."
            )
    else:
        logger.warning(f"Steering data parquet file not found at {steering_data_path}.")

    # Aggregate LM reports
    aggregated_lm_report = {
        "total_calls": sum([report["total_calls"] for report in lm_reports]),
        "total_cache_hits": sum([report["total_cache_hits"] for report in lm_reports]),
        "total_price": sum([report["total_price"] for report in lm_reports]),
    }
    logger.warning("=" * 20)
    logger.warning(
        f"Total calls: {aggregated_lm_report['total_calls']}, "
        f"Total cache hits: {aggregated_lm_report['total_cache_hits']}"
    )
    logger.warning(f"Total price: ${aggregated_lm_report['total_price']}")
    logger.warning("=" * 20)

    # Generate final plot
    if not return_results:
        logger.warning("Generating final plot...")
        plot_steering(
            aggregated_results,
            Path(dump_dir) / eval_run,
            args.evaluate.report_to,
            args.wandb.run_name,
            "steering",
        )

        # Plot winrate
        if "WinRateEvaluator" in args.evaluate.steering_evaluators:
            logger.warning("Generating winrate plot...")
            plot_win_rates(
                aggregated_results,
                Path(args.dump_dir) / eval_run,
                args.evaluate.report_to,
                args.wandb.run_name,
            )

    logger.warning("Evaluation completed!")

    if return_results:
        # IMPORTANT: if we return aggregated_results this is cumulative and leads to silent errors
        return current_aggregated_results


def load_jsonl(jsonl_path):
    """
    Load data from a JSON lines file.
    """
    jsonl_data = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            jsonl_data += [data]
    return jsonl_data


def log_results_to_wandb(
    dump_dir,
    eval_run,
    infer_run,
    concept_info=None,
    run=None,
):
    """
    Log evaluation results to Weights & Biases.

    Args:
        dump_dir (str): Path to the directory with results
        eval_run (str): Name of the evaluation run folder
        infer_run (str): Name of the inference run folder
        concept_info (list, optional): Concept information from dataset
        run (wandb.Run, optional): Active wandb run object
    """
    # Get concept info from config if not provided
    if concept_info is None:
        try:
            # Try to load from the saved config
            config_path = Path(dump_dir) / "config.yaml"
            if config_path.exists():
                cfg = OmegaConf.load(config_path)
                args = config_to_pydantic(cfg, ExperimentConfig)
                concept_info = load_dataset_for_inference(args)
        except Exception as e:
            logger.warning(f"Could not load concept info from dataset: {e}")
            concept_info = []

    concepts = []

    # Process steering results if available
    if (Path(dump_dir) / eval_run / "steering.jsonl").is_file() and concept_info:
        steering_path = Path(dump_dir) / eval_run / "steering.jsonl"
        steering_results = load_jsonl(steering_path)
        best_factors = get_best_factors(steering_results)
        # Check if any model results exist in WinRateEvaluator
        lsreft_included = (
            len(steering_results[0]["results"].get("WinRateEvaluator", {})) > 0
            if "WinRateEvaluator" in steering_results[0]["results"]
            else False
        )

        # Get the first available model name from LMJudgeEvaluator results
        lsreft_model_name = next(
            iter(steering_results[0]["results"].get("LMJudgeEvaluator", {})),
            None,
        )

        if lsreft_model_name:
            idx = 0
            for concept_entry in concept_info:
                concept = concept_entry["concept"]
                sae_link = concept_entry["ref"]
                winrate = (
                    steering_results[idx]["results"]["WinRateEvaluator"][
                        lsreft_model_name
                    ]["win_rate"]
                    if lsreft_included
                    and "WinRateEvaluator" in steering_results[idx]["results"]
                    else None
                )
                best_factor = (
                    best_factors[idx][lsreft_model_name]
                    if idx in best_factors
                    else None
                )
                if len(concepts) <= idx:
                    concepts += [
                        [idx, concept, winrate, None, None, best_factor, sae_link]
                    ]
                else:
                    concepts[idx][2] = winrate
                    concepts[idx][5] = best_factor
                idx += 1

        # Win-rate table logging
        winrate_path = Path(dump_dir) / eval_run / "steering_data.parquet"
        if winrate_path.exists():
            winrate_df = pd.read_parquet(winrate_path)
            wandb.log({"steering/winrate": wandb.Table(dataframe=winrate_df)})

    # Log concept table if we have data
    if concepts:
        wandb.log(
            {
                "concept_table": wandb.Table(
                    columns=[
                        "concept_id",
                        "concept",
                        "winrate",
                        "auc",
                        "max_act",
                        "best_factor",
                        "sae_link",
                    ],
                    data=concepts,
                )
            }
        )

    # Only finish the run if it was passed in
    if run is not None:
        run.finish()


def run_eval(args: ExperimentConfig, infer_run="inference"):
    logger.warning(
        f"Evaluating generations with the following configuration:\n{dump_json(args.evaluate.model_dump(), indent=4)}"
    )

    # Determine evaluation run directory logic
    if args.evaluate.eval_run:
        # If a custom eval_run value is provided, always use it
        eval_run = args.evaluate.eval_run
    elif args.evaluate.run_distinct_evals:
        # Otherwise, if distinct evals is requested, use a timestamped run
        eval_run = f"eval_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    else:
        # Otherwise, use the default "evaluate" directory
        eval_run = "evaluate"

    eval_dump_dir = Path(args.dump_dir) / eval_run
    if (
        not args.evaluate.overwrite_existing_eval
        and eval_dump_dir.exists()
        and any(eval_dump_dir.iterdir())
    ):
        logger.warning(
            f"Eval dump dir {eval_dump_dir} already exists and is nonempty. Skipping."
        )
        return

    eval_dump_dir.mkdir(parents=True, exist_ok=True)

    # start wandb logging if we are not in an active run
    if args.wandb.log and not wandb.run and args.evaluate.report_to == "wandb":
        _run_id = find_run_id(
            args.wandb.run_name,
            project=args.wandb.project,
            entity=args.wandb.entity,
            group=args.wandb.group,
        )
        run = wandb.init(
            project=args.wandb.project,
            entity=args.wandb.entity,
            id=_run_id,
            resume="allow",
        )

    eval_steering(args, eval_run=eval_run, infer_run=infer_run)

    # Log metadata about the factor selection run to help with traceability
    metadata_path = Path(args.dump_dir) / eval_run / "eval_metadata.json"
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "infer_run": str(infer_run),
        "eval_run": str(eval_run),
        "concept_ids": args.dataset.select_concept_ids
        if args.dataset.select_concept_ids
        else "all",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if args.wandb.log and args.evaluate.report_to == "wandb":
        # Load concept info for wandb logging
        try:
            concept_info = load_dataset_for_inference(args)
        except Exception as e:
            logger.warning(f"Could not load concept info for wandb logging: {e}")
            concept_info = None

        log_results_to_wandb(
            dump_dir=args.dump_dir,
            eval_run=eval_run,
            infer_run=infer_run,
            concept_info=concept_info,
            run=run if "run" in locals() else None,
        )


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Use experiment config if it exists, otherwise use the main config
    config = cfg.experiment if hasattr(cfg, "experiment") else cfg

    # Handle pretrained config merging if needed
    pretrained_cfg_path = Path(config.dump_dir) / "config.yaml"
    if pretrained_cfg_path.exists():
        logger.info(f"Loading pretrained config from {pretrained_cfg_path}")
        pretrained_cfg = OmegaConf.load(pretrained_cfg_path)
        OmegaConf.set_struct(config, False)
        config = OmegaConf.merge(config, pretrained_cfg)
        OmegaConf.set_struct(config, True)

    config = config_to_pydantic(config, ExperimentConfig)
    run_eval(config, infer_run=config.evaluate.infer_run or "inference")


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
