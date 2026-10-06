import asyncio
import os
from collections import namedtuple

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from hypersteer.utils.constants import EMPTY_CONCEPT
from hypersteer.utils.helpers import get_logger
from hypersteer.utils.language_models import LanguageModel
from hypersteer.utils.model_utils import get_suffix_length

from .base import (
    BaseDatasetFactory,
    register_factory,
)

logger = get_logger(__name__)


T_PROMPT_STEERING = """You must answer the question with content \
related to %s even if it is not related to the question or it does not make sense."""


T_GENERATE_STEERING_PROMPT = """Generate a prompt to guide a language \
model in producing responses. 

Objective: 
Direct the model to include content related to %s (the concept) in its responses. 
Ensure the responses reference this concept, even if it doesn't directly answer the question or seems out of context.
Optionally, provide in-context examples to reinforce this behavior.
        
Return only the final prompt without any additional text."""


# special types for dataset
Prompt = namedtuple("Prompt", ["concept", "tag", "content"])


async def run_tasks(tasks):
    # Gather and run all provided tasks concurrently, and collect their results
    results = await asyncio.gather(*tasks)
    return results


def apply_chat_template_llama(
    example, tokenizer, suffix_length, binarize=False, output_length=None
):
    """Apply chat template for Llama models"""
    if binarize:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        nobos = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )[1:-suffix_length]
        return {"input": tokenizer.decode(nobos)}
    else:
        # For instruction tuning
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["input"]},
        ]
        nobos = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )[1:]
        formatted_input = tokenizer.decode(nobos)

        # Handle output
        suffix_str = tokenizer.decode([tokenizer.eos_token_id])
        if output_length and len(tokenizer.tokenize(example["output"])) < output_length:
            formatted_output = example["output"] + suffix_str
        else:
            formatted_output = example["output"]

        return {"input": formatted_input, "output": formatted_output}


def apply_chat_template_generic(
    example, tokenizer, suffix_length, binarize=False, output_length=None
):
    """Apply chat template for generic models"""
    if binarize:
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        nobos = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )[1:-suffix_length]
        return {"input": tokenizer.decode(nobos)}
    else:
        # For instruction tuning
        messages = [{"role": "user", "content": example["input"]}]
        nobos = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )[1:]
        return {"input": tokenizer.decode(nobos), "output": example["output"]}


def apply_no_chat_template(example, tokenizer, binarize=False):
    """Apply no chat template for non-chat models"""
    if binarize:
        return {"input": example["input"] + example["output"]}
    else:
        return {"input": example["input"], "output": example["output"]}


def process_dataset_for_training(
    dataset,
    tokenizer,
    model_name,
    binarize=False,
    train_on_negative=False,
    output_length=None,
    max_num_of_examples=None,
    negative_example_ratio=1,
    replace_negative_description=True,
    is_chat_model=False,
):
    """
    Process HuggingFace dataset for training using map functions.

    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer instance
        model_name: Name of the model
        binarize: Whether to binarize the dataset
        train_on_negative: Whether to include negative examples in training
        output_length: Output length for chat models
        max_num_of_examples: Maximum number of examples per concept
        negative_example_ratio: Ratio of negative to positive examples
        replace_negative_description: Whether to replace negative descriptions

    Returns:
        Processed HuggingFace Dataset ready for training
    """
    suffix_length, suffix_str = get_suffix_length(tokenizer)

    # Filter positive and negative examples
    positive_dataset = dataset.filter(
        lambda x: x["output_concept"] != EMPTY_CONCEPT and x["category"] == "positive"
    )
    negative_dataset = dataset.filter(
        lambda x: x["output_concept"] == EMPTY_CONCEPT and x["category"] == "negative"
    )

    # Limit examples if specified
    if max_num_of_examples and max_num_of_examples > 0:
        positive_dataset = positive_dataset.select(
            range(min(max_num_of_examples // 2, len(positive_dataset)))
        )
        negative_dataset = negative_dataset.select(
            range(min(max_num_of_examples // 2, len(negative_dataset)))
        )

    # Handle negative example ratio
    if negative_example_ratio is not None and len(negative_dataset) > 0:
        # Group positive examples by concept_id
        concept_ids = list(set(positive_dataset["concept_id"]))

        processed_negative_examples = []
        for concept_id in concept_ids:
            concept_positive = positive_dataset.filter(
                lambda x: x["concept_id"] == concept_id
            )
            if len(concept_positive) == 0:
                continue

            positive_example_per_concept = len(concept_positive)
            negative_example_per_concept = int(
                positive_example_per_concept * negative_example_ratio
            )

            # Sample negative examples for this concept
            if negative_example_per_concept > 0 and len(negative_dataset) > 0:
                # Get the concept description from positive examples
                concept_description = concept_positive[0]["output_concept"]

                # Sample negative examples
                sampled_indices = list(
                    range(min(negative_example_per_concept, len(negative_dataset)))
                )
                concept_negative = negative_dataset.select(sampled_indices)

                # Replace description if needed
                if replace_negative_description:

                    def update_negative_example(example):
                        example["output_concept"] = concept_description
                        example["concept_id"] = concept_id
                        return example

                    concept_negative = concept_negative.map(update_negative_example)

                processed_negative_examples.append(concept_negative)

        # Combine all negative examples
        if processed_negative_examples:
            negative_dataset = concatenate_datasets(processed_negative_examples)

    # Combine datasets based on training mode
    if train_on_negative and len(negative_dataset) > 0:
        combined_dataset = concatenate_datasets([positive_dataset, negative_dataset])
    else:
        combined_dataset = positive_dataset

    # Apply chat templates
    if is_chat_model:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":

            def template_fn(x):
                return apply_chat_template_llama(
                    x, tokenizer, suffix_length, binarize, output_length
                )
        else:

            def template_fn(x):
                return apply_chat_template_generic(
                    x, tokenizer, suffix_length, binarize, output_length
                )
    else:

        def template_fn(x):
            return apply_no_chat_template(x, tokenizer, binarize)

    # Apply the template function
    processed_dataset = combined_dataset.map(template_fn, num_proc=4)

    # Add labels for binarized datasets
    if binarize:

        def add_labels(example):
            if example.get("category") == "positive":
                example["labels"] = 1
            else:
                example["labels"] = 0
            return example

        processed_dataset = processed_dataset.map(add_labels)

    return processed_dataset


def get_seed_sentences_dataset():
    """
    Returns a DatasetDict with text, math, and code seed sentences (train/test splits).
    """
    # text data
    wikisum_ds = load_dataset("zhengxuanzenwu/wikitext-2-split-128")
    text_train = [ex["text"] for ex in wikisum_ds["train"]]
    ag_ds = load_dataset("fancyzhx/ag_news")
    text_test = [ex["text"] for ex in ag_ds["train"]]
    # math data
    gsm_ds = load_dataset("openai/gsm8k", "main")
    math_train = [ex["answer"] for ex in gsm_ds["train"]]
    comp_ds = load_dataset("qwedsacf/competition_math")
    math_test = [ex["problem"] for ex in comp_ds["train"]]
    # code data
    code_ds = load_dataset("christopher/rosetta-code")
    code_all = [ex["code"][:500] for ex in code_ds["train"] if len(ex["code"]) > 500]
    code_train = code_all[: len(code_all) // 2]
    code_test = code_all[len(code_all) // 2 :]
    data = {
        "text_train": text_train[:1000],
        "text_test": text_test[:1000],
        "math_train": math_train[:1000],
        "math_test": math_test[:1000],
        "code_train": code_train[:1000],
        "code_test": code_test[:1000],
    }
    return DatasetDict(
        {
            "text_train": Dataset.from_dict({"input": data["text_train"]}),
            "text_test": Dataset.from_dict({"input": data["text_test"]}),
            "math_train": Dataset.from_dict({"input": data["math_train"]}),
            "math_test": Dataset.from_dict({"input": data["math_test"]}),
            "code_train": Dataset.from_dict({"input": data["code_train"]}),
            "code_test": Dataset.from_dict({"input": data["code_test"]}),
        }
    )


def get_seed_instructions_dataset():
    """
    Returns a DatasetDict with text, math, and code seed instructions (train/test splits).
    """
    # text instructions
    dolly_ds = load_dataset("databricks/databricks-dolly-15k")
    text_train = [
        ex["instruction"]
        for ex in dolly_ds["train"]
        if ex["category"] == "open_qa"
        and ex["context"] == ""
        and len(ex["instruction"]) < 500
    ]
    # math instructions
    gsm_ds = load_dataset("openai/gsm8k", "main")
    math_train = [ex["question"] for ex in gsm_ds["train"] if len(ex["question"]) < 500]
    # code instructions
    alpaca_ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
    code_train = [
        ex["instruction"]
        for ex in alpaca_ds["train"]
        if ex["input"] == "" and len(ex["instruction"]) < 500
    ]
    data = {
        "text_train": text_train[:1000],
        "math_train": math_train[:1000],
        "code_train": code_train[:1000],
        "text_test": text_train[1000:2000],
        "math_test": math_train[1000:2000],
        "code_test": code_train[1000:2000],
    }
    return DatasetDict(
        {
            "text_train": Dataset.from_dict({"input": data["text_train"]}),
            "text_test": Dataset.from_dict({"input": data["text_test"]}),
            "math_train": Dataset.from_dict({"input": data["math_train"]}),
            "math_test": Dataset.from_dict({"input": data["math_test"]}),
            "code_train": Dataset.from_dict({"input": data["code_train"]}),
            "code_test": Dataset.from_dict({"input": data["code_test"]}),
        }
    )


@register_factory("axbench")
class AxbenchDatasetFactory(BaseDatasetFactory):
    """AxBench dataset factory for generating HuggingFace datasets for training and evaluation"""

    def __init__(
        self,
        model=None,
        client=None,
        tokenizer=None,
        dataset_category="instruction",
        num_of_examples=1000,
        output_length=32,
        dump_dir=None,
        use_cache=True,
        start_concept_id=0,
        is_chat_model=True,
        include_system_prompt=False,
        has_prompt_steering=False,
        master_data_dir=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.use_cache = use_cache
        self.dataset_category = dataset_category
        self.num_of_examples = num_of_examples
        self.output_length = output_length
        self.include_system_prompt = include_system_prompt
        self.is_chat_model = is_chat_model
        self.seed = kwargs.get("seed", 42)
        self.logger = kwargs.get("logger", logger)
        self.lm_model = None
        self.has_prompt_steering = has_prompt_steering or kwargs.get(
            "has_prompt_steering", False
        )
        self.master_data_dir = master_data_dir or kwargs.get("master_data_dir", None)
        if client is not None:
            lm_model = kwargs.get("lm_model", "gpt-4o-mini")
            self.lm_model = LanguageModel(
                lm_model,
                client,
                dump_dir,
                use_cache=use_cache,
                master_data_dir=self.master_data_dir,
            )
        # Optionally load pregenerated data
        self.overwrite_inference_data_dir = kwargs.get(
            "overwrite_inference_data_dir", None
        )
        if self.overwrite_inference_data_dir is not None and os.path.exists(
            self.overwrite_inference_data_dir
        ):
            self.pregenerated_inference_df = pd.read_parquet(
                os.path.join(
                    self.overwrite_inference_data_dir, "latent_eval_data.parquet"
                )
            )
            self.logger.warning(
                f"Loaded pre-generated data from {self.overwrite_inference_data_dir}."
            )
        # Load seed sentences and instructions
        self.seed_sentences = get_seed_sentences_dataset()
        self.seed_instructions = get_seed_instructions_dataset()

    async def _get_steering_prompts(self, concepts):
        # Use the LanguageModel to generate steering prompts for each concept
        prompts = [T_GENERATE_STEERING_PROMPT % (concept) for concept in concepts]
        completions = await self.lm_model.chat_completions(
            api_names=[self.lm_model.model] * len(prompts),
            prompts=prompts,
            batch_size=8,
        )
        return [c.strip() for c in completions]

    def create_training_ds(
        self,
        dataset_name,
        data_files=None,
        split="train",
        cache_dir=None,
        binarize=False,
        train_on_negative=False,
        output_length=None,
        max_num_of_examples=None,
        negative_example_ratio=1,
        replace_negative_description=True,
        select_concept_ids=None,
        max_concepts=None,
        **kwargs,
    ):
        """
        Load and process AxBench training dataset from HuggingFace, returning a HuggingFace dataset.
        """
        logger.info(f"Loading training dataset {dataset_name} with files {data_files}")
        dataset = load_dataset(
            dataset_name,
            data_files=data_files,
            split=split if kwargs.get("use_split", False) else "train",
            cache_dir=cache_dir,
        )
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        if select_concept_ids:
            dataset = dataset.filter(
                lambda x: x["concept_id"] in select_concept_ids, num_proc=4
            )
            logger.info(f"Filtered to {len(dataset)} examples for selected concepts")
        if max_concepts:
            concept_ids = list(set(dataset["concept_id"]))
            concept_ids = [cid for cid in concept_ids if cid >= 0]
            concept_ids.sort()
            # set: list membership per row is O(n_concepts), slow at 16k concepts x 1M rows
            limited_concept_ids = set(concept_ids[:max_concepts])
            dataset = dataset.filter(
                lambda x: x["concept_id"] in limited_concept_ids, num_proc=4
            )
            logger.info(
                f"Limited to {len(limited_concept_ids)} concepts with {len(dataset)} examples"
            )
        if self.tokenizer is not None and self.model is not None:
            dataset = process_dataset_for_training(
                dataset=dataset,
                tokenizer=self.tokenizer,
                model_name=self.model,
                binarize=binarize,
                train_on_negative=train_on_negative,
                output_length=output_length or self.output_length,
                max_num_of_examples=max_num_of_examples or self.num_of_examples,
                negative_example_ratio=negative_example_ratio,
                replace_negative_description=replace_negative_description,
                is_chat_model=self.is_chat_model,
            )
            logger.info(
                f"Processed dataset with {len(dataset)} examples ready for training"
            )
        return dataset

    def create_eval_ds(
        self,
        concepts,
        subset_n=1,
        steering_factors=None,
        steering_datasets=None,
        steering_model_name=None,
        **kwargs,
    ):
        """
        Generate a HuggingFace Dataset for steering evaluation, for all provided concepts, factors, and dataset types.
        Args:
            concepts: List of concept names/strings to evaluate
            subset_n: Number of prompts/examples per concept
            steering_factors: List of steering factors to use
            steering_datasets: List of steering dataset types (e.g., 'OUATPrefix', 'AlpacaEval', ...)
            steering_model_name: (Optional) Model name for formatting
        Returns:
            HuggingFace Dataset containing all evaluation examples
        """
        assert concepts is not None, "concepts must be provided"
        assert steering_factors is not None, "steering_factors must be provided"
        assert steering_datasets is not None, "steering_datasets must be provided"
        all_datasets = []
        for dataset_name in steering_datasets:
            if dataset_name == "OUATPrefix":
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        for factor in steering_factors:
                            all_examples.append(
                                {
                                    "dataset_name": dataset_name,
                                    "concept_id": idx,
                                    "input_concept": concept,
                                    "input_id": i,
                                    "factor": factor,
                                    "input": "Once upon a time, there was a ",
                                }
                            )
                all_datasets.append(
                    Dataset.from_dict(
                        {k: [ex[k] for ex in all_examples] for k in all_examples[0]}
                    )
                )
            elif dataset_name == "AlpacaEval":
                alpaca_eval_df = load_dataset(
                    "tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True
                ).to_pandas()
                if self.has_prompt_steering and self.lm_model is not None:
                    steering_prompts = asyncio.run(self._get_steering_prompts(concepts))
                else:
                    steering_prompts = [
                        T_PROMPT_STEERING % (concept) for concept in concepts
                    ]
                all_examples = []
                for idx, concept in enumerate(concepts):
                    sampled_prompts = alpaca_eval_df.sample(
                        subset_n, random_state=int(idx)
                    )["instruction"].tolist()
                    for i in range(subset_n):
                        sampled_prompt = sampled_prompts[i]
                        steering_prompt = (
                            steering_prompts[idx]
                            if steering_prompts[idx] != ""
                            else T_PROMPT_STEERING % (concept)
                        )
                        steered_prompt = (
                            f" {steering_prompt}\n\nQuestion: {sampled_prompt}"
                        )
                        if steering_model_name == "meta-llama/Llama-3.1-8B-Instruct":
                            formatted_steered_prompt = (
                                self.tokenizer.apply_chat_template(
                                    [
                                        {
                                            "role": "system",
                                            "content": "You are a helpful assistant.",
                                        },
                                        {"role": "user", "content": steered_prompt},
                                    ],
                                    tokenize=True,
                                    add_generation_prompt=True,
                                )[1:]
                            )
                            formatted_steered_prompt = self.tokenizer.decode(
                                formatted_steered_prompt
                            )
                            formatted_prompt = self.tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant.",
                                    },
                                    {"role": "user", "content": sampled_prompt},
                                ],
                                tokenize=True,
                                add_generation_prompt=True,
                            )[1:]
                            formatted_prompt = self.tokenizer.decode(formatted_prompt)
                        else:
                            formatted_steered_prompt = (
                                self.tokenizer.apply_chat_template(
                                    [{"role": "user", "content": steered_prompt}],
                                    tokenize=True,
                                    add_generation_prompt=True,
                                )[1:]
                            )
                            formatted_steered_prompt = self.tokenizer.decode(
                                formatted_steered_prompt
                            )
                            formatted_prompt = self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": sampled_prompt}],
                                tokenize=True,
                                add_generation_prompt=True,
                            )[1:]
                            formatted_prompt = self.tokenizer.decode(formatted_prompt)
                        for factor in steering_factors:
                            all_examples.append(
                                {
                                    "dataset_name": dataset_name,
                                    "concept_id": idx,
                                    "input_concept": concept,
                                    "input_id": i,
                                    "factor": factor,
                                    "original_prompt": sampled_prompt,
                                    "steered_input": formatted_steered_prompt,
                                    "input": formatted_prompt,
                                }
                            )
                all_datasets.append(
                    Dataset.from_dict(
                        {k: [ex[k] for ex in all_examples] for k in all_examples[0]}
                    )
                )
            elif dataset_name in ("AlpacaEval_Suppress", "AlpacaEval_Synergy"):
                alpaca_eval_df = load_dataset(
                    "tatsu-lab/alpaca_eval", split="eval", trust_remote_code=True
                ).to_pandas()
                common_steering_factors = steering_factors
                if dataset_name == "AlpacaEval_Suppress":
                    common_steering_factors = [
                        f * -1.0 for f in common_steering_factors
                    ]
                if self.has_prompt_steering and self.lm_model is not None:
                    steering_prompts = asyncio.run(self._get_steering_prompts(concepts))
                else:
                    steering_prompts = [
                        T_PROMPT_STEERING % (concept) for concept in concepts
                    ]
                all_examples = []
                for idx, concept in enumerate(concepts):
                    for i in range(subset_n):
                        sampled_prompt = alpaca_eval_df.sample(1)[
                            "instruction"
                        ].tolist()[0]
                        steering_prompt = (
                            steering_prompts[idx]
                            if steering_prompts[idx] != ""
                            else T_PROMPT_STEERING % (concept)
                        )
                        steered_prompt = (
                            f" {steering_prompt}\n\nQuestion: {sampled_prompt}"
                        )
                        formatted_steered_prompt = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": steered_prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for factor in common_steering_factors:
                            all_examples.append(
                                {
                                    "dataset_name": dataset_name,
                                    "concept_id": idx,
                                    "input_concept": concept,
                                    "input_id": i,
                                    "factor": factor,
                                    "original_prompt": sampled_prompt,
                                    "input": formatted_steered_prompt,
                                }
                            )
                all_datasets.append(
                    Dataset.from_dict(
                        {k: [ex[k] for ex in all_examples] for k in all_examples[0]}
                    )
                )
            else:
                raise NotImplementedError(
                    f"Steering dataset {dataset_name} not implemented."
                )
        if len(all_datasets) == 1:
            return all_datasets[0]
        return concatenate_datasets(all_datasets)

    def get_concept_info(
        self,
        dataset_name,
        split="train",
        data_files=None,
        cache_dir=None,
        select_concept_ids=None,
        max_concepts=None,
        **kwargs,
    ):
        """
        Load the AxBench dataset for the specified split and return concept info dicts.
        Args:
            split: "train" or "eval"
            data_files: Optional data files dict
            cache_dir: Optional cache dir
            select_concept_ids: Optional list of concept IDs to filter
            max_concepts: Optional max number of concepts
        Returns:
            List of dicts: {concept_id, concept, ref, concept_genres_map}
        """
        logger.info(f"Loading concept info from split={split}, data_files={data_files}")
        dataset = load_dataset(
            dataset_name,
            data_files=data_files,
            split=split,
            cache_dir=cache_dir,
        )
        logger.info(f"Loaded dataset with {len(dataset)} examples for concept info")
        if select_concept_ids:
            dataset = dataset.filter(
                lambda x: x["concept_id"] in select_concept_ids, num_proc=4
            )
            logger.info(f"Filtered to {len(dataset)} examples for selected concepts")
        if max_concepts:
            concept_ids = list(set(dataset["concept_id"]))
            concept_ids = [cid for cid in concept_ids if cid >= 0]
            concept_ids.sort()
            # set: list membership per row is O(n_concepts), slow at 16k concepts x 1M rows
            limited_concept_ids = set(concept_ids[:max_concepts])
            dataset = dataset.filter(
                lambda x: x["concept_id"] in limited_concept_ids, num_proc=4
            )
            logger.info(
                f"Limited to {len(limited_concept_ids)} concepts with {len(dataset)} examples"
            )
        df = dataset.to_pandas()
        concept_info = []
        unique_concepts = df.groupby("concept_id").first()
        for concept_id, row in unique_concepts.iterrows():
            if concept_id >= 0:
                concept_info.append(
                    {
                        "concept_id": concept_id,
                        "concept": row.get("output_concept", f"concept_{concept_id}"),
                        "ref": f"https://neuronpedia.org/api/feature/{concept_id}",
                        "concept_genres_map": {
                            row.get("output_concept", f"concept_{concept_id}"): ["text"]
                        },
                    }
                )
        return concept_info
