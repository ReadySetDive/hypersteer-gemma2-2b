from pathlib import Path
from typing import TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from hypersteer.utils.helpers import get_logger

# Initialize the logger
logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def config_to_pydantic(cfg: DictConfig, model_class: type[T]) -> T:
    """
    converts a Hydra DictConfig to a Pydantic model for validation.
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return model_class(**config_dict)


def load_experiment_config(
    experiment_name: str, config_path: str = "config"
) -> DictConfig:
    """
    Load and merge experiment configuration manually.

    Args:
        experiment_name: Name of the experiment (e.g., 'base', 'hypersteer')
        config_path: Path to the config directory

    Returns:
        Merged DictConfig with base defaults + experiment overrides
    """
    config_path = Path(config_path)

    # Load base defaults
    base_defaults_path = config_path / "base_defaults.yaml"
    if not base_defaults_path.exists():
        raise FileNotFoundError(
            f"Base defaults config not found at {base_defaults_path}"
        )

    base_config = OmegaConf.load(base_defaults_path)
    logger.info(f"Loaded base defaults from {base_defaults_path}")

    # Load experiment overrides
    experiment_path = config_path / "experiment" / f"{experiment_name}.yaml"
    if not experiment_path.exists():
        logger.warning(
            f"Experiment config not found at {experiment_path}, using base defaults only"
        )
        return base_config

    experiment_config = OmegaConf.load(experiment_path)
    logger.info(f"Loaded experiment overrides from {experiment_path}")

    # Merge configs (experiment overrides base)
    merged_config = OmegaConf.merge(base_config, experiment_config)
    logger.info(
        f"Successfully merged experiment '{experiment_name}' with base defaults"
    )

    return merged_config


class BaseConfigModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class WandbConfig(BaseConfigModel):
    log: bool = True
    log_code: bool = True
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    group: str | None = None
    tags: str | list[str] | None = None
    notes: str | None = None
    watch_grads: bool = False
    watch_grads_freq: int = 100


class DatasetConfig(BaseConfigModel):
    """Dataset arguments"""

    # Top-level select_concept_ids for the dataset (applies globally if not overridden)
    select_concept_ids: list[int] = Field(default_factory=list)

    # Core dataset parameters (for backward compatibility and as defaults)
    dataset_name: str | None = None
    dataset_category: str = "instruction"
    dataset_split: str = "train"
    task_name: str | None = None
    concept_path: str | Path | None = None
    max_concepts: int = 500  # Updated to match config
    num_of_examples: int = 144  # Updated to match config
    filter_length: int = 512
    master_data_dir: str | Path | None = None
    shuffle: bool = True
    dev_size: float = 0.1
    select_column: str = "prompt"
    response_column: str = "completion"
    cache_dir: str | Path | None = None
    train_dir: str | Path | None = None
    held_out_eval: bool = False
    input_condition_concept: bool = False
    max_seq_length: int | None = None
    use_split: bool = False

    # HuggingFace dataset configuration
    hf_dataset_name: str | None = None
    hf_data_files: dict[str, str] | str | None = None
    hf_split: str = "train"
    hf_cache_dir: str | Path | None = None
    hf_revision: str | None = None
    hf_use_auth_token: bool | str | None = None
    hf_streaming: bool = False
    hf_num_proc: int | None = None
    hf_keep_in_memory: bool = False
    hf_download_config: dict | None = None
    hf_download_mode: str | None = None
    hf_verification_mode: str | None = None
    hf_ignore_verifications: bool = False
    hf_save_infos: bool = False
    hf_trust_remote_code: bool = False
    dataset_type: str = "axbench"

    # Train and eval sub-configs (override defaults above if specified)
    train: "DatasetConfig" = None
    eval: "DatasetConfig" = None

    # For pydantic self-referencing
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        # Allow train/eval to be dicts or DatasetConfig
        train = data.pop("train", None)
        eval_ = data.pop("eval", None)
        super().__init__(**data)
        if train is not None:
            self.train = DatasetConfig(**train) if isinstance(train, dict) else train
        if eval_ is not None:
            self.eval = DatasetConfig(**eval_) if isinstance(eval_, dict) else eval_


class FactorSelectionConfig(BaseConfigModel):
    """Configuration for selecting optimal steering factors using Optuna."""

    enable: bool = False
    n_calls: int = 10
    factor_min: float = 0.0
    factor_max: float = 3.0
    metric: str = "lm_judge_rating"  # Which metric to optimize
    model: str = "HyperSteer"
    models: list[str] = Field(
        default_factory=list
    )  # Multiple models for factor selection
    # New parameters for Optuna
    discrete_space: bool = False  # Whether to use discrete values
    discrete_steps: int = 10  # Number of steps for discrete space
    log_scale: bool = False  # Whether to sample in log scale
    n_startup_trials: int = 5  # Number of random trials before TPE kicks in
    capture_all_metrics: bool = False


class VisualizationConfig(BaseConfigModel):
    log_heatmap: bool = True
    log_heatmap_freq: int = 10
    pdf_visualization: bool = False
    png_visualization: bool = False
    log_all_examples: bool = True  # Enable slider to view all examples in batch


class ModelConfig(BaseConfigModel):
    """Configuration for model architecture and model-specific parameters."""

    # Core model parameters
    model_name: str = "HyperSteer"
    target_model_name: str = "google/gemma-2-2b-it"
    base_model_name: str = "google/gemma-2-2b"

    # Architecture parameters
    layer: int = 20
    steering_layers: list[int] | str | None = None
    component: str = "res"
    hypernet_type: str = "regression"
    cross_attn_hidden_layers: int = 8
    low_rank_dimension: int = 1
    topk: int = 8

    # Model behavior parameters
    intervention_positions: str = "all"
    intervention_type: str = "addition"
    exclude_bos: bool = True
    special_tokens: list[str] = Field(default_factory=lambda: [])

    # Model-specific features
    use_synergy: bool = False
    include_sentence_in_embedding: bool = False
    use_selection_head: bool = False
    selection_l1_loss_coeff: float = 1e-3
    use_selection_ln: bool = True
    compute_sparsity_loss: bool = False
    selection_head_start_temperature: float = 1.0
    selection_head_end_temperature: float = 0.05
    selection_head_learnable_temperature: bool = False
    selection_head_anneal_temperature: bool = False
    selection_head_add_gumbel_noise: bool = True
    inference_binarize_mask: bool = False
    selection_head_threshold: float = 0.5
    selection_head_straight_through: bool = False
    temperature_lr: float = 0.1

    # Debugging
    debug_print: bool = False

    # Visualization configs (model-specific)
    logit_diff_visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig
    )
    cross_attn_heatmap_visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig
    )
    mask_visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)


class TrainingArgs(BaseConfigModel):
    """Training procedure arguments and optimization parameters."""

    # Training procedure
    batch_size: int = 16
    test_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    n_epochs: int = 3
    n_steps: int = -1
    val_interval: int = 100
    checkpoint_per_step: int | None = None

    # Optimization parameters
    lr: float = 0.01
    warmup_steps: int = 0
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    max_grad_norm: float = 100.0
    adaptive_row_lr: bool = False

    # Loss configuration
    coeff_latent_l1_loss: float = 0.005
    reconstruction_loss_ratio: float = 0.0
    steering_loss_ratio: float = 1.0

    # Data configuration
    binarize_dataset: bool = False
    train_on_negative: bool = True
    negative_example_ratio: float = 1.0

    # Training environment
    debug: bool = False
    debug_model: bool = False
    save_dir: str | Path = "./saved_models"
    resume_from: str | Path | None = None
    seed: int = 42
    use_bf16: bool = True
    run_eval_suite_at_end: bool = False


class GenerateConfig(BaseConfigModel):
    """Configuration for generate.py"""

    lm_model: str = "gpt-4o-mini"
    output_length: int = 128
    num_of_examples: int = 144
    max_concepts: int = 500
    master_data_dir: str | Path | None = None
    dataset_category: str = "instruction"
    lm_use_cache: bool = False
    seed: int = 42


class InferenceConfig(BaseConfigModel):
    """Configuration for inference.py"""

    use_bf16: bool = True
    mode: str = "all"
    model_name: str = "google/gemma-2-2b-it"
    models: list[str] = Field(
        default_factory=list
    )  # Multiple models to run inference on

    # Steering related params
    steering_intervention_type: str = "addition"
    steering_model_name: str = "google/gemma-2-2b-it"
    steering_datasets: list[str] = Field(default_factory=lambda: ["AlpacaEval"])
    steering_batch_size: int = 8
    steering_output_length: int = 128
    steering_layers: list[int] = Field(default_factory=lambda: [20])
    steering_num_of_examples: int = 10
    steering_factors: list[float] = Field(default_factory=lambda: [1.0])
    ignore_steering_state: bool = False

    master_data_dir: str | Path | None = None
    seed: int = 42
    lm_model: str = "gpt-4o-mini"
    use_cache: bool = True

    # Generation related params
    temperature: float = 0.7

    factor_selection: FactorSelectionConfig = Field(
        default_factory=FactorSelectionConfig
    )

    run_distinct_infers: bool = False
    infer_run: str | None = None
    overwrite_existing_infer: bool = False
    run_eval: bool = False
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    enable_judge_verification: bool = False


class EvalArgs(BaseConfigModel):
    """Evaluation arguments derived from eval_args.py"""

    mode: str = "all"
    models: list[str] = Field(default_factory=list)  # Multiple models to evaluate
    evaluators: list[str] = Field(default_factory=list)
    latent_evaluators: list[str] = Field(
        default_factory=lambda: ["AUCROCEvaluator", "HardNegativeEvaluator"]
    )
    steering_evaluators: list[str] = Field(
        default_factory=lambda: ["PerplexityEvaluator", "LMJudgeEvaluator"]
    )
    top_k_concepts: int = 10
    num_of_workers: int = 32
    is_sanity_run: bool = False
    save_progress: bool = True
    use_cached_data: bool = False
    force_recalculation: bool = False
    lm_model: str = "gpt-4o-mini"
    lm_api_key: str | None = None
    lm_use_cache: bool = True
    lm_temperature: float = 0.0
    lm_top_p: float = 1.0
    winrate_split_ratio: float = 0.5
    enable_progress_bar: bool = True
    run_winrate: bool = False
    winrate_baseline: str = "PromptSteering"
    master_data_dir: str | Path | None = None
    report_to: str | None = None

    run_distinct_evals: bool = False
    overwrite_existing_eval: bool = False
    infer_run: str | None = None
    eval_run: str | None = None


class ExperimentConfig(BaseConfigModel):
    """Main configuration that contains all sub-configurations"""

    debug: bool = False
    seed: int = 42
    dump_dir: str | Path | None = None
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    train: TrainingArgs = Field(default_factory=TrainingArgs)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluate: EvalArgs = Field(default_factory=EvalArgs)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
