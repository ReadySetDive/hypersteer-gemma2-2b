from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def make_model(self, **kwargs):
        pass

    @abstractmethod
    def make_dataloader(self, examples, **kwargs):
        pass

    @abstractmethod
    def train(self, examples, **kwargs):
        pass

    @abstractmethod
    def save(self, dump_dir, **kwargs):
        pass

    @abstractmethod
    def load(self, dump_dir, **kwargs):
        pass

    @abstractmethod
    def predict_steer(self, examples, **kwargs):
        pass

    @abstractmethod
    def predict_step(self, batch_examples, batch_idx, **kwargs):
        """
        Model-specific prediction step for a single batch.

        Args:
            batch_examples: DataFrame slice for the current batch
            batch_idx: Index of the current batch
            **kwargs: Additional arguments

        Returns:
            Dictionary with keys: generations, perplexities, strengths, steering_vectors
        """
        pass

    @abstractmethod
    def get_logits(self, concept_id, k=10):
        pass

    def to(self, device):
        """Optional method for moving model to device"""
        pass

    # InferenceMixin methods with default implementations
    def on_inference_start(self, **kwargs):
        """Called at the start of inference."""
        pass

    def on_inference_end(self, **kwargs):
        """Called at the end of inference."""
        pass

    def on_batch_start(self, batch_idx, **kwargs):
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, batch_idx, **kwargs):
        """Called at the end of each batch."""
        pass
