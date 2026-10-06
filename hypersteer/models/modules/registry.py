from collections.abc import Callable
from typing import Generic, TypeVar

from hypersteer.models.base import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelRegistry(Generic[T]):
    """Simple registry for model types"""

    _models: dict[str, type[T]] = {}

    @classmethod
    def register(cls, name: str, model_class: type[T]):
        """Register a model class"""
        cls._models[name] = model_class

    @classmethod
    def get_model(cls, name: str) -> type[T]:
        """Get a registered model class"""
        if name not in cls._models:
            raise ValueError(
                f"Unknown model type: {name}. Available: {list(cls._models.keys())}"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names"""
        return list(cls._models.keys())


def get_model(model_type: str, **kwargs) -> BaseModel:
    """
    Get a model instance by type.

    Args:
        model_type: The type of model to create (e.g., 'HyperSteer', 'PromptSteering')
        **kwargs: Arguments to pass to the model constructor

    Returns:
        An instance of the requested model
    """
    model_class = ModelRegistry.get_model(model_type)
    return model_class(**kwargs)


def list_available_models() -> list[str]:
    """
    List all available model types.

    Returns:
        List of available model type names
    """
    return ModelRegistry.list_models()


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a model"""

    def decorator(cls):
        ModelRegistry.register(name, cls)
        return cls

    return decorator
