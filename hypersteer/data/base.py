class DatasetFactoryRegistry:
    """Simple registry for dataset factory types"""

    _factories: dict[str, type["BaseDatasetFactory"]] = {}

    @classmethod
    def register(cls, name: str, factory_class: type["BaseDatasetFactory"]):
        """Register a dataset factory class"""
        cls._factories[name] = factory_class

    @classmethod
    def get_factory(cls, name: str) -> type["BaseDatasetFactory"]:
        """Get a registered factory class"""
        if name not in cls._factories:
            raise ValueError(
                f"Unknown dataset factory type: {name}. Available: {list(cls._factories.keys())}"
            )
        return cls._factories[name]

    @classmethod
    def list_factories(cls) -> list:
        """List all registered factory names"""
        return list(cls._factories.keys())


class BaseDatasetFactory:
    """Base class for dataset factories (no abstract methods)"""

    def __init__(self, **kwargs):
        """Initialize the dataset factory with common parameters"""
        pass


class BaseSteeringDatasetFactory:
    """Base class for steering dataset factories (no abstract methods)"""

    def __init__(self, **kwargs):
        """Initialize the steering dataset factory"""
        pass


def register_factory(name: str):
    """Decorator to register a dataset factory"""

    def decorator(cls):
        DatasetFactoryRegistry.register(name, cls)
        return cls

    return decorator


def get_dataset_factory(factory_type: str, **kwargs):
    """
    Get a dataset factory instance by type.

    Args:
        factory_type: The type of factory to create (e.g., 'axbench')
        **kwargs: Arguments to pass to the factory constructor

    Returns:
        An instance of the requested dataset factory
    """
    factory_class = DatasetFactoryRegistry.get_factory(factory_type)
    return factory_class(**kwargs)


def get_steering_dataset_factory(factory_type: str, **kwargs):
    """
    Get a steering dataset factory instance by type.

    Args:
        factory_type: The type of factory to create (e.g., 'axbench')
        **kwargs: Arguments to pass to the factory constructor

    Returns:
        An instance of the requested steering dataset factory
    """
    # Automatically append '_steering' suffix if not present
    if not factory_type.endswith("_steering"):
        factory_type = f"{factory_type}_steering"

    factory_class = DatasetFactoryRegistry.get_factory(factory_type)
    return factory_class(**kwargs)


def list_available_factories():
    """
    List all available factory types.

    Returns:
        List of available factory type names
    """
    return DatasetFactoryRegistry.list_factories()
