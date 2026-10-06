# Dataset Factory Registry System

This module provides an abstract base class system with a simple registry for dataset factories. This allows you to easily switch between different dataset implementations and add new ones without modifying existing code.

## Architecture

- `base.py`: Contains abstract base classes and the registry system
- `axbench.py`: Contains the AxBench dataset factory implementation (the original implementation)
- `dataset.py`: Main entry point with backward compatibility and registry functions

## Usage

### Using the Registry System

```python
from hypersteer.data import get_dataset_factory, get_steering_dataset_factory, list_available_factories

# List all available factory types
print(list_available_factories())  # ['axbench', 'axbench_steering']

# Get a dataset factory by type
dataset_factory = get_dataset_factory(
    "axbench",
    model=model,
    client=client,
    tokenizer=tokenizer,
    dataset_category="instruction",
    num_of_examples=1000,
    output_length=32,
    dump_dir="./output",
    master_data_dir="./data"
)

# Get a steering dataset factory
steering_factory = get_steering_dataset_factory(
    "axbench",  # Will automatically append '_steering'
    tokenizer=tokenizer,
    dump_dir="./output",
    has_prompt_steering=True,
    master_data_dir="./data"
)
```

## Adding New Dataset Factory Types

To add a new dataset factory type:

1. Create a new file (e.g., `my_dataset.py`) in the `hypersteer/data/` directory
2. Import the base classes and registry decorator:
   ```python
   from .base import BaseDatasetFactory, BaseSteeringDatasetFactory, register_factory
   ```
3. Implement your factory classes:
   ```python
   @register_factory("my_dataset")
   class MyDatasetFactory(BaseDatasetFactory):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Your initialization code
       
       def save_cache(self):
           # Implementation
           pass
       
       def reset_stats(self):
           # Implementation
           pass
       
       # Implement all other abstract methods...
   
   @register_factory("my_dataset_steering")
   class MySteeringDatasetFactory(BaseSteeringDatasetFactory):
       # Implementation
       pass
   ```
4. Import your module in `dataset.py` to register the factories:
   ```python
   from . import my_dataset  # This will register your factories
   ```

## Available Factory Types

- `axbench`: The original AxBench dataset factory implementation
- `axbench_steering`: The original AxBench steering dataset factory implementation

## Abstract Methods

### BaseDatasetFactory

All dataset factories must implement these methods:

- `save_cache()`: Save the language model cache
- `reset_stats()`: Reset API costs
- `prepare_genre_concepts(concepts, **kwargs)`: Prepare genre concepts
- `prepare_concepts(concepts, **kwargs)`: Prepare concepts and contrast concepts
- `create_eval_df(...)`: Create evaluation dataframe
- `create_train_df(...)`: Create training dataframe
- `create_dpo_df(...)`: Create DPO dataframe
- `create_imbalance_eval_df(...)`: Create imbalanced evaluation dataframe (optional)

### BaseSteeringDatasetFactory

All steering dataset factories must implement:

- `create_eval_df(...)`: Create evaluation dataframe for steering
- `augment_train_df_with_steered_prompts(...)`: Augment training dataframe (optional)

## Configuration

You can specify which dataset factory to use in your training configuration:

```python
# In your training script
factory_type = "axbench"  # or any other registered type
dataset_factory = get_dataset_factory(factory_type, **factory_kwargs)
```

This makes it easy to switch between different dataset implementations by just changing the `factory_type` parameter. 