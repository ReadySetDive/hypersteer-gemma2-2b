import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import datasets
import torch
import transformers

from hypersteer.data.base import get_dataset_factory
from hypersteer.utils.configs import ExperimentConfig


def load_dataset_for_inference(args: ExperimentConfig):
    """
    Load HuggingFace dataset for inference and extract concept information.
    """

    # Load the dataset using the same function as training
    factory = get_dataset_factory(
        args.dataset.dataset_type,
        tokenizer=args.dataset.eval.tokenizer
        if hasattr(args.dataset.eval, "tokenizer")
        else None,
        dump_dir=args.dataset.eval.cache_dir
        if hasattr(args.dataset.eval, "cache_dir")
        else None,
    )
    dataset = factory.create_eval_ds(
        dataset_name=args.dataset.eval.hf_dataset_name,
        data_files=args.dataset.eval.hf_data_files,
        split=args.dataset.eval.hf_split,
        cache_dir=args.dataset.eval.cache_dir,
        select_concept_ids=args.dataset.eval.select_concept_ids,
        max_concepts=args.dataset.eval.max_concepts,
        master_data_dir=args.dataset.master_data_dir,
    )

    # Extract unique concept information from the dataset
    df = dataset.to_pandas()
    concept_info = []

    # Get unique concepts with their IDs
    unique_concepts = df.groupby("concept_id").first()

    for concept_id, row in unique_concepts.iterrows():
        if concept_id >= 0:  # Skip negative concept IDs
            concept_info.append(
                {
                    "concept_id": concept_id,
                    "concept": row.get("input_concept", f"concept_{concept_id}"),
                    "ref": f"https://neuronpedia.org/api/feature/{concept_id}",  # Default SAE link format
                    "concept_genres_map": {
                        row.get("input_concept", f"concept_{concept_id}"): ["text"]
                    },
                }
            )

    return concept_info


def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = (
            [i for i in range(first_n)]
            + [i for i in range(last_position - last_n, last_position)]
            + [pad_position for _ in range(pad_amount)]
        )
        intervention_locations = [position_list] * num_interventions
    else:
        left_pad_amount = _first_n - first_n
        right_pad_amount = _last_n - last_n
        left_intervention_locations = [i for i in range(first_n)] + [
            pad_position for _ in range(left_pad_amount)
        ]
        right_intervention_locations = [
            i for i in range(last_position - last_n, last_position)
        ] + [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [
                pad_position for _ in range(left_len - right_len)
            ]
        else:
            left_intervention_locations += [
                pad_position for _ in range(right_len - left_len)
            ]
        intervention_locations = [left_intervention_locations] * (
            num_interventions // 2
        ) + [right_intervention_locations] * (num_interventions // 2)

    return intervention_locations


@dataclass
class InterventionDataCollator:
    """Collate examples for Intervention."""

    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator  # type: ignore
    include_concept: bool = False
    concept_tokenizer: transformers.AutoTokenizer = None
    max_seq_length: int = None

    def __call__(
        self, instances: Sequence[dict] | dict[Sequence]
    ) -> dict[str, torch.Tensor]:
        if isinstance(instances, dict):
            # transpose
            instances = [
                dict(zip(instances.keys(), values))
                for values in zip(*instances.values())
            ]

        max_intervention_len = max(
            [len(inst["intervention_locations"][0]) for inst in instances]
        )
        max_seq_len = max([len(inst["input_ids"]) for inst in instances])

        if self.max_seq_length is not None:
            max_seq_len = max(max_seq_len, self.max_seq_length)
            if max_seq_len > self.max_seq_length:
                warnings.warn(
                    f"Max sequence length {max_seq_len} is greater than the provided bound {self.max_seq_length}. Should specify a higher value."
                )

            # This accounts for the bos token
            max_seq_len -= 1

        max_concept_seq_len = max(
            [len(inst["concept_input_ids"]) for inst in instances]
        )

        for inst in instances:
            non_pad_len = len(inst["input_ids"])
            concept_non_pad_len = len(inst["concept_input_ids"])

            _intervention_mask = torch.ones_like(inst["intervention_locations"][0])
            _intervention_location_paddings = torch.tensor(
                [
                    [
                        len(inst["input_ids"])
                        for _ in range(
                            max_intervention_len
                            - len(inst["intervention_locations"][0])
                        )
                    ]
                ]
            )
            _intervention_mask_paddings = torch.tensor(
                [
                    0
                    for _ in range(
                        max_intervention_len - len(inst["intervention_locations"][0])
                    )
                ]
            )
            inst["intervention_locations"] = torch.cat(
                [inst["intervention_locations"], _intervention_location_paddings],
                dim=-1,
            ).int()
            inst["intervention_masks"] = torch.cat(
                [_intervention_mask, _intervention_mask_paddings], dim=-1
            ).int()
            inst["prompt_intervention_masks"] = inst["intervention_masks"].clone()
            inst["prompt_intervention_masks"][inst["prompt_lengths"] :] = (
                0  # mask out the intervention locations after prompt length
            )

            _input_id_paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(max_seq_len - non_pad_len)]
            )
            inst["input_ids"] = torch.cat(
                (
                    inst["input_ids"],
                    torch.tensor([self.tokenizer.pad_token_id]),
                    _input_id_paddings,
                )
            ).int()

            _concept_input_id_paddings = torch.tensor(
                [
                    self.concept_tokenizer.pad_token_id
                    for _ in range(max_concept_seq_len - concept_non_pad_len)
                ]
            )
            inst["concept_input_ids"] = torch.cat(
                (_concept_input_id_paddings, inst["concept_input_ids"])
            ).int()

            _label_paddings = torch.tensor(
                [-100 for _ in range(max_seq_len - non_pad_len + 1)]
            )
            inst["labels"] = torch.cat((inst["labels"], _label_paddings))

            inst["attention_mask"] = (
                inst["input_ids"] != self.tokenizer.pad_token_id
            ).int()
            inst["concept_attention_mask"] = (
                inst["concept_input_ids"] != self.concept_tokenizer.pad_token_id
            ).int()

            # hypernetwork only sees prompt
            inst["hypernet_input_mask"] = inst["attention_mask"].clone()
            inst["hypernet_input_mask"][inst["prompt_lengths"] :] = 0

        batch_inputs = self.data_collator(instances)
        return batch_inputs


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    df,
    dataset_category="continuation",
    positions="all",  # "all_prompt" or "all" or "f1+l1" (pyreft formatting)
    exclude_bos=True,
    prefix_length=1,
    include_concept=False,
    input_condition_concept=False,
    concept_tokenizer=None,
    max_seq_length=None,
    **kwargs,
):
    """Make dataset and collator for supervised fine-tuning with kl div loss."""
    if not exclude_bos:
        prefix_length = 0

    concept_tokenizer = (
        concept_tokenizer if concept_tokenizer is not None else tokenizer
    )

    (
        all_base_input_ids,
        all_intervention_locations,
        all_output_ids,
        all_concept_input_ids,
    ) = [], [], [], []
    all_prompt_lengths = []
    all_concept_ids = []

    for _, row in df.iterrows():
        _concept, _input, _output = row["output_concept"], row["input"], row["output"]
        all_concept_ids.append(row["concept_id"])

        # prepare input ids
        base_prompt = _input
        if isinstance(_output, float):
            _output = tokenizer.eos_token
        base_input = base_prompt + _output
        base_prompt_ids = tokenizer(
            base_prompt, max_length=1024, truncation=True, return_tensors="pt"
        )["input_ids"][0]
        base_input_ids = tokenizer(
            base_input, max_length=1024, truncation=True, return_tensors="pt"
        )["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        base_length = len(base_input_ids)

        # output ids with prompt token mask
        output_ids = base_input_ids.clone()
        output_ids[:base_prompt_length] = -100

        if input_condition_concept:
            # We assume special tokens are added to tokenizer
            _concept = f"<concept>{_concept}</concept><input>{_input}</input>"

        concept_input_ids = concept_tokenizer(
            _concept, max_length=1024, truncation=True, return_tensors="pt"
        )["input_ids"][0]

        if positions is None or positions == "all_prompt":
            intervention_locations = torch.tensor(
                [[i for i in range(prefix_length, base_prompt_length)]]
            )
        elif positions == "all":
            intervention_locations = torch.tensor(
                [[i for i in range(prefix_length, base_length)]]
            )
        elif isinstance(positions, str) and positions.startswith("s"):
            # Support for skip mode: sK means intervene every K tokens
            try:
                skip_k = int(positions[1:])
                if skip_k < 1:
                    raise ValueError("Skip value must be >= 1")
            except Exception as e:
                raise ValueError(
                    f"Invalid skip mode positions string: {positions}"
                ) from e
            # Intervene every skip_k tokens in the prompt region
            intervention_locations = torch.tensor(
                [
                    [
                        i
                        for i in range(prefix_length, base_length)
                        if (i - prefix_length) % skip_k == 0
                    ]
                ]
            )
        else:
            first_n, last_n = parse_positions(positions)
            intervention_locations = get_intervention_locations(
                last_position=base_length - prefix_length,
                first_n=first_n,
                last_n=last_n,
                pad_mode="last",
                num_interventions=1,
                share_weights=True,
            )
            # shift intervention locations by prefix length
            shifted_intervention_locations = [
                [loc + prefix_length for loc in intervention_locations[0]]
            ]
            intervention_locations = shifted_intervention_locations

        all_concept_input_ids.append(concept_input_ids)
        all_intervention_locations.append(intervention_locations)
        all_base_input_ids.append(base_input_ids)
        all_output_ids.append(output_ids)
        all_prompt_lengths.append(
            torch.tensor(base_prompt_length - 1)
        )  # exclude bos token

    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": all_base_input_ids,
            "intervention_locations": all_intervention_locations,
            "labels": all_output_ids,
            "prompt_lengths": all_prompt_lengths,
            "concept_input_ids": all_concept_input_ids,
            "concept_ids": torch.tensor(all_concept_ids),
        }
    )
    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "intervention_locations",
            "prompt_lengths",
            "labels",
            "concept_input_ids",
            "concept_ids",
        ],
    )

    data_collator_fn = transformers.DefaultDataCollator(return_tensors="pt")
    data_collator = InterventionDataCollator(
        tokenizer=tokenizer,
        data_collator=data_collator_fn,
        include_concept=include_concept,
        concept_tokenizer=concept_tokenizer,
        max_seq_length=max_seq_length,
    )

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def get_batch_locs(**kwargs):
    # Process each input in the batch to get intervention locations
    batch_intervention_locations = []
    attention_mask = kwargs.get("attention_mask", None)

    for idx, input_text in enumerate(kwargs["inputs"]):
        # Tokenize to get the prompt length for this specific input
        input_ids = kwargs["tokenizer"](
            input_text, return_tensors="pt", add_special_tokens=True
        )["input_ids"][0]
        base_prompt_length = len(input_ids)

        # Calculate offset based on attention mask if provided
        offset = kwargs["prefix_length"]
        if attention_mask is not None:
            # Count padding tokens at the beginning (left padding)
            mask = attention_mask[idx]
            pad_count = 0
            for i in range(len(mask)):
                if mask[i] == 0:
                    pad_count += 1
                else:
                    break
            offset += pad_count

        positions = kwargs.get("intervention_positions", "all")
        # Parse positions and get intervention locations for this input
        if positions is None or positions == "all":
            intervention_locations = torch.tensor(
                [[i for i in range(offset, base_prompt_length + offset)]]
            )
        else:
            first_n, last_n = parse_positions(positions)
            intervention_locations = get_intervention_locations(
                last_position=base_prompt_length,
                first_n=first_n,
                last_n=last_n,
                pad_mode="last",
                num_interventions=1,
                share_weights=True,
            )
            # shift intervention locations by offset
            shifted_intervention_locations = [
                [loc + offset for loc in intervention_locations[0]]
            ]
            intervention_locations = shifted_intervention_locations
        batch_intervention_locations.append(intervention_locations[0])

    # Convert to tensor with proper padding
    max_len = max(len(locs) for locs in batch_intervention_locations)
    padded_locs = []
    for locs in batch_intervention_locations:
        padded = torch.cat(
            [
                locs,
                torch.full((max_len - len(locs),), max(locs), dtype=torch.long),
            ]
        )
        padded_locs.append(padded)

    return torch.stack(padded_locs)
