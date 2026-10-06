"""Some utils for debugging data processing."""

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def compress_sequence(seq, threshold=64, mask_token="-100", pad_token="pad"):
    compressed = []
    count = 0
    last_red_token = None
    for item in seq:
        if item.startswith(RED):
            count += 1
            last_red_token = item
            if count == 1:
                compressed.append(item)
        else:
            if count > threshold:
                token_content = "-100" if "-100" in last_red_token else "pad"
                compressed.append(f"{RED}...({count - 2} {token_content})...{RESET}")
                compressed.append(last_red_token)
            elif count > 1:
                compressed.extend([last_red_token] * (count - 1))
            count = 0
            compressed.append(item)
    if count > threshold:
        token_content = "mask" if "-100" in last_red_token else "pad"
        compressed.append(f"{RED}...({count - 2} {token_content})...{RESET}")
        compressed.append(last_red_token)
    elif count > 1:
        compressed.extend([last_red_token] * (count - 1))
    return compressed


def debug_print(
    input_ids,
    attention_mask,
    labels,
    intervention_locations,
    concept_input_ids,
    tokenizer,
    threshold=64,
):
    # Ensure inputs are on CPU and convert to Python lists
    input_ids = input_ids.cpu().tolist()
    attention_mask = attention_mask.cpu().tolist()
    labels = labels.cpu().tolist()
    concept_input_ids = (
        concept_input_ids.cpu().tolist() if concept_input_ids is not None else None
    )

    # Convert intervention locations to mask if provided
    intervention_masks = None
    if intervention_locations is not None:
        intervention_locations = intervention_locations.cpu().tolist()
        # Create mask where 1 is at each intervention location
        intervention_masks = []
        for batch_idx in range(len(input_ids)):
            # Create a mask of zeros
            mask = [0] * len(input_ids[batch_idx])
            # Set 1 at each intervention location
            if batch_idx < len(intervention_locations):
                for loc in intervention_locations[batch_idx][
                    0
                ]:  # First row of each batch
                    if 0 <= loc < len(mask):
                        mask[loc] = 1
            intervention_masks.append(mask)

    for batch_idx in range(len(input_ids)):
        print(f"\033[1mBATCH {batch_idx}:\033[0m")

        # Decode input_ids
        tokens = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])

        # Decode concept_input_ids if available
        concept_tokens = None
        if concept_input_ids is not None and batch_idx < len(concept_input_ids):
            concept_tokens = tokenizer.convert_ids_to_tokens(
                concept_input_ids[batch_idx]
            )

        # Prepare colored strings for input_ids, attention_mask, intervention_mask and labels
        input_str = []
        attn_str = []
        label_str = []
        interv_str = []
        concept_str = []

        for i, (token, mask, label) in enumerate(
            zip(tokens, attention_mask[batch_idx], labels[batch_idx])
        ):
            # Handle intervention mask if available
            is_intervention = False
            if intervention_masks and i < len(intervention_masks[batch_idx]):
                is_intervention = intervention_masks[batch_idx][i] == 1

            # Apply special coloring based on the new requirements:
            # - Green: visible in attention mask (mask=1)
            # - Yellow: intervened on but not visible (intervention=1, mask=0)
            # - Red: everything else
            if mask == 1:
                token_color = GREEN  # Visible in attention mask
            elif is_intervention and mask == 0:
                token_color = YELLOW  # Intervened on but not visible
            else:
                token_color = RED  # Everything else

            input_str.append(f"{token_color}{token}{RESET}")
            attn_str.append(f"{GREEN}1{RESET}" if mask else f"{RED}0{RESET}")

            if intervention_masks:
                interv_str.append(
                    f"{YELLOW}1{RESET}" if is_intervention else f"{RED}0{RESET}"
                )

            if label == -100:
                label_str.append(f"{RED}-100{RESET}")
            else:
                label_token = tokenizer.convert_ids_to_tokens([label])[0]
                label_str.append(f"{BLUE}{label_token}{RESET}")

        # Process concept tokens if available
        if concept_tokens is not None:
            for concept_token in concept_tokens:
                concept_str.append(f"{BLUE}{concept_token}{RESET}")

        # Compress long sequences of padding/masked tokens
        input_str = compress_sequence(input_str, threshold)
        attn_str = compress_sequence(attn_str, threshold)
        label_str = compress_sequence(label_str, threshold)
        if intervention_masks:
            interv_str = compress_sequence(interv_str, threshold)
        if concept_tokens is not None:
            concept_str = compress_sequence(concept_str, threshold)

        # Print results
        print("  input_ids:", "[" + ", ".join(input_str) + "]")
        print("  attn_mask:", "[" + ", ".join(attn_str) + "]")
        if intervention_masks:
            print("  interv_mask:", "[" + ", ".join(interv_str) + "]")
        if concept_tokens is not None:
            print("  concept_ids:", "[" + ", ".join(concept_str) + "]")
        print("  labels:", "[" + ", ".join(label_str) + "]")
