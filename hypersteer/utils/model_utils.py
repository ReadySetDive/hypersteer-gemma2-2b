#################################
#
# Model utils.
#
#################################
from contextlib import contextmanager

import einops
import torch
from tqdm.auto import tqdm


def calculate_perplexity(model, tokenizer, generated_texts, device):
    """
    Helper function to calculate perplexity for generated texts.
    This is common across all model implementations.
    """
    batch_input_ids = tokenizer(
        generated_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).input_ids.to(device)
    batch_attention_mask = (batch_input_ids != tokenizer.pad_token_id).float()

    # Forward pass without labels to get logits
    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

    logits = outputs.logits[:, :-1, :].contiguous()  # Remove last token prediction
    target_ids = batch_input_ids[:, 1:].contiguous()  # Shift right by 1

    # Calculate loss for each token
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))

    # Reshape losses and mask
    token_losses = token_losses.view(batch_input_ids.size(0), -1)
    mask = batch_attention_mask[:, 1:].contiguous()

    # Calculate perplexity for each sequence
    seq_lengths = mask.sum(dim=1)
    seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
    seq_perplexities = torch.exp(seq_losses).cpu().float().tolist()

    return seq_perplexities


def clip_grad_norm_sparse(parameters, max_norm, norm_type=2.0, eps=1e-6):
    """
    Clips gradient norm of an iterable of parameters (e.g. model.parameters())
    supporting both dense and sparse gradients.

    Args:
        parameters (Iterable[Tensor] or Tensor): model parameters.
        max_norm (float): maximum norm of the gradients.
        norm_type (float): type of the used p-norm. Can be 'inf' for max-norm.
        eps (float): small epsilon for numerical stability.

    Returns:
        total_norm (float): the (possibly clipped) total norm of the gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # compute total norm
    total_norm = 0.0
    if norm_type == float("inf"):
        # max absolute value over all grads
        max_val = 0.0
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                values = grad.coalesce()._values()
            else:
                values = grad.view(-1)
            max_val = max(max_val, values.abs().max().item())
        total_norm = max_val
    else:
        # sum of norms ** norm_type across all params
        sum_pow = 0.0
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                values = grad.coalesce()._values()
            else:
                values = grad.view(-1)
            sum_pow += values.norm(norm_type).item() ** norm_type
        total_norm = sum_pow ** (1.0 / norm_type)

    # clip coeff
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad
            if grad.is_sparse:
                # only scale the non-zero values
                grad = grad.coalesce()  # make sure indices/values are unique
                grad._values().mul_(clip_coef)
            else:
                grad.mul_(clip_coef)

    return total_norm


def get_lrs(optimizer):
    _lrs = []
    for param_group in optimizer.param_groups:
        _lrs.append(param_group["lr"])

    return _lrs


@contextmanager
def freeze_parameters(model: torch.nn.Module, exclude_keys: list[str] = None):
    for name, param in model.named_parameters():
        if exclude_keys and any(exclude_key in name for exclude_key in exclude_keys):
            continue
        param.requires_grad = False
    yield
    for param in model.parameters():
        param.requires_grad = True


def get_model_continues(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    is_chat_model=True,
    batch_size=8,
    include_system_prompt=False,
    verbose=False,
):
    """we ground examples with the model's original generation."""
    tokenizer.padding_side = "left"
    if is_chat_model:
        if include_system_prompt:

            def apply_chat_template(prompt):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
                nobos = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )[1:]
                return tokenizer.decode(nobos)
        else:

            def apply_chat_template(prompt):
                messages = [{"role": "user", "content": prompt}]
                nobos = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )[1:]
                return tokenizer.decode(nobos)

        prompts = [apply_chat_template(prompt) for prompt in prompts]

    # Process prompts in batches
    all_generated_texts = []
    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc="Generating responses",
        disable=not verbose,
    ):
        batch_prompts = prompts[i : i + batch_size]
        encoding = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
            model.device
        )
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding, max_new_tokens=max_new_tokens, do_sample=False
            )
            generated_ids = generated_ids[:, encoding.input_ids.shape[1] :]
        batch_generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        all_generated_texts.extend(batch_generated_texts)

    return all_generated_texts


def gather_residual_activations(
    model, target_layer, inputs
) -> torch.Tensor | list[torch.Tensor]:
    """
    Gathers the residual activations from the specified target layer(s).
    If target_layer is a list, returns a list of activations in the same order.
    Otherwise, returns a single activation tensor.
    """
    if isinstance(target_layer, list):
        activations = [None for _ in target_layer]

        def make_hook(idx):
            def hook(mod, inputs, outputs):
                activations[idx] = outputs[0]
                return outputs

            return hook

        handles = []
        for idx, layer_idx in enumerate(target_layer):
            handle = model.model.layers[layer_idx].register_forward_hook(
                make_hook(idx), always_call=True
            )
            handles.append(handle)
        _ = model.forward(**inputs)
        for handle in handles:
            handle.remove()
        return activations
    else:
        target_act = None

        def gather_target_act_hook(mod, inputs, outputs):
            nonlocal target_act
            target_act = outputs[0]
            return outputs

        handle = model.model.layers[target_layer].register_forward_hook(
            gather_target_act_hook, always_call=True
        )
        _ = model.forward(**inputs)
        handle.remove()
        return target_act


@torch.no_grad()
def set_decoder_norm_to_unit_norm(model):
    assert model.proj.weight is not None, "Decoder weight was not initialized."

    eps = torch.finfo(model.proj.weight.dtype).eps
    norm = torch.norm(model.proj.weight.data, dim=1, keepdim=True)
    model.proj.weight.data /= norm + eps


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(model):
    assert model.proj.weight is not None, "Decoder weight was not initialized."
    assert model.proj.weight.grad is not None  # keep pyright happy

    # Skip if sparse
    if model.proj.weight.grad.is_sparse:
        return

    parallel_component = einops.einsum(
        model.proj.weight.grad,
        model.proj.weight.data,
        "d_out d_in, d_out d_in -> d_out",
    )
    model.proj.weight.grad -= einops.einsum(
        parallel_component,
        model.proj.weight.data,
        "d_out, d_out d_in -> d_out d_in",
    )


def calculate_l1_losses(
    latent, non_topk_latent, labels=None, mask=None, batchmean=True
):
    """
    Calculate L1 losses with masked mean.

    Parameters:
    - latent: latent representation, shape [batch_size, seq_len]
    - non_topk_latent: non-topk latent representation, shape [batch_size, seq_len]
    - labels: labels, shape [batch_size]
    - mask: long mask, shape [batch_size, seq_len]
    - batchmean: whether to use batchmean
    """
    if mask is None:
        mask = torch.ones_like(latent, dtype=torch.long)

    mask = mask.bool()

    valid_counts = mask.sum(dim=-1)  # [batch_size]
    eps = torch.finfo(latent.dtype).eps
    if non_topk_latent is not None:
        masked_non_topk_sum = (non_topk_latent * mask).sum(dim=-1)  # [batch_size]
        mean_non_topk = masked_non_topk_sum / (valid_counts + eps)
        if batchmean:
            l1_loss = mean_non_topk.mean()  # mean across batch
        else:
            l1_loss = mean_non_topk
    else:
        masked_sum = (latent * mask).sum(dim=-1)  # [batch_size]
        mean_all = masked_sum / (valid_counts + eps)
        if batchmean:
            l1_loss = mean_all.mean()  # mean across batch
        else:
            l1_loss = mean_all
    return l1_loss


def compute_lm_loss(logits, labels, pad_token_id=-100, reduce=True):
    labels = labels.clone()
    labels[labels == pad_token_id] = -100
    shift_logits = logits[..., :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[..., 1:].contiguous()  # [B, T-1]
    batch, seq, vocab = shift_logits.shape
    flat_logits = shift_logits.view(-1, vocab)
    flat_labels = shift_labels.view(-1)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    flat_losses = loss_fct(flat_logits, flat_labels)  # [B*(T-1)]
    losses = flat_losses.view(batch, seq)  # [B, T-1]
    mask = shift_labels != -100  # [B, T-1]
    per_seq = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    mask = shift_labels != -100  # [B, T-1]
    counts = mask.sum(dim=1).float().clamp_min(1)  # [B]

    # perfect match with HF loss implementation
    if reduce:
        length_avg_loss = (per_seq * counts).sum() / counts.sum()
        return length_avg_loss

    # NOTE: this is not the same as HF loss implementation
    return per_seq  # [B]


def masked_kl_div(
    pred_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    avg_over_sequence: bool = False,
):
    """Compute the KL divergence between two distributions, optionally ignoring masked tokens.

    Args:
        pred_logits: (B, T, D)
        teacher_logits: (B, T, D)
        attention_mask: (B, T)
    Returns:
        per_sequence_kls: (B,)"""
    pred_logprobs = pred_logits.log_softmax(-1)  # (B, T, D)
    teacher_logprobs = teacher_logits.log_softmax(-1)  # (B, T, D)
    per_token_kls = (teacher_logprobs.exp() * (teacher_logprobs - pred_logprobs)).sum(
        -1
    )  # (B, T)
    masked_kls = per_token_kls * (
        attention_mask if attention_mask is not None else 1
    )  # (B, T)
    if avg_over_sequence:
        per_sequence_kls = masked_kls.sum(-1) / (
            attention_mask.sum(-1)
            if attention_mask is not None
            else masked_kls.shape[-1]
        )  # (B,)
    else:
        per_sequence_kls = masked_kls.sum(-1)  # (B,)
    return per_sequence_kls


def get_prefix_length(tokenizer, common_prefix=None):
    if common_prefix is None:
        message_a = [{"role": "user", "content": "1"}]
        message_b = [{"role": "user", "content": "2"}]
        tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
        tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
        print("Detecting sequence a:", tokens_a)
        print("Detecting sequence b:", tokens_b)
        prefix_length = 0
        for i, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
            if ta != tb:
                prefix_length = i
                break
    else:
        message = [{"role": "user", "content": common_prefix}]
        tokens = tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True
        )
        prefix_length = len(tokens)
    return prefix_length


def get_suffix_length(tokenizer):
    message_a = [{"role": "user", "content": "1"}]
    message_b = [{"role": "user", "content": "2"}]
    tokens_a = tokenizer.apply_chat_template(message_a, tokenize=True)
    tokens_b = tokenizer.apply_chat_template(message_b, tokenize=True)
    suffix_length = 0
    for i, (ta, tb) in enumerate(zip(reversed(tokens_a), reversed(tokens_b))):
        if ta != tb:
            suffix_length = i
            break
    return suffix_length, tokenizer.decode(tokens_a[-suffix_length:])
