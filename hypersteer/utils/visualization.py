from __future__ import annotations

import atexit
import concurrent.futures
import html
import os
import string
import unicodedata

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from hypersteer.utils.helpers import get_logger

logger = get_logger(__name__)

# Use Agg backend for matplotlib to avoid GUI dependencies
matplotlib.use("Agg")

_MAX_THREADPOOL_WORKERS = 20

# Remove global queue and thread, add ThreadPoolExecutor
_viz_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_MAX_THREADPOOL_WORKERS
)


class Visualizer:
    def __init__(self):
        self._current_viz_step = 0
        self.console_logging = True
        self.visualization_enabled = True

    def set_console_logging(self, enabled=True):
        self.console_logging = enabled

    def set_visualization_enabled(self, enabled=True):
        self.visualization_enabled = enabled

    def visualize_all_samples(self, enabled=True):
        self.visualize_all_samples_enabled = enabled

    def clean_text_for_display(self, text):
        """Clean text by removing non-printable characters and skip empty/garbage tokens."""

        def clean_token(t):
            if t is None:
                return None
            t = "".join(
                c
                for c in t
                if c in string.printable and not unicodedata.category(c).startswith("C")
            )
            t = t.strip()
            t = "".join(
                c for c in t if unicodedata.category(c) not in ["Cf", "Cc", "Cs"]
            )
            t = " ".join(t.split())
            if t == "" or t.isspace():
                return None
            return t

        if text is None:
            return None
        if isinstance(text, list):
            cleaned = [self.clean_text_for_display(t) for t in text]
            if len(cleaned) > 0 and isinstance(cleaned[0], list):
                return [[tok for tok in toks if tok is not None] for toks in cleaned]
            else:
                return [tok for tok in cleaned if tok is not None]
        return clean_token(text)

    def _process_visualization(
        self,
        mask_tensor,
        step,
        dump_dir,
        batch_tokens,
        attention_mask,
        concept_ids,
        concept_strings,
        viz_mode,
        visualize_single_sample,
        pdf_visualization=False,
        png_visualization=False,
        title_prefix=None,
        custom_title=None,
        log_to_console=True,
        enable_visualization=True,
        log_all_examples=True,
    ):
        """
        Create ONE HTML per batch/step with a slider to go through all examples in that batch.
        Organize by viz_mode instead of step to avoid creating new panels for each step.
        """
        try:
            if log_to_console:
                print(
                    f"[viz] Starting visualization for step {step}, viz_mode: {viz_mode}"
                )
            # Initial validation
            if mask_tensor is None:
                if log_to_console:
                    print("[viz] Cannot process visualization: mask_tensor is None")
                return

            if mask_tensor.dim() < 2:
                if log_to_console:
                    print(
                        f"[viz] Cannot process visualization: mask_tensor has invalid shape: {mask_tensor.shape}"
                    )
                return

            if (
                attention_mask is not None
                and attention_mask.shape[0] != mask_tensor.shape[0]
            ):
                if log_to_console:
                    print(
                        f"[viz] Cannot process visualization: mask_tensor and attention_mask batch dimensions don't match: {mask_tensor.shape} vs {attention_mask.shape}"
                    )
                return

            if batch_tokens is not None:
                if not isinstance(batch_tokens, list):
                    if log_to_console:
                        print(
                            f"[viz] Cannot process visualization: batch_tokens must be a list, got {type(batch_tokens)}"
                        )
                    return
                if len(batch_tokens) == 0:
                    if log_to_console:
                        print(
                            "[viz] Cannot process visualization: batch_tokens is empty"
                        )
                    return
                if len(batch_tokens) != mask_tensor.shape[0] and mask_tensor.dim() > 2:
                    if log_to_console:
                        print(
                            f"[viz] Visualization warning: batch_tokens length ({len(batch_tokens)}) doesn't match mask_tensor batch size ({mask_tensor.shape[0]})"
                        )

            # Create viz_mode-specific directory instead of step-specific
            viz_mode_clean = (
                viz_mode.replace("/", "_").replace(" ", "_") if viz_mode else "default"
            )
            viz_mode_folder = os.path.join(dump_dir, viz_mode_clean)
            os.makedirs(viz_mode_folder, exist_ok=True)
            if log_to_console:
                print(f"[viz] Created directory: {viz_mode_folder}")

            # Clean batch tokens if available
            if batch_tokens is not None:
                batch_tokens = self.clean_text_for_display(batch_tokens)

            # Determine which samples to visualize
            sample_range = (
                range(mask_tensor.shape[0]) if not visualize_single_sample else [0]
            )

            # Collect all samples for the batch
            all_samples_data = []
            concept_texts = []

            for i in sample_range:
                if log_to_console:
                    print(f"[viz] Processing sample {i}")

                # Assume batch dim is first
                vis_mask = mask_tensor[i].float().numpy()

                # Skip only if vis_mask is completely empty/invalid
                if vis_mask.size == 0:
                    if log_to_console:
                        print(f"[viz] Skipping sample {i}: vis_mask is empty")
                    continue

                tokens = None
                if attention_mask is not None:
                    valid_indices = torch.where(attention_mask[i] > 0)[0].tolist()
                    if valid_indices:
                        # Only apply attention mask filtering if we have valid indices
                        vis_mask = (
                            vis_mask[:, valid_indices[0] : valid_indices[-1] + 1]
                            if vis_mask.ndim > 1
                            else vis_mask
                        )
                        if batch_tokens is not None and i < len(batch_tokens):
                            tokens = batch_tokens[i][
                                valid_indices[0] : valid_indices[-1] + 1
                            ]
                    else:
                        # No valid attention indices, but still include the sample with full tokens
                        if log_to_console:
                            print(
                                f"[viz] Sample {i}: no valid attention indices, using full tokens"
                            )
                        if batch_tokens is not None and i < len(batch_tokens):
                            tokens = batch_tokens[i]
                elif batch_tokens is not None and i < len(batch_tokens):
                    tokens = batch_tokens[i]

                concept_id = None
                concept_string = None
                if concept_ids is not None and i < len(concept_ids):
                    if isinstance(concept_ids, list):
                        concept_id = concept_ids[i]
                    else:
                        concept_id = int(concept_ids[i])
                    if concept_strings is not None and i < len(concept_strings):
                        concept_string = concept_strings[i]
                    elif (
                        hasattr(self, "concept_id_to_text")
                        and self.concept_id_to_text is not None
                    ):
                        try:
                            concept_string = self.concept_id_to_text.get(
                                concept_id, None
                            )
                        except Exception:
                            concept_string = None

                # Create tokens if we don't have them
                if tokens is None or len(tokens) == 0:
                    if log_to_console:
                        print(
                            f"[viz] Sample {i}: tokens is None/empty, creating placeholder tokens"
                        )
                    # Create placeholder tokens based on vis_mask size
                    if vis_mask.ndim > 1:
                        num_tokens = vis_mask.shape[-1]
                    else:
                        num_tokens = (
                            len(vis_mask) if hasattr(vis_mask, "__len__") else 1
                        )
                    tokens = [f"<sample_{i}_token_{j}>" for j in range(num_tokens)]

                # Get token weights from vis_mask
                if vis_mask.ndim > 1 and vis_mask.shape[0] > 1:
                    token_weights = np.mean(vis_mask, axis=0)
                else:
                    token_weights = (
                        vis_mask.flatten()
                        if vis_mask.ndim > 0
                        else np.array([vis_mask])
                    )

                # Ensure token_weights matches tokens length
                if len(token_weights) != len(tokens):
                    if log_to_console:
                        print(
                            f"[viz] Sample {i}: token_weights length {len(token_weights)} != tokens length {len(tokens)}, adjusting"
                        )
                    min_len = min(len(token_weights), len(tokens))
                    if min_len > 0:
                        token_weights = token_weights[:min_len]
                        tokens = tokens[:min_len]
                    else:
                        # Fallback: create single token
                        tokens = [f"<sample_{i}>"]
                        token_weights = np.array([1.0])

                sample_data = {
                    "sample_id": i,
                    "tokens": tokens,
                    "token_weights": token_weights,
                    "concept_id": concept_id,
                    "concept_string": concept_string,
                }
                all_samples_data.append(sample_data)
                if concept_string:
                    concept_texts.append(concept_string)

                if log_to_console:
                    print(
                        f"[viz] Added sample {i} with {len(tokens)} tokens, concept_id={concept_id}"
                    )

            if log_to_console:
                print(
                    f"[viz] Total samples collected: {len(all_samples_data)} out of {len(sample_range)} in batch"
                )

            if not all_samples_data:
                if log_to_console:
                    print(f"[viz] No valid samples to visualize for step {step}")
                return

            # Create title with basic information only (no concept info since it varies per example)
            title = (
                custom_title
                or f"{title_prefix or 'Training'} - Step {step} - {len(all_samples_data)} samples"
            )

            # Generate ONE HTML for the entire batch, with step in filename to avoid conflicts
            html_content = self._create_batch_html_visualization(
                all_samples_data, title
            )
            html_filename = f"batch_step{step}.html"
            html_path = os.path.join(viz_mode_folder, html_filename)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            if log_to_console:
                print(f"[viz] Saved batch HTML: {html_path}")
                print(f"[viz] To view, open this file in your browser: {html_path}")

            try:
                wandb_html = wandb.Html(html_content)
            except Exception as e:
                if log_to_console:
                    print(f"[viz] W&B HTML error: {e}")
                wandb_html = None
            if wandb.run and wandb_html:
                section_name = self._get_viz_section_name(viz_mode)
                # Use viz_mode as the key instead of including step, so all visualizations
                # for the same mode go to the same panel
                wandb_key = f"{section_name}/batch_html"
                wandb.log(
                    {wandb_key: wandb_html},
                    commit=True,
                )
            if log_to_console:
                print(f"[viz] Finished batch HTML visualization for step {step}")
        except Exception as e:
            if log_to_console:
                print(f"[viz] Error in visualization processing: {e}")
                import traceback

                print(traceback.format_exc())

    def log_visualization(
        self,
        mask,
        step=None,
        dump_dir="assets/cache/visualizations",
        batch_tokens=None,
        input_ids=None,
        attention_mask=None,
        concept_ids=None,
        concept_strings=None,
        viz_mode=None,
        pdf_visualization=False,
        png_visualization=False,
        title_prefix=None,
        custom_title=None,
        log_to_console=True,
        enable_visualization=True,
        visualize_all_samples=True,
        log_all_examples=True,
    ):
        """
        Non-blocking visualization that queues the request for background processing.
        Accepts custom title_prefix and custom_title for more flexible plot labeling.
        """
        if not enable_visualization:
            return
        if batch_tokens is None and input_ids is None:
            if log_to_console:
                print(
                    "[viz] Cannot visualize mask: both batch_tokens and input_ids are None"
                )
            return
        step = step or self._current_viz_step
        if torch.is_tensor(step):
            step = int(step.item())
        try:
            mask_tensor = mask.detach().cpu()
            attn_mask = (
                attention_mask.detach().cpu() if attention_mask is not None else None
            )
            concept_ids_cpu = None
            if concept_ids is not None:
                if isinstance(concept_ids, torch.Tensor):
                    concept_ids_cpu = concept_ids.detach().cpu().tolist()
                else:
                    concept_ids_cpu = concept_ids
            if batch_tokens is not None:
                if isinstance(batch_tokens, list) and len(batch_tokens) == 0:
                    if log_to_console:
                        print("[viz] Cannot visualize mask: batch_tokens is empty list")
                    return
                if log_to_console:
                    tokens_shape = f"batch of {len(batch_tokens)}"
                    if isinstance(batch_tokens[0], list):
                        tokens_shape += f" lists with avg {sum(len(t) for t in batch_tokens) / len(batch_tokens):.1f} tokens"
                    print(
                        f"[viz] Visualizing mask: {tokens_shape}, mask shape: {mask_tensor.shape}"
                    )

            # Submit to thread pool
            _viz_executor.submit(
                self._process_visualization,
                mask_tensor,
                step,
                dump_dir,
                batch_tokens,
                attn_mask,
                concept_ids_cpu,
                concept_strings,
                viz_mode,
                not visualize_all_samples,
                pdf_visualization,
                png_visualization,
                title_prefix,
                custom_title,
                log_to_console,
                True,
                log_all_examples,
            )
            if log_to_console:
                print(f"[viz] Submitted visualization for step {step}")
        except Exception as e:
            if log_to_console:
                print(f"[viz] Error preparing visualization: {e}")
                import traceback

                print(traceback.format_exc())
        return

    def set_current_step(self, step):
        """Set the current step for visualization."""
        self._current_viz_step = step

    def wait_for_visualizations(self, timeout=None):
        """Wait for pending visualizations to complete."""
        try:
            _viz_executor.shutdown(wait=True)
            return True
        except:  # noqa: E722
            return False

    def _create_batch_html_visualization(self, all_samples_data, title=None):
        """
        Create an HTML-based interactive token visualization that can be displayed in W&B.

        Args:
            all_samples_data: List of dictionaries, each containing 'sample_id', 'tokens', 'token_weights', 'concept_id', 'concept_string'
            title: Optional title for the visualization

        Returns:
            HTML string
        """

        # Generate HTML with inline CSS for token visualization
        html_output = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="UTF-8">',
            "<style>",
            '  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }',
            "  .container { max-width: 1200px; margin: 0 auto; padding: 20px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; }",
            "  h2 { color: #333; margin-top: 0; }",
            "  .controls { margin: 15px 0; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }",
            "  .token-container { line-height: 2.5; word-wrap: break-word; background-color: white; padding: 15px; border-radius: 4px; }",
            "  .token { display: inline-block; padding: 3px 5px; margin: 2px; border-radius: 3px; cursor: pointer; transition: all 0.2s; }",
            "  .token:hover { filter: brightness(1.2); box-shadow: 0 0 5px rgba(0,0,0,0.3); transform: translateY(-2px); }",
            "  .tooltip { position: fixed; background: #333; color: white; padding: 8px; border-radius: 4px; font-size: 12px; z-index: 100; display: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }",
            "  .legend { margin-top: 20px; display: flex; align-items: center; }",
            "  .legend-gradient { width: 200px; height: 20px; background: linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725); margin-right: 10px; border-radius: 2px; }",
            "  .legend-labels { display: flex; justify-content: space-between; width: 200px; font-size: 12px; }",
            "  button, select { padding: 8px 12px; background-color: #f1f1f1; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }",
            "  button:hover { background-color: #e2e2e2; }",
            "  .active { background-color: #4a8eff; color: white; }",
            "  .hidden { display: none; }",
            "  #threshold-value { margin-left: 10px; min-width: 40px; display: inline-block; }",
            "  .slider { width: 200px; }",
            "  .stats { margin: 10px 0; font-size: 14px; color: #555; }",
            "  @media (max-width: 768px) { .controls { flex-direction: column; align-items: flex-start; } }",
            "</style>",
            "</head>",
            "<body>",
            '<div class="container">',
        ]

        # Add title if provided
        if title:
            html_output.append(f"<h2>{html.escape(title)}</h2>")

        # Add control panel
        html_output.extend(
            [
                '<div class="controls">',
                '  <button id="highlight-high">Highlight Top 25%</button>',
                '  <button id="highlight-all">Show All</button>',
                '  <label>Threshold: <input type="range" id="threshold-slider" class="slider" min="0" max="100" value="0"><span id="threshold-value">0.00</span></label>',
                '  <select id="sort-tokens">',
                '    <option value="position">Original Order</option>',
                '    <option value="weight-desc">Importance (High → Low)</option>',
                '    <option value="weight-asc">Importance (Low → High)</option>',
                "  </select>",
                "</div>",
                '<div class="stats">',
                f'  <span id="token-count">Total tokens: {sum(len(sample["tokens"]) for sample in all_samples_data)}</span> | ',
                '  <span id="highlighted-count">Highlighted: 0</span>',
                "</div>",
            ]
        )

        # Start token container
        html_output.append('<div class="token-container" id="token-container">')

        # Function to convert a weight to a color in the viridis colormap
        def weight_to_color(weight):
            # Viridis-like colormap with smoother interpolation
            colors = [
                (68, 1, 84),  # Dark purple (0.0)
                (59, 82, 139),  # Blue (0.25)
                (33, 145, 140),  # Teal (0.5)
                (94, 201, 98),  # Green (0.75)
                (253, 231, 37),  # Yellow (1.0)
            ]

            # Improved interpolation
            if weight <= 0:
                return f"rgb{colors[0]}"
            elif weight >= 1:
                return f"rgb{colors[-1]}"
            else:
                idx = weight * (len(colors) - 1)
                lower_idx = int(idx)
                upper_idx = min(lower_idx + 1, len(colors) - 1)
                fraction = idx - lower_idx

                r = int(
                    colors[lower_idx][0]
                    + fraction * (colors[upper_idx][0] - colors[lower_idx][0])
                )
                g = int(
                    colors[lower_idx][1]
                    + fraction * (colors[upper_idx][1] - colors[lower_idx][1])
                )
                b = int(
                    colors[lower_idx][2]
                    + fraction * (colors[upper_idx][2] - colors[lower_idx][2])
                )

                return f"rgb({r}, {g}, {b})"

        # Add tokens with colors based on weights
        for sample_idx, sample_data in enumerate(all_samples_data):
            tokens = sample_data["tokens"]
            token_weights = sample_data["token_weights"]
            concept_id = sample_data["concept_id"]
            concept_string = sample_data["concept_string"]

            # Normalize weights to [0, 1] if not already
            if np.max(token_weights) > 1.0 or np.min(token_weights) < 0.0:
                token_weights = (token_weights - np.min(token_weights)) / (
                    np.max(token_weights) - np.min(token_weights) + 1e-8
                )

            html_output.append(f'<div class="sample" id="sample_{sample_idx}">')
            html_output.append(
                f"<h3>Sample {sample_idx} - Concept ID: {concept_id} - {concept_string}</h3>"
            )
            html_output.append('<div class="token-container">')

            for i, (token, weight) in enumerate(zip(tokens, token_weights)):
                # Clean token for display
                clean_token = html.escape(token.replace("\n", " ").replace("\t", " "))
                if clean_token == "":
                    clean_token = "□"  # Use placeholder for empty/whitespace tokens

                # Determine text color based on background brightness
                text_color = "white" if weight > 0.5 else "black"
                bg_color = weight_to_color(weight)

                # Create token span with inline style and data attributes for interactivity
                html_output.append(
                    f'<span class="token" style="background-color: {bg_color}; color: {text_color};" '
                    f'data-weight="{weight:.6f}" data-token="{clean_token}" data-idx="{i}" data-pos="{i}">'
                    f"{clean_token}"
                    f"</span>"
                )

            html_output.append("</div>")
            html_output.append("</div>")

        # Close token container
        html_output.append("</div>")

        # Add legend
        html_output.extend(
            [
                '<div class="legend">',
                "  <div>",
                '    <div class="legend-gradient"></div>',
                '    <div class="legend-labels">',
                "      <span>0.0</span>",
                "      <span>0.5</span>",
                "      <span>1.0</span>",
                "    </div>",
                "  </div>",
                '  <div style="margin-left: 10px;">Token Importance</div>',
                "</div>",
                "</div>",  # Close container
            ]
        )

        # Add tooltip element
        html_output.append('<div id="tooltip" class="tooltip"></div>')

        # Add JavaScript for interactivity
        html_output.extend(
            [
                "<script>",
                'document.addEventListener("DOMContentLoaded", function() {',
                '  const tokenContainer = document.getElementById("token-container");',
                '  const tooltip = document.getElementById("tooltip");',
                '  const tokens = document.querySelectorAll(".token");',
                '  const highlightHighBtn = document.getElementById("highlight-high");',
                '  const highlightAllBtn = document.getElementById("highlight-all");',
                '  const thresholdSlider = document.getElementById("threshold-slider");',
                '  const thresholdValue = document.getElementById("threshold-value");',
                '  const sortSelect = document.getElementById("sort-tokens");',
                '  const highlightedCount = document.getElementById("highlighted-count");',
                "",
                "  // Store original token order for reset",
                "  const originalOrder = Array.from(tokens).map(token => ({",
                "    el: token,",
                '    pos: parseInt(token.getAttribute("data-pos"))',
                "  }));",
                "",
                "  // Setup tooltip functionality",
                "  tokens.forEach(token => {",
                '    token.addEventListener("mouseover", function(e) {',
                '      const weight = parseFloat(this.getAttribute("data-weight"));',
                '      const tokenText = this.getAttribute("data-token");',
                '      const idx = this.getAttribute("data-idx");',
                "      ",
                '      tooltip.innerHTML = `Token: "${tokenText}"<br>Weight: ${weight.toFixed(4)}<br>Position: ${idx}`;',
                '      tooltip.style.left = (e.pageX + 10) + "px";',
                '      tooltip.style.top = (e.pageY + 10) + "px";',
                '      tooltip.style.display = "block";',
                "    });",
                "    ",
                '    token.addEventListener("mouseout", function() {',
                '      tooltip.style.display = "none";',
                "    });",
                "  });",
                "",
                "  // Threshold slider functionality",
                '  thresholdSlider.addEventListener("input", function() {',
                "    const threshold = parseFloat(this.value) / 100;",
                "    thresholdValue.textContent = threshold.toFixed(2);",
                "    ",
                "    let highlightCount = 0;",
                "    tokens.forEach(token => {",
                '      const weight = parseFloat(token.getAttribute("data-weight"));',
                "      if (weight >= threshold) {",
                '        token.style.opacity = "1";',
                '        token.style.filter = "none";',
                "        highlightCount++;",
                "      } else {",
                '        token.style.opacity = "0.3";',
                '        token.style.filter = "grayscale(100%)";',
                "      }",
                "    });",
                "    highlightedCount.textContent = `Highlighted: ${highlightCount}`;",
                "  });",
                "",
                "  // Highlight buttons",
                '  highlightHighBtn.addEventListener("click", function() {',
                '    const weights = Array.from(tokens).map(t => parseFloat(t.getAttribute("data-weight")));',
                "    weights.sort((a, b) => b - a);",
                "    const threshold = weights[Math.floor(weights.length * 0.25)] || 0;",
                "    thresholdSlider.value = Math.round(threshold * 100);",
                "    thresholdValue.textContent = threshold.toFixed(2);",
                "    ",
                "    let highlightCount = 0;",
                "    tokens.forEach(token => {",
                '      const weight = parseFloat(token.getAttribute("data-weight"));',
                "      if (weight >= threshold) {",
                '        token.style.opacity = "1";',
                '        token.style.filter = "none";',
                "        highlightCount++;",
                "      } else {",
                '        token.style.opacity = "0.3";',
                '        token.style.filter = "grayscale(100%)";',
                "      }",
                "    });",
                "    highlightedCount.textContent = `Highlighted: ${highlightCount}`;",
                "  });",
                "",
                '  highlightAllBtn.addEventListener("click", function() {',
                "    thresholdSlider.value = 0;",
                '    thresholdValue.textContent = "0.00";',
                "    tokens.forEach(token => {",
                '      token.style.opacity = "1";',
                '      token.style.filter = "none";',
                "    });",
                "    highlightedCount.textContent = `Highlighted: ${tokens.length}`;",
                "  });",
                "",
                "  // Sorting functionality",
                "  sortSelect.addEventListener('change', function() {",
                "    const sortBy = this.value;",
                "    const tokenArr = Array.from(tokens);",
                "    ",
                "    tokenArr.sort((a, b) => {",
                '      if (sortBy === "position") {',
                '        return parseInt(a.getAttribute("data-pos")) - parseInt(b.getAttribute("data-pos"));',
                '      } else if (sortBy === "weight-desc") {',
                '        return parseFloat(b.getAttribute("data-weight")) - parseFloat(a.getAttribute("data-weight"));',
                '      } else if (sortBy === "weight-asc") {',
                '        return parseFloat(a.getAttribute("data-weight")) - parseFloat(b.getAttribute("data-weight"));',
                "      }",
                "    });",
                "    ",
                "    // Clear container and append sorted tokens",
                '    tokenContainer.innerHTML = "";',
                "    tokenArr.forEach(token => tokenContainer.appendChild(token));",
                "  });",
                "});",
                "</script>",
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html_output)

    def _create_batch_html_visualization(self, all_samples_data, title=None):
        """
        Create an HTML-based interactive batch visualization with a slider to navigate through examples.

        Args:
            all_samples_data: List of dicts with 'sample_id', 'tokens', 'token_weights', 'concept_id', 'concept_string'
            title: Optional title for the visualization

        Returns:
            HTML string
        """
        # Debug print to verify number of samples in the slider
        print(
            f"[viz] _create_batch_html_visualization: {len(all_samples_data)} samples in slider"
        )
        if not all_samples_data:
            return "<html><body><h2>No data to visualize</h2></body></html>"

        # Generate HTML with inline CSS for batch token visualization
        html_output = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="UTF-8">',
            "<style>",
            '  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }',
            "  .container { max-width: 1200px; margin: 0 auto; padding: 20px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; }",
            "  h2 { color: #333; margin-top: 0; }",
            "  .sample-controls { margin: 15px 0; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; background-color: #f8f9fa; padding: 15px; border-radius: 8px; }",
            "  .controls { margin: 15px 0; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }",
            "  .sample-slider { width: 300px; margin: 0 10px; }",
            "  .sample-info { margin-left: 20px; font-weight: bold; color: #333; }",
            "  .token-container { line-height: 2.5; word-wrap: break-word; background-color: white; padding: 15px; border-radius: 4px; min-height: 200px; }",
            "  .token { display: inline-block; padding: 3px 5px; margin: 2px; border-radius: 3px; cursor: pointer; transition: all 0.2s; }",
            "  .token:hover { filter: brightness(1.2); box-shadow: 0 0 5px rgba(0,0,0,0.3); transform: translateY(-2px); }",
            "  .tooltip { position: fixed; background: #333; color: white; padding: 8px; border-radius: 4px; font-size: 12px; z-index: 100; display: none; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }",
            "  .legend { margin-top: 20px; display: flex; align-items: center; }",
            "  .legend-gradient { width: 200px; height: 20px; background: linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725); margin-right: 10px; border-radius: 2px; }",
            "  .legend-labels { display: flex; justify-content: space-between; width: 200px; font-size: 12px; }",
            "  button, select { padding: 8px 12px; background-color: #f1f1f1; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; color: #333; }",
            "  button:hover { background-color: #e2e2e2; color: #333; }",
            "  .active { background-color: #4a8eff; color: white; }",
            "  .hidden { display: none; }",
            "  #threshold-value { margin-left: 10px; min-width: 40px; display: inline-block; }",
            "  .slider { width: 200px; }",
            "  .stats { margin: 10px 0; font-size: 14px; color: #555; }",
            "  .navigation-buttons { display: flex; gap: 10px; }",
            "  @media (max-width: 768px) { .controls { flex-direction: column; align-items: flex-start; } }",
            "</style>",
            "</head>",
            "<body>",
            '<div class="container">',
        ]

        # Add title if provided
        if title:
            html_output.append(f"<h2>{html.escape(title)}</h2>")

        # Add sample navigation controls
        html_output.extend(
            [
                '<div class="sample-controls">',
                '  <div class="navigation-buttons">',
                '    <button id="prev-sample">← Previous</button>',
                '    <button id="next-sample">Next →</button>',
                "  </div>",
                f'  <label>Sample: <input type="range" id="sample-slider" class="sample-slider" min="0" max="{len(all_samples_data) - 1}" value="0"></label>',
                '  <div class="sample-info" id="sample-info">Sample 0</div>',
                "</div>",
            ]
        )

        # Add control panel for current sample
        html_output.extend(
            [
                '<div class="controls">',
                '  <button id="highlight-high">Highlight Top 25%</button>',
                '  <button id="highlight-all">Show All</button>',
                '  <label>Threshold: <input type="range" id="threshold-slider" class="slider" min="0" max="100" value="0"><span id="threshold-value">0.00</span></label>',
                '  <select id="sort-tokens">',
                '    <option value="position">Original Order</option>',
                '    <option value="weight-desc">Importance (High → Low)</option>',
                '    <option value="weight-asc">Importance (Low → High)</option>',
                "  </select>",
                "</div>",
                '<div class="stats">',
                '  <span id="token-count">Total tokens: 0</span> | ',
                '  <span id="highlighted-count">Highlighted: 0</span>',
                "</div>",
            ]
        )

        # Start token container
        html_output.append('<div class="token-container" id="token-container">')
        html_output.append("</div>")  # Will be populated by JavaScript

        # Add legend
        html_output.extend(
            [
                '<div class="legend">',
                "  <div>",
                '    <div class="legend-gradient"></div>',
                '    <div class="legend-labels">',
                "      <span>0.0</span>",
                "      <span>0.5</span>",
                "      <span>1.0</span>",
                "    </div>",
                "  </div>",
                '  <div style="margin-left: 10px;">Token Importance</div>',
                "</div>",
                "</div>",  # Close container
            ]
        )

        # Add tooltip element
        html_output.append('<div id="tooltip" class="tooltip"></div>')

        # Embed all sample data as JSON
        import json

        samples_json = json.dumps(
            [
                {
                    "sample_id": sample["sample_id"],
                    "tokens": sample["tokens"],
                    "token_weights": sample["token_weights"].tolist(),
                    "concept_id": sample["concept_id"],
                    "concept_string": sample["concept_string"],
                }
                for sample in all_samples_data
            ]
        )

        # Add JavaScript for interactivity
        html_output.extend(
            [
                "<script>",
                f"const allSamplesData = {samples_json};",
                "let currentSampleIndex = 0;",
                "",
                'document.addEventListener("DOMContentLoaded", function() {',
                '  const tokenContainer = document.getElementById("token-container");',
                '  const tooltip = document.getElementById("tooltip");',
                '  const sampleSlider = document.getElementById("sample-slider");',
                '  const sampleInfo = document.getElementById("sample-info");',
                '  const prevBtn = document.getElementById("prev-sample");',
                '  const nextBtn = document.getElementById("next-sample");',
                '  const highlightHighBtn = document.getElementById("highlight-high");',
                '  const highlightAllBtn = document.getElementById("highlight-all");',
                '  const thresholdSlider = document.getElementById("threshold-slider");',
                '  const thresholdValue = document.getElementById("threshold-value");',
                '  const sortSelect = document.getElementById("sort-tokens");',
                '  const tokenCount = document.getElementById("token-count");',
                '  const highlightedCount = document.getElementById("highlighted-count");',
                "",
                "  // Function to convert a weight to a color in the viridis colormap",
                "  function weightToColor(weight) {",
                "    const colors = [",
                "      [68, 1, 84],    // Dark purple (0.0)",
                "      [59, 82, 139],  // Blue (0.25)",
                "      [33, 145, 140], // Teal (0.5)",
                "      [94, 201, 98],  // Green (0.75)",
                "      [253, 231, 37], // Yellow (1.0)",
                "    ];",
                "",
                "    if (weight <= 0) return `rgb(${colors[0].join(',')})`;",
                "    if (weight >= 1) return `rgb(${colors[colors.length-1].join(',')})`;",
                "",
                "    const idx = weight * (colors.length - 1);",
                "    const lowerIdx = Math.floor(idx);",
                "    const upperIdx = Math.min(lowerIdx + 1, colors.length - 1);",
                "    const fraction = idx - lowerIdx;",
                "",
                "    const r = Math.round(colors[lowerIdx][0] + fraction * (colors[upperIdx][0] - colors[lowerIdx][0]));",
                "    const g = Math.round(colors[lowerIdx][1] + fraction * (colors[upperIdx][1] - colors[lowerIdx][1]));",
                "    const b = Math.round(colors[lowerIdx][2] + fraction * (colors[upperIdx][2] - colors[lowerIdx][2]));",
                "",
                "    return `rgb(${r}, ${g}, ${b})`;",
                "  }",
                "",
                "  // Function to render current sample",
                "  function renderCurrentSample() {",
                "    const sample = allSamplesData[currentSampleIndex];",
                "    if (!sample) return;",
                "",
                "    // Update sample info",
                "    let infoText = `Sample ${sample.sample_id}`;",
                "    if (sample.concept_string) {",
                "      infoText += ` - ${sample.concept_string}`;",
                "    } else if (sample.concept_id !== null) {",
                "      infoText += ` - Concept ID: ${sample.concept_id}`;",
                "    }",
                "    sampleInfo.textContent = infoText;",
                "",
                "    // Normalize weights",
                "    const weights = sample.token_weights;",
                "    const minWeight = Math.min(...weights);",
                "    const maxWeight = Math.max(...weights);",
                "    const normalizedWeights = weights.map(w => (w - minWeight) / (maxWeight - minWeight + 1e-8));",
                "",
                "    // Clear container and add tokens",
                "    tokenContainer.innerHTML = '';",
                "    sample.tokens.forEach((token, i) => {",
                "      const cleanToken = token.replace(/\\n/g, ' ').replace(/\\t/g, ' ') || '□';",
                "      const weight = normalizedWeights[i];",
                "      const textColor = weight > 0.5 ? 'white' : 'black';",
                "      const bgColor = weightToColor(weight);",
                "",
                "      const tokenSpan = document.createElement('span');",
                "      tokenSpan.className = 'token';",
                "      tokenSpan.style.backgroundColor = bgColor;",
                "      tokenSpan.style.color = textColor;",
                "      tokenSpan.setAttribute('data-weight', weight.toFixed(6));",
                "      tokenSpan.setAttribute('data-token', cleanToken);",
                "      tokenSpan.setAttribute('data-idx', i);",
                "      tokenSpan.setAttribute('data-pos', i);",
                "      tokenSpan.textContent = cleanToken;",
                "",
                "      // Add hover events",
                "      tokenSpan.addEventListener('mouseover', function(e) {",
                "        const weight = parseFloat(this.getAttribute('data-weight'));",
                "        const tokenText = this.getAttribute('data-token');",
                "        const idx = this.getAttribute('data-idx');",
                '        tooltip.innerHTML = `Token: "${tokenText}"<br>Weight: ${weight.toFixed(4)}<br>Position: ${idx}`;',
                "        tooltip.style.left = (e.pageX + 10) + 'px';",
                "        tooltip.style.top = (e.pageY + 10) + 'px';",
                "        tooltip.style.display = 'block';",
                "      });",
                "",
                "      tokenSpan.addEventListener('mouseout', function() {",
                "        tooltip.style.display = 'none';",
                "      });",
                "",
                "      tokenContainer.appendChild(tokenSpan);",
                "    });",
                "",
                "    // Update stats",
                "    tokenCount.textContent = `Total tokens: ${sample.tokens.length}`;",
                "    highlightedCount.textContent = `Highlighted: ${sample.tokens.length}`;",
                "",
                "    // Reset controls",
                "    thresholdSlider.value = 0;",
                "    thresholdValue.textContent = '0.00';",
                "    sortSelect.value = 'position';",
                "  }",
                "",
                "  // Sample navigation",
                "  sampleSlider.addEventListener('input', function() {",
                "    currentSampleIndex = parseInt(this.value);",
                "    renderCurrentSample();",
                "  });",
                "",
                "  prevBtn.addEventListener('click', function() {",
                "    if (currentSampleIndex > 0) {",
                "      currentSampleIndex--;",
                "      sampleSlider.value = currentSampleIndex;",
                "      renderCurrentSample();",
                "    }",
                "  });",
                "",
                "  nextBtn.addEventListener('click', function() {",
                "    if (currentSampleIndex < allSamplesData.length - 1) {",
                "      currentSampleIndex++;",
                "      sampleSlider.value = currentSampleIndex;",
                "      renderCurrentSample();",
                "    }",
                "  });",
                "",
                "  // Threshold slider functionality",
                "  thresholdSlider.addEventListener('input', function() {",
                "    const threshold = parseFloat(this.value) / 100;",
                "    thresholdValue.textContent = threshold.toFixed(2);",
                "    ",
                "    const tokens = document.querySelectorAll('.token');",
                "    let highlightCount = 0;",
                "    tokens.forEach(token => {",
                "      const weight = parseFloat(token.getAttribute('data-weight'));",
                "      if (weight >= threshold) {",
                "        token.style.opacity = '1';",
                "        token.style.filter = 'none';",
                "        highlightCount++;",
                "      } else {",
                "        token.style.opacity = '0.3';",
                "        token.style.filter = 'grayscale(100%)';",
                "      }",
                "    });",
                "    highlightedCount.textContent = `Highlighted: ${highlightCount}`;",
                "  });",
                "",
                "  // Highlight buttons",
                "  highlightHighBtn.addEventListener('click', function() {",
                "    const tokens = document.querySelectorAll('.token');",
                "    const weights = Array.from(tokens).map(t => parseFloat(t.getAttribute('data-weight')));",
                "    const sortedWeights = [...weights].sort((a, b) => b - a);",
                "    const threshold = sortedWeights[Math.floor(sortedWeights.length * 0.25)];",
                "    thresholdSlider.value = Math.round(threshold * 100);",
                "    thresholdValue.textContent = threshold.toFixed(2);",
                "    ",
                "    let highlightCount = 0;",
                "    tokens.forEach(token => {",
                "      const weight = parseFloat(token.getAttribute('data-weight'));",
                "      if (weight >= threshold) {",
                "        token.style.opacity = '1';",
                "        token.style.filter = 'none';",
                "        highlightCount++;",
                "      } else {",
                "        token.style.opacity = '0.3';",
                "        token.style.filter = 'grayscale(100%)';",
                "      }",
                "    });",
                "    highlightedCount.textContent = `Highlighted: ${highlightCount}`;",
                "  });",
                "",
                '  highlightAllBtn.addEventListener("click", function() {',
                "    thresholdSlider.value = 0;",
                '    thresholdValue.textContent = "0.00";',
                "    const tokens = document.querySelectorAll('.token');",
                "    tokens.forEach(token => {",
                '      token.style.opacity = "1";',
                '      token.style.filter = "none";',
                "    });",
                "    highlightedCount.textContent = `Highlighted: ${tokens.length}`;",
                "  });",
                "",
                "  // Sorting functionality",
                "  sortSelect.addEventListener('change', function() {",
                "    const sortBy = this.value;",
                "    const tokens = Array.from(document.querySelectorAll('.token'));",
                "    ",
                "    tokens.sort((a, b) => {",
                "      if (sortBy === 'position') {",
                "        return parseInt(a.getAttribute('data-pos')) - parseInt(b.getAttribute('data-pos'));",
                "      } else if (sortBy === 'weight-desc') {",
                "        return parseFloat(b.getAttribute('data-weight')) - parseFloat(a.getAttribute('data-weight'));",
                "      } else if (sortBy === 'weight-asc') {",
                "        return parseFloat(a.getAttribute('data-weight')) - parseFloat(b.getAttribute('data-weight'));",
                "      }",
                "    });",
                "    ",
                "    tokenContainer.innerHTML = '';",
                "    tokens.forEach(token => tokenContainer.appendChild(token));",
                "  });",
                "",
                "  // Initialize with first sample",
                "  renderCurrentSample();",
                "});",
                "</script>",
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html_output)

    def _create_compact_token_heatmap(
        self, tokens, token_weights, title=None, max_tokens_per_row=50
    ):
        """
        Create a compact heatmap visualization for long token sequences.
        Presents tokens in a grid format with color indicating importance.

        Args:
            tokens: List of token strings
            token_weights: Array of importance values for each token
            title: Optional title for the visualization
            max_tokens_per_row: Maximum number of tokens to display per row

        Returns:
            matplotlib figure
        """
        # Normalize weights to [0, 1] if not already
        if np.max(token_weights) > 1.0 or np.min(token_weights) < 0.0:
            token_weights = (token_weights - np.min(token_weights)) / (
                np.max(token_weights) - np.min(token_weights) + 1e-8
            )

        num_tokens = len(tokens)

        # Calculate the number of rows and columns
        num_rows = int(np.ceil(num_tokens / max_tokens_per_row))
        num_cols = min(max_tokens_per_row, num_tokens)

        # Create a grid of token weights
        grid = np.ones((num_rows, num_cols)) * -1  # -1 for padding
        for i, weight in enumerate(token_weights):
            row = i // max_tokens_per_row
            col = i % max_tokens_per_row
            grid[row, col] = weight

        # Create figure and axes
        fig_width = min(20, max(12, num_cols * 0.2))
        fig_height = min(20, max(3, num_rows * 0.3 + 2))  # +2 for title and colorbar
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create a custom colormap with masked values
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")

        # Mask the -1 values (padding)
        masked_grid = np.ma.masked_where(grid < 0, grid)

        # Create heatmap
        im = ax.imshow(masked_grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(
            im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05
        )
        cbar.set_label("Token Importance", fontsize=10)

        # Set title
        if title:
            ax.set_title(title, fontsize=12, pad=10)

        # Customize ticks and labels
        # Add token labels for columns and rows
        ax.set_xticks(np.arange(num_cols))
        ax.set_yticks(np.arange(num_rows))

        # Set row labels to show token positions
        row_labels = [
            f"{i * max_tokens_per_row}-{min((i + 1) * max_tokens_per_row - 1, num_tokens - 1)}"
            for i in range(num_rows)
        ]
        ax.set_yticklabels(row_labels, fontsize=8)

        # For short sequences, display actual tokens
        if num_tokens <= 100:  # Only show token texts for reasonably sized sequences
            # Add text annotations with token values
            for i in range(num_rows):
                for j in range(num_cols):
                    idx = i * max_tokens_per_row + j
                    if idx < num_tokens:
                        token_text = tokens[idx]
                        token_text = token_text.replace("\n", "⏎").replace("\t", "→")
                        # Truncate long tokens
                        if len(token_text) > 10:
                            token_text = token_text[:8] + "..."

                        # Choose text color based on background darkness
                        weight = token_weights[idx]
                        text_color = "white" if weight > 0.5 else "black"

                        # Add token text
                        ax.text(
                            j,
                            i,
                            token_text,
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=7,
                        )

        # If too many tokens, hide x-axis labels
        if num_cols > 30:
            ax.set_xticklabels([])
        else:
            # Only show some column indices to avoid overcrowding
            step = max(1, num_cols // 10)
            col_indices = np.arange(0, num_cols, step)
            ax.set_xticks(col_indices)
            ax.set_xticklabels([str(i) for i in col_indices], fontsize=8, rotation=45)

        # Add grid lines to separate cells
        ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_pdf_visualization(self, tokens, token_weights, title, pdf_path):
        """
        Create a simple, clean PDF visualization of tokens with saliency highlighting.
        Args:
            tokens: List of token strings
            token_weights: Array of importance values for each token
            title: Title for the visualization
            pdf_path: Path to save the PDF
        """
        import numpy as np
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas

        # Normalize weights to [0, 1]
        token_weights = np.array(token_weights)
        if np.max(token_weights) > 1.0 or np.min(token_weights) < 0.0:
            token_weights = (token_weights - np.min(token_weights)) / (
                np.max(token_weights) - np.min(token_weights) + 1e-8
            )

        # Color mapping: blue for low, red for high
        def weight_to_color(weight):
            # Use a simple blue-red interpolation
            if weight < 0.5:
                # Blue (low)
                r, g, b = 0.8, 0.87, 1.0  # light blue
                r2, g2, b2 = 0.2, 0.4, 1.0  # strong blue
                f = weight * 2
                return colors.Color(
                    r + (r2 - r) * f, g + (g2 - g) * f, b + (b2 - b) * f
                )
            else:
                # Red (high)
                r, g, b = 1.0, 0.8, 0.8  # light red
                r2, g2, b2 = 1.0, 0.2, 0.2  # strong red
                f = (weight - 0.5) * 2
                return colors.Color(
                    r + (r2 - r) * f, g + (g2 - g) * f, b + (b2 - b) * f
                )

        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        margin = 0.75 * inch
        y = height - margin
        x = margin
        line_height = 18
        font_size = 14
        c.setFont("Helvetica", font_size)

        # Title
        if title:
            c.setFont("Helvetica-Bold", font_size + 2)
            c.drawString(x, y, title)
            y -= line_height * 1.5
            c.setFont("Helvetica", font_size)

        # Render tokens as a paragraph, wrapping as needed
        max_width = width - 2 * margin
        space_width = c.stringWidth(" ", "Helvetica", font_size)
        curr_x = x
        curr_y = y
        for token, weight in zip(tokens, token_weights):
            display_token = token.replace("\n", " ").replace("\t", " ")
            if display_token == "":
                display_token = "□"
            token_width = c.stringWidth(display_token, "Helvetica", font_size)
            # Wrap line if needed
            if curr_x + token_width > x + max_width:
                curr_x = x
                curr_y -= line_height
                if curr_y < margin:
                    c.showPage()
                    curr_y = height - margin
            # Draw background
            color = weight_to_color(weight)
            c.setFillColor(color)
            c.rect(
                curr_x - 2,
                curr_y - 3,
                token_width + 2,
                line_height - 2,
                fill=1,
                stroke=0,
            )
            # Draw text
            c.setFillColor(colors.black)
            c.drawString(curr_x, curr_y, display_token)
            curr_x += token_width + space_width
        c.save()

    def _create_png_visualization(self, tokens, token_weights, title, png_path):
        """
        Create a PNG visualization of tokens with saliency highlighting using Pillow.
        Args:
            tokens: List of token strings
            token_weights: Array of importance values for each token
            title: Title for the visualization
            png_path: Path to save the PNG
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        # Percentile-based normalization to avoid outlier effects
        token_weights = np.array(token_weights)
        lower = np.percentile(token_weights, 1)
        upper = np.percentile(token_weights, 99)
        token_weights = np.clip(token_weights, lower, upper)
        token_weights = (token_weights - lower) / (upper - lower + 1e-8)

        # Viridis colormap (256 steps), blended with white for lighter backgrounds
        def viridis_colormap(val, blend=0):
            import matplotlib.cm

            cmap = matplotlib.cm.get_cmap("viridis")
            r, g, b = [int(x * 255) for x in cmap(val)[:3]]
            # Blend with white
            r = int(r + (255 - r) * blend)
            g = int(g + (255 - g) * blend)
            b = int(b + (255 - b) * blend)
            return (r, g, b)

        # Font settings
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        font_size = font.size
        line_height = int(font_size * 1.6)
        margin = 30
        colorbar_width = 40
        colorbar_margin = 80  # Increased from 20 to 80 for more space
        max_width = 1300  # Increased from 1200 to 1300 for more space

        # Prepare lines of tokens (wrap by width)
        lines = []
        curr_line = []
        curr_width = margin
        dummy_img = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy_img)
        for token in tokens:
            display_token = token.replace("\n", " ").replace("\t", " ")
            if display_token == "":
                display_token = "□"
            token_width = draw.textlength(display_token + " ", font=font)
            if (
                curr_width + token_width
                > max_width - margin - colorbar_width - colorbar_margin
            ):
                lines.append(curr_line)
                curr_line = []
                curr_width = margin
            curr_line.append(display_token)
            curr_width += token_width
        if curr_line:
            lines.append(curr_line)

        # Calculate image size
        img_width = max_width
        img_height = margin + line_height * len(lines) + margin
        if title:
            img_height += line_height

        # Create image
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw title
        y = margin
        if title:
            draw.text((margin, y), title, fill=(0, 0, 0), font=font)
            y += line_height

        # Draw tokens with background
        token_idx = 0
        for line in lines:
            x = margin
            for token in line:
                if token_idx >= len(token_weights):
                    break
                weight = token_weights[token_idx]
                bg_color = viridis_colormap(weight, blend=0)
                token_width = draw.textlength(token, font=font)
                # Draw background rectangle
                draw.rectangle(
                    [x - 2, y - 2, x + token_width + 2, y + font_size + 4],
                    fill=bg_color,
                )
                # Always use black text (no outline)
                draw.text(
                    (x, y),
                    token,
                    fill=(0, 0, 0),
                    font=font,
                )
                x += token_width + draw.textlength(" ", font=font)
                token_idx += 1
            y += line_height

        # Draw colorbar
        bar_x = img_width - colorbar_width - colorbar_margin
        bar_y = margin + (line_height if title else 0)
        bar_height = line_height * max(6, len(lines))
        for i in range(bar_height):
            frac = 1.0 - i / bar_height
            color = viridis_colormap(frac, blend=0)
            draw.rectangle(
                [bar_x, bar_y + i, bar_x + colorbar_width, bar_y + i + 1], fill=color
            )
        # Add colorbar labels
        label_font = font
        # Compute label widths for proper placement
        label_1 = "1.0"
        label_0 = "0.0"
        label_title = "Token Importance"
        draw.textlength(label_1, font=label_font)
        draw.textlength(label_0, font=label_font)
        label_title_width = draw.textlength(label_title, font=label_font)
        # Place labels with enough space
        draw.text(
            (bar_x + colorbar_width + 10, bar_y - 8),
            label_1,
            fill=(0, 0, 0),
            font=label_font,
        )
        draw.text(
            (bar_x + colorbar_width + 10, bar_y + bar_height - 8),
            label_0,
            fill=(0, 0, 0),
            font=label_font,
        )
        draw.text(
            (bar_x - (label_title_width // 2) + colorbar_width // 2, bar_y - 32),
            label_title,
            fill=(0, 0, 0),
            font=label_font,
        )

        img.save(png_path)

    def visualize_logit_diff(
        self,
        base_output,
        cf_output,
        inputs,
        step=None,
        mode="logit_diff",
        log_to_console=True,
        enable_visualization=True,
        **kwargs,
    ):
        """
        Visualize logit difference between base and counterfactual outputs.
        For now, just print/log the mean logit diff. Expand as needed.
        """
        if not enable_visualization:
            return
        if base_output is None or cf_output is None:
            if log_to_console:
                print(
                    f"[viz] Warning: base_output or cf_output is None. Skipping logit diff visualization. base_output is None: {base_output is None}, cf_output is None: {cf_output is None}"
                )
            return
        if step is None:
            step = self._current_viz_step
        # Compute logit diff (mean over batch and sequence)
        try:
            cf_logps = torch.log_softmax(cf_output.logits, dim=-1)
            base_logps = torch.log_softmax(base_output.logits, dim=-1)
            logit_diff = cf_logps - base_logps
            # Optionally mask by attention_mask if present
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"]
                logit_diff = logit_diff * mask.unsqueeze(-1)
                mean_logit_diff = logit_diff.sum() / mask.sum()
            else:
                mean_logit_diff = logit_diff.mean()
            if log_to_console:
                print(
                    f"[viz] Logit diff visualization at step {step}: mean diff = {mean_logit_diff.item():.6f}"
                )
            if wandb.run:
                wandb.log(
                    {
                        f"{mode}/mean_logit_diff": mean_logit_diff.item(),
                        f"{mode}/step": step,
                    }
                )
        except Exception as e:
            if log_to_console:
                print(f"[viz] Error in logit diff visualization: {e}")

    def _get_viz_section_name(self, viz_mode):
        """
        Map viz_mode to appropriate wandb section names for organized visualization logging.
        """
        if viz_mode is None:
            return "token_heatmap"

        # Extract the prefix to determine the section
        if viz_mode.startswith("train/"):
            return "train_vis"
        elif viz_mode.startswith("val/"):
            return "val_vis"
        elif viz_mode.startswith("pred/") or viz_mode.startswith("steer/"):
            return "steer_vis"
        else:
            # For any other modes, use a generic visualization section
            return "viz"


atexit.register(lambda: _viz_executor.shutdown(wait=True))
