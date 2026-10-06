import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from plotnine import (
    aes,
    coord_flip,
    element_text,
    facet_wrap,
    geom_abline,
    geom_bar,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_bw,
    ylim,
)

# Predefined color and marker sequences for consistency
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


def _save_roc_matplotlib(df, save_path):
    """Fallback matplotlib function for saving ROC plots."""
    plt.figure(figsize=(8, 8))
    for model in df["Model"].unique():
        model_data = df[df["Model"] == model]
        plt.plot(model_data["FPR"], model_data["TPR"], label=model, linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_metrics_matplotlib(df, save_path):
    """Fallback matplotlib function for saving metrics plots."""
    metrics = df["Metric"].unique()
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        metric_data = df[df["Metric"] == metric]
        for method in metric_data["Method"].unique():
            method_data = metric_data[metric_data["Method"] == method]
            ax.plot(
                method_data["Factor"],
                method_data["TransformedValue"],
                "o-",
                label=method,
            )

        ax.set_title(metric)
        ax.set_xlabel("Factor")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_accuracy_bars_matplotlib(df, save_path):
    """Fallback matplotlib function for saving accuracy bar plots."""
    plt.figure(figsize=(10, 4))
    bars = plt.bar(df["Method"], df["Accuracy"])

    plt.ylim(0, 1)
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _save_win_rates_matplotlib(df, save_path):
    """Fallback matplotlib function for saving win rate plots."""
    methods = df["Method"].unique()
    outcomes = ["Loss", "Tie", "Win"]
    colors = {"Win": "#a6cee3", "Tie": "#bdbdbd", "Loss": "#fbb4ae"}

    plt.figure(figsize=(8, len(methods) * 0.5 + 2))
    bottom = np.zeros(len(methods))

    for outcome in outcomes:
        outcome_data = df[df["Outcome"] == outcome]["Percentage"].values
        plt.barh(
            methods, outcome_data, left=bottom, label=outcome, color=colors[outcome]
        )
        bottom += outcome_data

    # Add win percentage labels
    win_data = df[df["Outcome"] == "Win"]
    for i, (method, row) in enumerate(win_data.iterrows()):
        plt.text(105, i, f"{row['Percentage']:.1f}%", va="center", ha="left")

    plt.xlabel("Percentage (%)")
    plt.xlim(0, 120)  # Give space for percentage labels
    plt.grid(True, alpha=0.3)
    plt.legend(title="Outcome")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_aggregated_roc(jsonl_data, write_to_path=None, report_to=[], wandb_name=None):
    # Collect ROC data for each model
    metrics_list = [
        aggregated_result["results"]["AUCROCEvaluator"]
        for aggregated_result in jsonl_data
    ]

    # Define common FPR thresholds for interpolation
    common_fpr = np.linspace(0, 1, 100)

    tprs = {}
    aucs = {}
    for metrics in metrics_list:
        for model_name, value in metrics.items():
            fpr = value["roc_curve"]["fpr"]
            tpr = value["roc_curve"]["tpr"]
            auc = value["roc_auc"]

            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # Ensure TPR starts at 0
            if model_name not in tprs:
                tprs[model_name] = []
                aucs[model_name] = []
            tprs[model_name].append(interp_tpr)
            aucs[model_name].append(auc)

    # Prepare data for plotting
    plot_data = []
    for model_name in tprs.keys():
        mean_tpr = np.mean(tprs[model_name], axis=0)
        mean_auc = np.mean(aucs[model_name])
        for fpr, tpr in zip(common_fpr, mean_tpr):
            plot_data.append(
                {
                    "FPR": fpr,
                    "TPR": tpr,
                    "Model": f"{model_name} (AUC = {mean_auc:.2f})",
                }
            )

    df = pd.DataFrame(plot_data)

    # Create the plot
    p = (
        ggplot(df, aes(x="FPR", y="TPR", color="Model"))
        + geom_line(size=1)
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="gray")
        + theme_bw()
        + labs(x="False Positive Rate (FPR)", y="True Positive Rate (TPR)")
        + theme(
            figure_size=(4, 4),
            legend_title=element_text(size=8),
            legend_text=element_text(size=6),
            axis_title=element_text(size=10),
            axis_text=element_text(size=8),
            plot_title=element_text(size=12),
            legend_position="right",
        )
    )

    # Optional: Customize colors if needed
    # COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', ...]  # Define your color palette
    # p += scale_color_manual(values=COLORS)

    # Save or show the plot
    if write_to_path:
        save_path = os.path.abspath(str(write_to_path / "aggregated_roc.png"))
        try:
            print(f"Attempting plotnine save to: {save_path}")
            p.save(save_path, dpi=300, bbox_inches="tight")
            print("Plot saved successfully with plotnine")
        except Exception as e:
            print(f"Plotnine save failed: {str(e)}, trying matplotlib fallback")
            try:
                _save_roc_matplotlib(df, save_path)
                print("Plot saved successfully with matplotlib")
            except Exception as e2:
                print(f"Matplotlib save also failed: {str(e2)}")

    # Report to wandb if wandb_name is provided
    if report_to is not None and "wandb" in report_to:
        # Prepare data for wandb.plot.line_series
        xs = common_fpr.tolist()
        ys = [np.mean(tprs[model], axis=0).tolist() for model in tprs]
        keys = [f"{model} (AUC = {np.mean(aucs[model]):.2f})" for model in tprs]
        wandb.log(
            {
                "latent/roc_curve": wandb.plot.line_series(
                    xs=xs,
                    ys=ys,
                    keys=keys,
                    title="Aggregated ROC Curve",
                    xname="False Positive Rate (FPR)",
                )
            }
        )


def plot_metrics(
    jsonl_data, configs, write_to_path=None, report_to=[], wandb_name=None, mode=None
):
    # Collect data into a list
    data = []
    for config in configs:
        evaluator_name = config["evaluator_name"]
        metric_name = config["metric_name"]
        y_label = config["y_label"]
        use_log_scale = config["use_log_scale"]

        for entry in jsonl_data:
            results = entry.get("results", {}).get(evaluator_name, {})
            for method, res in results.items():
                factors = res.get("factor", [])
                metrics = res.get(metric_name, [])
                # Ensure factors and metrics are lists
                if not isinstance(factors, list):
                    factors = [factors]
                if not isinstance(metrics, list):
                    metrics = [metrics]
                for f, m in zip(factors, metrics):
                    data.append(
                        {
                            "Factor": f,
                            "Value": m,
                            "Method": method,
                            "Metric": y_label,
                            "UseLogScale": use_log_scale,
                        }
                    )

    # Create DataFrame and average metrics
    df = pd.DataFrame(data)
    df = df.groupby(
        ["Method", "Factor", "Metric", "UseLogScale"], as_index=False
    ).mean()

    # Apply log transformation if needed
    df["TransformedValue"] = df.apply(
        lambda row: np.log10(row["Value"]) if row["UseLogScale"] else row["Value"],
        axis=1,
    )

    # Create the plot
    p = (
        ggplot(
            df, aes(x="Factor", y="TransformedValue", color="Method", group="Method")
        )
        + geom_line()
        + geom_point()
        + theme_bw()
        + labs(x="Factor", y="Value")
        + facet_wrap("~ Metric", scales="free_y", nrow=1)  # Plots in a row
        + theme(
            subplots_adjust={"wspace": 0.1},
            figure_size=(1.5 * len(configs), 3),  # Wider for more plots, taller height
            legend_position="right",
            legend_title=element_text(size=4),
            legend_text=element_text(size=6),
            axis_title=element_text(size=6),
            axis_text=element_text(size=6),
            axis_text_x=element_text(rotation=90, hjust=1),  # Rotate x-axis labels
            strip_text=element_text(size=6),
        )
    )

    # Save or show the plot
    if write_to_path:
        save_path = os.path.abspath(str(write_to_path / f"{mode}_plot.png"))
        try:
            print(f"Attempting plotnine save to: {save_path}")
            p.save(save_path, dpi=300, bbox_inches="tight")
            print("Plot saved successfully with plotnine")
        except Exception as e:
            print(f"Plotnine save failed: {str(e)}, trying matplotlib fallback")
            try:
                _save_metrics_matplotlib(df, save_path)
                print("Plot saved successfully with matplotlib")
            except Exception as e2:
                print(f"Matplotlib save also failed: {str(e2)}")

    # Report to wandb if wandb_name is provided
    if report_to is not None and "wandb" in report_to:
        # Separate data by metrics to prepare for wandb line series plotting
        line_series_plots = {}
        for metric in df["Metric"].unique():
            metric_data = df[df["Metric"] == metric]

            xs = metric_data["Factor"].unique().tolist()
            ys = [
                metric_data[metric_data["Method"] == method][
                    "TransformedValue"
                ].tolist()
                for method in metric_data["Method"].unique()
            ]
            keys = [f"{method}" for method in metric_data["Method"].unique()]

            line_series_plots[f"{mode}/{metric}"] = wandb.plot.line_series(
                xs=xs, ys=ys, keys=keys, title=f"{metric}", xname="Factor"
            )
        wandb.log(line_series_plots)


def plot_accuracy_bars(
    jsonl_data, evaluator_name, write_to_path=None, report_to=[], wandb_name=None
):
    # Get unique methods and sort them
    methods = set()
    for entry in jsonl_data:
        methods.update(entry["results"][evaluator_name].keys())
    methods = sorted(list(methods))

    # Initialize data structure for 'Seen' accuracy
    seen_accuracies = {method: [] for method in methods}

    # Collect data from all concepts
    for entry in jsonl_data:
        results = entry["results"][evaluator_name]
        for method in methods:
            if method in results:
                if "macro_avg_accuracy" in results[method]:
                    seen_accuracies[method].append(
                        results[method]["macro_avg_accuracy"]
                    )

    # Calculate means
    seen_means = {
        method: np.mean(vals) if len(vals) > 0 else 0
        for method, vals in seen_accuracies.items()
    }

    # Prepare data for plotting
    data = []
    for method in methods:
        data.append({"Method": method, "Accuracy": seen_means[method]})

    df = pd.DataFrame(data)

    # Create the plot
    p = (
        ggplot(df, aes(x="Method", y="Accuracy", fill="Method"))
        + geom_bar(stat="identity", width=0.7)
        + geom_text(
            aes(label="round(Accuracy, 2)"), va="bottom", size=8, format_string="{:.2f}"
        )
        + ylim(0, 1)  # Set y-axis limits from 0 to 1
        + theme_bw()
        + labs(x="Method", y="Accuracy")
        + theme(
            figure_size=(5, 2),
            legend_position="none",  # Remove legend since 'fill' corresponds to 'Method'
            axis_title=element_text(size=5),
            axis_text=element_text(size=5),
            plot_title=element_text(size=5),
        )
    )

    # Save or show the plot
    if write_to_path:
        save_path = os.path.abspath(
            str(write_to_path / "macro_avg_accuracy_incl_hard_neg.png")
        )
        try:
            print(f"Attempting plotnine save to: {save_path}")
            p.save(save_path, dpi=300, bbox_inches="tight")
            print("Plot saved successfully with plotnine")
        except Exception as e:
            print(f"Plotnine save failed: {str(e)}, trying matplotlib fallback")
            try:
                _save_accuracy_bars_matplotlib(df, save_path)
                print("Plot saved successfully with matplotlib")
            except Exception as e2:
                print(f"Matplotlib save also failed: {str(e2)}")

    if report_to is not None and "wandb" in report_to:
        wandb.log(
            {
                "latent/macro_avg_accuracy_incl_hard_neg": wandb.Image(
                    str(write_to_path / "macro_avg_accuracy_incl_hard_neg.png")
                )
            }
        )


def plot_win_rates(jsonl_data, write_to_path=None, report_to=[], wandb_name=None):
    # Collect methods and baseline models
    methods = set()
    baseline_models = set()
    for entry in jsonl_data:
        winrate_results = entry.get("results", {}).get("WinRateEvaluator", {})
        for method_name, res in winrate_results.items():
            methods.add(method_name)
            baseline_models.add(res.get("baseline_model", "Unknown"))
    methods = sorted(list(methods))
    baseline_models = sorted(list(baseline_models))

    # Assuming all methods are compared against the same baseline
    if len(baseline_models) == 1:
        baseline_model = baseline_models[0]
    else:
        # Handle multiple baselines if necessary
        baseline_model = baseline_models[0]  # For now, take the first one

    # Add the baseline method to methods if not already present
    if baseline_model not in methods:
        methods.append(baseline_model)

    # Initialize data structures
    win_rates = {method: [] for method in methods}
    loss_rates = {method: [] for method in methods}
    tie_rates = {method: [] for method in methods}

    # Collect data from all concepts
    num_concepts = len(jsonl_data)
    for entry in jsonl_data:
        winrate_results = entry.get("results", {}).get("WinRateEvaluator", {})
        for method in methods:
            if method == baseline_model:
                continue  # Handle baseline separately
            if method in winrate_results:
                res = winrate_results[method]
                win_rates[method].append(res.get("win_rate", 0) * 100)
                loss_rates[method].append(res.get("loss_rate", 0) * 100)
                tie_rates[method].append(res.get("tie_rate", 0) * 100)
            else:
                # If method is not present in this concept, assume zero rates
                win_rates[method].append(0.0)
                loss_rates[method].append(0.0)
                tie_rates[method].append(0.0)

    # For the baseline method, set win_rate=50%, loss_rate=50%, tie_rate=0%
    win_rates[baseline_model] = [50.0] * num_concepts
    loss_rates[baseline_model] = [50.0] * num_concepts
    tie_rates[baseline_model] = [0.0] * num_concepts

    # Calculate mean percentages
    win_means = {method: np.mean(vals) for method, vals in win_rates.items()}
    loss_means = {method: np.mean(vals) for method, vals in loss_rates.items()}
    tie_means = {method: np.mean(vals) for method, vals in tie_rates.items()}

    # Sort methods: baseline at top, then methods by descending win rate
    non_baseline_methods = [m for m in methods if m != baseline_model]
    sorted_methods = sorted(
        non_baseline_methods, key=lambda m: win_means[m], reverse=True
    )

    # Prepare data for plotting
    data = []
    for method in sorted_methods:
        data.append(
            {"Method": method, "Outcome": "Loss", "Percentage": loss_means[method]}
        )
        data.append(
            {"Method": method, "Outcome": "Tie", "Percentage": tie_means[method]}
        )
        data.append(
            {"Method": method, "Outcome": "Win", "Percentage": win_means[method]}
        )

    df = pd.DataFrame(data)

    # Set the order of Outcome to control stacking order
    df["Outcome"] = pd.Categorical(
        df["Outcome"], categories=["Loss", "Tie", "Win"], ordered=True
    )
    # Reverse the methods list for coord_flip to display baseline at the top
    df["Method"] = pd.Categorical(
        df["Method"], categories=sorted_methods[::-1], ordered=True
    )

    # Ensure df is sorted properly
    df = df.sort_values(["Method", "Outcome"])
    # Convert 'Percentage' to float
    df["Percentage"] = df["Percentage"].astype(float)

    # Compute cumulative percentage per method
    df["cum_percentage"] = df.groupby("Method")["Percentage"].cumsum()
    # Shift cumulative percentages per method
    df["cum_percentage_shifted"] = (
        df.groupby("Method")["cum_percentage"].shift(1).fillna(0)
    )

    # For the 'Win' outcome, get the cumulative percentage up to before 'Win'
    df_win = df[df["Outcome"] == "Win"].copy()
    df_win["text_position"] = df_win["cum_percentage_shifted"]
    # Convert 'text_position' to float
    df_win["text_position"] = 100.0 - df_win["text_position"].astype(float)
    # Format the win percentage label
    df_win["win_percentage_label"] = df_win["Percentage"].map(lambda x: f"{x:.1f}%")

    # Create the plot
    p = (
        ggplot(df, aes(x="Method", y="Percentage", fill="Outcome"))
        + geom_bar(stat="identity", position="stack", width=0.8)
        +
        # Add the geom_text layer to include win rate numbers
        geom_text(
            data=df_win,
            mapping=aes(x="Method", y="text_position", label="win_percentage_label"),
            ha="right",
            va="center",
            size=6,  # Adjust size as needed
            color="black",
            nudge_y=18,  # Adjust this value as needed for proper positioning
        )
        + coord_flip()  # Flip coordinates for horizontal bars
        + theme_bw()
        + labs(y="Percentage (%)", x="")
        + theme(
            axis_text_x=element_text(size=6),
            axis_text_y=element_text(size=6),
            axis_title=element_text(size=6),
            legend_title=element_text(size=6),
            legend_text=element_text(size=6),
            figure_size=(3, len(sorted_methods) * 0.3 + 0.3),
        )
        + scale_fill_manual(
            values={"Win": "#a6cee3", "Tie": "#bdbdbd", "Loss": "#fbb4ae"},
            guide="legend",
            name="Outcome",
        )
    )

    # Save or show the plot
    if write_to_path:
        save_path = os.path.abspath(str(write_to_path / "winrate_plot.png"))
        try:
            print(f"Attempting plotnine save to: {save_path}")
            p.save(save_path, dpi=300, bbox_inches="tight")
            print("Plot saved successfully with plotnine")
        except Exception as e:
            print(f"Plotnine save failed: {str(e)}, trying matplotlib fallback")
            try:
                _save_win_rates_matplotlib(df, save_path)
                print("Plot saved successfully with matplotlib")
            except Exception as e2:
                print(f"Matplotlib save also failed: {str(e2)}")

    if report_to is not None and "wandb" in report_to:
        wandb.log(
            {
                "steering/winrate_plot": wandb.Image(
                    str(write_to_path / "winrate_plot.png")
                )
            }
        )
