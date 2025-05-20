"""
Reporting and visualization for hydro-forecast-ch evaluation.

Generates:
- Text reports
- HTML reports
- Plots: predictions vs actuals, residuals, feature importance
- Comparison tables
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import compute_all_metrics
from .backtest import BacktestResult

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


def plot_predictions_vs_actuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    title: str = "Predictions vs Actuals",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot predictions vs actuals over time.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_lower: Lower prediction interval
        y_upper: Upper prediction interval
        dates: Date array for x-axis
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = len(y_true)
    x = dates if dates is not None else np.arange(n)

    # Plot actuals
    ax.plot(x, y_true, "k-", label="Actual", linewidth=1.5, alpha=0.8)

    # Plot predictions
    ax.plot(x, y_pred, "b-", label="Predicted", linewidth=1.5, alpha=0.8)

    # Plot prediction interval
    if y_lower is not None and y_upper is not None:
        ax.fill_between(x, y_lower, y_upper, alpha=0.2, color="blue", label="90% PI")

    ax.set_xlabel("Date" if dates is not None else "Time step")
    ax.set_ylabel("Capacity Factor")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    if dates is not None:
        fig.autofmt_xdate()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot residual analysis.

    Creates 4-panel plot:
    - Residuals over time
    - Residual histogram
    - Q-Q plot
    - Residuals vs predicted
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Residuals over time
    x = dates if dates is not None else np.arange(len(residuals))
    axes[0, 0].plot(x, residuals, "b-", alpha=0.6)
    axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Date" if dates is not None else "Time step")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].set_title("Residuals over Time")
    axes[0, 0].grid(True, alpha=0.3)

    # Residual histogram
    axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Residual Distribution")

    # Normal overlay
    from scipy import stats
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
                    "r-", label="Normal")
    axes[0, 1].legend()

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")

    # Residuals vs Predicted
    axes[1, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].set_title("Residuals vs Predicted")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved residual plot to {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    df = importance_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar

    bars = ax.barh(range(len(df)), df["importance"], color="steelblue", edgecolor="white")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["importance"] + 0.01 * df["importance"].max(), i,
                f"{row['importance']:.3f}", va="center", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to {save_path}")

    return fig


def plot_horizon_metrics(
    horizon_metrics: pd.DataFrame,
    metrics: List[str] = ["nrmse", "nse", "kge", "skill_score"],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot metrics across forecast horizons.

    Args:
        horizon_metrics: DataFrame from compute_metrics_per_horizon
        metrics: Metrics to plot
        figsize: Figure size
        save_path: Path to save

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if metric not in horizon_metrics.columns:
            continue

        ax = axes[i]
        x = range(len(horizon_metrics))
        ax.plot(x, horizon_metrics[metric], "o-", linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(horizon_metrics["horizon"])
        ax.set_xlabel("Horizon")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} by Horizon")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved horizon metrics plot to {save_path}")

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ["nrmse", "nse", "kge", "skill_score"],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot model comparison as grouped bar chart.

    Args:
        comparison_df: DataFrame from compare_models with model column and metrics
        metrics: Metrics to plot
        figsize: Figure size
        save_path: Path to save

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    models = comparison_df["model"].tolist()
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.8 / n_metrics if n_metrics > 1 else 0.8

    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for i, metric in enumerate(metrics):
        if metric not in comparison_df.columns:
            continue

        ax = axes[i]
        values = comparison_df[metric].values

        bars = ax.bar(x, values, width, label=metric, color=colors[i], edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} Comparison")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved model comparison plot to {save_path}")

    return fig


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    coverage: float = 0.9,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot prediction interval calibration.

    Shows actual coverage vs nominal coverage.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sort by prediction
    sort_idx = np.argsort(y_pred)
    y_true_s = y_true[sort_idx]
    y_pred_s = y_pred[sort_idx]
    y_lower_s = y_lower[sort_idx]
    y_upper_s = y_upper[sort_idx]

    # Plot 1: Intervals
    ax = axes[0]
    n = min(100, len(y_true_s))  # Plot subset for clarity
    idx = np.linspace(0, len(y_true_s) - 1, n, dtype=int)

    x = np.arange(n)
    ax.plot(x, y_true_s[idx], "ko", markersize=3, label="Actual")
    ax.plot(x, y_pred_s[idx], "bo", markersize=3, label="Predicted")
    ax.fill_between(x, y_lower_s[idx], y_upper_s[idx], alpha=0.3, color="blue", label=f"{int(coverage*100)}% PI")

    ax.set_xlabel("Sample (sorted by prediction)")
    ax.set_ylabel("Capacity Factor")
    ax.set_title("Prediction Intervals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Coverage by quantile
    ax = axes[1]

    # Compute empirical coverage for different intervals
    quantiles = np.arange(0.1, 1.0, 0.1)
    coverages = []

    # This requires full quantile predictions - simplified here
    # Just show nominal vs actual for the given interval
    actual_cov = np.mean((y_true >= y_lower) & (y_true <= y_upper))

    ax.bar(["Nominal", "Actual"], [coverage, actual_cov], color=["gray", "steelblue"], edgecolor="white")
    ax.set_ylabel("Coverage")
    ax.set_title(f"Interval Coverage (Nominal: {int(coverage*100)}%)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    ax.text(0, coverage + 0.02, f"{coverage:.2f}", ha="center")
    ax.text(1, actual_cov + 0.02, f"{actual_cov:.2f}", ha="center")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved prediction interval plot to {save_path}")

    return fig


def generate_evaluation_report(
    result: BacktestResult,
    output_dir: Union[str, Path],
    model_name: Optional[str] = None,
) -> Path:
    """
    Generate comprehensive evaluation report.

    Creates:
    - HTML report with metrics, plots, and tables
    - JSON summary
    - Individual plot files

    Args:
        result: BacktestResult
        output_dir: Output directory
        model_name: Model name (defaults to result.model_name)

    Returns:
        Path to HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_name or result.model_name

    # Generate HTML report
    html_path = output_dir / f"{model_name}_evaluation_report.html"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{model_name} Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; }}
            .good {{ color: #27ae60; }}
            .bad {{ color: #e74c3c; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>{model_name} Evaluation Report</h1>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Aggregated Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    """

    for metric, value in result.aggregated_metrics.items():
        css_class = "good" if (metric in ["nse", "kge", "skill_score", "picp"] and value > 0) or \
                            (metric in ["nrmse", "mae", "mpiw"] and value < 0.3) else "bad"
        html_content += f"<tr><td>{metric}</td><td class='metric {css_class}'>{value:.4f}</td></tr>"

    html_content += """
        </table>

        <h2>Fold Details</h2>
        <table>
            <tr><th>Fold</th><th>Train Size</th><th>Test Size</th><th>Train Period</th><th>Test Period</th></tr>
    """

    for fold in result.fold_results:
        html_content += f"""
            <tr>
                <td>{fold['fold']}</td>
                <td>{fold['train_size']}</td>
                <td>{fold['test_size']}</td>
                <td>{fold['train_start']} to {fold['train_end']}</td>
                <td>{fold['test_start']} to {fold['test_end']}</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>Plots</h2>
        <div class="plot">
            <img src="{model_name}_predictions.png" alt="Predictions vs Actuals">
            <p>Predictions vs Actuals</p>
        </div>
        <div class="plot">
            <img src="{model_name}_residuals.png" alt="Residuals">
            <p>Residual Analysis</p>
        </div>
        <div class="plot">
            <img src="{model_name}_feature_importance.png" alt="Feature Importance">
            <p>Feature Importance</p>
        </div>
        <div class="plot">
            <img src="{model_name}_horizon_metrics.png" alt="Horizon Metrics">
            <p>Metrics by Horizon</p>
        </div>
    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html_content)

    logger.info(f"Generated evaluation report at {html_path}")
    return html_path


def save_metrics_json(
    result: BacktestResult,
    output_path: Union[str, Path],
) -> Path:
    """Save metrics to JSON file."""
    import json

    output_path = Path(output_path)

    # Convert to serializable format
    data = {
        "model_name": result.model_name,
        "aggregated_metrics": {k: float(v) if not np.isnan(v) else None
                               for k, v in result.aggregated_metrics.items()},
        "fold_results": result.fold_results,
        "metadata": result.metadata,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved metrics to {output_path}")
    return output_path


def create_summary_table(
    results: Dict[str, BacktestResult],
    metrics: List[str] = ["nrmse", "nse", "kge", "skill_score", "picp", "mpiw"],
) -> pd.DataFrame:
    """
    Create summary comparison table from multiple model results.

    Args:
        results: Dict of model_name -> BacktestResult
        metrics: Metrics to include

    Returns:
        DataFrame with models as rows, metrics as columns
    """
    rows = []

    for name, result in results.items():
        row = {"model": name}
        for metric in metrics:
            row[metric] = result.aggregated_metrics.get(metric, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Format for display
    for metric in metrics:
        if metric in df.columns:
            df[metric] = df[metric].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")

    return df