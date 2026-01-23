from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def _save_fig_to_file(fig, save_path: str, base_dir: Optional[str] = None) -> str:
    """Save figure to file and return relative path
    
    Args:
        fig: The matplotlib figure to save
        save_path: Absolute path to save the figure
        base_dir: Base directory to compute relative path from (e.g., model directory)
    
    Returns:
        Relative path from base_dir if provided, otherwise falls back to heuristic extraction.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    # If base_dir is provided, use it to compute relative path
    if base_dir and os.path.isabs(save_path):
        return os.path.relpath(save_path, base_dir)
    
    # Fallback: Extract the relative path (plots/*.png or <hash>/plots/*.png)
    # The save_path is absolute, but we need to return the part after the model directory
    parts = save_path.split(os.sep)
    # Find the 'plots' directory in the path
    if 'plots' in parts:
        plots_idx = parts.index('plots')
        # If there's a directory before 'plots', include it (the hash)
        if plots_idx > 0 and parts[plots_idx - 1] not in ['results', '']:
            # This is a sample-specific plot: <hash>/plots/*.png
            return os.path.join(parts[plots_idx - 1], 'plots', parts[-1])
        else:
            # This is a summary plot: plots/*.png
            return os.path.join('plots', parts[-1])
    # Fallback: just return the basename
    return os.path.basename(save_path)


def create_infogain_plot(infogain_per_turn: List[float], title: str, save_path: str, base_dir: Optional[str] = None) -> Optional[str]:
    fig, ax = plt.subplots(figsize=(8, 4))
    turns = list(range(1, len(infogain_per_turn) + 1))
    ax.plot(turns, infogain_per_turn, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Average Information Gain')
    ax.set_title(f'Average Information Gain per Turn - {title}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(infogain_per_turn) + 0.5)
    return _save_fig_to_file(fig, save_path, base_dir)


def create_scalar_metric_plot(
    values: List[Optional[float]],
    title: str,
    y_label: str,
    save_path: str,
    ylim: Optional[tuple[float, float]] = (0.0, 1.0),
    base_dir: Optional[str] = None,
) -> Optional[str]:
    if not isinstance(values, list) or len(values) == 0:
        return None
    y = [np.nan if not isinstance(v, (int, float)) else float(v) for v in values]
    if not any(isinstance(v, (int, float)) for v in values):
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    turns = list(range(1, len(y) + 1))
    ax.plot(turns, y, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Turn')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(turns) + 0.5)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    return _save_fig_to_file(fig, save_path, base_dir)


def create_cogmap_metrics_plot(
    series: Dict[str, List[Optional[float]]],
    title: str,
    save_path: str,
    include_dir: bool = True,
    include_facing: bool = True,
    include_pos: bool = True,
    include_overall: bool = True,
    base_dir: Optional[str] = None,
) -> Optional[str]:
    keys = [
        ('dir', include_dir, 'Direction'),
        ('facing', include_facing, 'Facing'),
        ('pos', include_pos, 'Position'),
        ('overall', include_overall, 'Overall'),
    ]

    any_data = any(
        include and isinstance(series.get(k), list) and any(isinstance(v, (int, float)) for v in series.get(k, []))
        for k, include, _ in keys
    )
    if not any_data:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    turns = None
    for k, include, label in keys:
        if not include:
            continue
        vals = series.get(k, [])
        if not isinstance(vals, list) or len(vals) == 0:
            continue
        y = [np.nan if not isinstance(v, (int, float)) else float(v) for v in vals]
        if turns is None:
            turns = list(range(1, len(y) + 1))
        ax.plot(turns, y, marker='o', linewidth=2, markersize=3, label=label)

    if turns is None:
        plt.close(fig)
        return None

    ax.set_xlabel('Turn')
    ax.set_ylabel('Similarity')
    ax.set_title(f'Cognitive Map Similarity per Turn - {title}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(turns) + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    return _save_fig_to_file(fig, save_path, base_dir)


def create_correlation_scatter_plot(x_values: List[float], y_values: List[float],
                                   x_label: str, y_label: str, title: str,
                                   save_path: str,
                                   correlation_info: Dict = None,
                                   base_dir: Optional[str] = None) -> Optional[str]:
    """
    Create a scatter plot to display correlation between two variables.

    Args:
        x_values: X-axis data
        y_values: Y-axis data
        x_label: X-axis label
        y_label: Y-axis label
        title: Chart title
        correlation_info: Correlation information dictionary containing pearson_r, p_value, etc.
        save_path: Path to save the figure

    Returns:
        Chart data URI string or file path, or None if no data available
    """
    if not x_values or not y_values or len(x_values) != len(y_values):
        return None

    # Filter valid data points
    valid_pairs = []
    for x, y in zip(x_values, y_values):
        if (isinstance(x, (int, float)) and isinstance(y, (int, float)) and
            not np.isnan(x) and not np.isnan(y)):
            valid_pairs.append((float(x), float(y)))

    if len(valid_pairs) < 2:
        return None

    valid_x = [pair[0] for pair in valid_pairs]
    valid_y = [pair[1] for pair in valid_pairs]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw scatter plot
    ax.scatter(valid_x, valid_y, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

    # Add trend line
    if len(valid_pairs) >= 2:
        try:
            z = np.polyfit(valid_x, valid_y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(valid_x), max(valid_x), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        except Exception:
            pass

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add correlation information to the chart
    if correlation_info:
        pearson_r = correlation_info.get('pearson_r')
        p_value = correlation_info.get('p_value')
        significant = correlation_info.get('significant', False)

        if pearson_r is not None and p_value is not None:
            sig_text = "***" if significant else "n.s."
            text = f"r = {pearson_r:.3f}, p = {p_value:.3f} {sig_text}"
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=10)

    return _save_fig_to_file(fig, save_path, base_dir)


def create_correlation_plot(x_values: List[float], y_values: List[float],
                          x_label: str, y_label: str, title: str,
                          save_path: str,
                          correlation_info: Dict = None,
                          base_dir: Optional[str] = None) -> Optional[str]:
    """
    Create a correlation scatter plot between two lists.

    Args:
        x_values: X-axis data list
        y_values: Y-axis data list
        x_label: X-axis label
        y_label: Y-axis label
        title: Chart title
        correlation_info: Optional correlation information dictionary, auto-calculated if not provided
        save_path: Path to save the figure
        base_dir: Base directory to compute relative path from (e.g., model directory)

    Returns:
        Chart data URI string or file path, or None if no data available
    """
    if not x_values or not y_values:
        return None

    return create_correlation_scatter_plot(
        x_values, y_values, x_label, y_label, title, correlation_info, save_path, base_dir
    )

