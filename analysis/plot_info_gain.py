"""
Simplified information gain visualization script

Generates two plots:
1. Information gain comparison for Active Text configurations
2. Information gain comparison for Active Vision configurations

Usage:
    python analysis/plot_info_gain.py --results_dir result/ --output_dir plots/
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def read_info_gain_from_models(results_dir: str) -> tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[int]]]]:
    """Read env_data.json files under different model directories and extract
    information gain values and sample end steps (active configurations only).

    Returns:
        tuple: (info_gains, sample_end_steps, cogmap_full, sample_counts_per_turn)
        - info_gains: model_name -> config_name -> list of information gains
        - sample_end_steps: model_name -> config_name -> list of end step numbers
        - cogmap_full: model_name -> config_name -> list of cogmap_full_per_turn averages
        - sample_counts_per_turn: model_name -> config_name -> list of sample counts at each turn
    """
    info_gains = {}
    sample_end_steps = {}
    cogmap_full = {}
    sample_counts_per_turn = {}

    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        env_data_path = os.path.join(model_path, "env_data.json")
        if not os.path.exists(env_data_path):
            continue

        try:
            with open(env_data_path, 'r') as f:
                data = json.load(f)

            model_name = data.get('meta_info', {}).get('model_name', model_dir)
            
            # Skip oracle baselines
            if model_name in ["Qwen3-VL-32B-Instruct", "GLM-4.5V"]:
                continue
            
            exp_summary = data.get('exp_summary', {}).get('group_performance', {})
            cogmap_summary = data.get('cogmap_summary', {}).get('group_performance', {})
            samples = data.get('samples', {})

            model_configs = {}
            model_end_steps = {}
            model_cogmap_full = {}
            model_sample_counts = {}

            # Extract information gain from aggregated experiment summary
            for config_name, config_data in exp_summary.items():
                # Only include configurations that are active
                if "active" not in config_name.lower():
                    continue

                infogain_per_turn = config_data.get('infogain_per_turn')
                if infogain_per_turn is not None and isinstance(infogain_per_turn, list) and len(infogain_per_turn) > 0:
                    model_configs[config_name] = infogain_per_turn
                else:
                    # Fall back to using avg_final_information_gain as a single value
                    final_info_gain = config_data.get('avg_final_information_gain')
                    if final_info_gain is not None:
                        model_configs[config_name] = [final_info_gain]  # convert to list format
            
            # Extract cogmap_full_per_turn from cogmap_summary
            for config_name, cogmap_config_data in cogmap_summary.items():
                # Only include configurations that are active
                if "active" not in config_name.lower():
                    continue
                
                # Navigate to per_turn_metrics to get cogmap_full_per_turn
                per_turn_metrics = cogmap_config_data.get('per_turn_metrics', {})
                cogmap_full_per_turn_dict = per_turn_metrics.get('cogmap_full_per_turn', {})
                
                # Extract the 'overall' key which contains the list of values
                if isinstance(cogmap_full_per_turn_dict, dict) and 'overall' in cogmap_full_per_turn_dict:
                    cogmap_full_per_turn = cogmap_full_per_turn_dict['overall']
                    if isinstance(cogmap_full_per_turn, list) and len(cogmap_full_per_turn) > 0:
                        model_cogmap_full[config_name] = cogmap_full_per_turn

            # For each config, collect per-sample end steps from env_turn_logs
            for config_name in model_configs.keys():
                end_steps = []
                for sample_name, sample_data in samples.items():
                    if config_name in sample_data:
                        config_sample_data = sample_data[config_name]
                        if isinstance(config_sample_data, dict) and 'env_turn_logs' in config_sample_data:
                            turn_logs = config_sample_data['env_turn_logs']
                            if turn_logs and len(turn_logs) > 0:
                                # Use the last turn's 'turn_number' as the end step
                                last_turn_number = turn_logs[-1].get('turn_number', 0)
                                end_steps.append(last_turn_number)

                if end_steps:
                    model_end_steps[config_name] = end_steps
                    
                    # Calculate sample counts per turn
                    if config_name in model_configs:
                        max_turns = len(model_configs[config_name])
                        sample_counts = []
                        for turn_idx in range(max_turns):
                            # Count how many samples are still active at this turn (turn is 1-indexed)
                            count = sum(1 for step in end_steps if step >= turn_idx + 1)
                            sample_counts.append(count)
                        model_sample_counts[config_name] = sample_counts

            if model_configs:
                info_gains[model_name] = model_configs
                sample_end_steps[model_name] = model_end_steps
                cogmap_full[model_name] = model_cogmap_full
                sample_counts_per_turn[model_name] = model_sample_counts
                print(f"Found information gain for {model_name}: {list(model_configs.keys())}")
                print(f"  Sample end steps: {[(config, len(steps)) for config, steps in model_end_steps.items()]}")

        except Exception as e:
            print(f"{env_data_path} error: {e}")

    return info_gains, sample_end_steps, cogmap_full, sample_counts_per_turn


def plot_config_group_line_style(info_gains: Dict[str, Dict[str, List[float]]],
                                sample_end_steps: Dict[str, Dict[str, List[int]]],
                                group_name: str,
                                group_configs: list,
                                save_path: str,
                                cogmap_full: Optional[Dict[str, Dict[str, List[float]]]] = None,
                                plot_cogmap: bool = False,
                                sample_counts_per_turn: Optional[Dict[str, Dict[str, List[int]]]] = None,
                                plot_info_gain: bool = True,
                                cogmap_configs: Optional[List[str]] = None) -> None:
    """Create a line-style plot for a group of configurations (visual style is tuned).
    
    Args:
        cogmap_full: Optional dict containing cogmap_full_per_turn data
        plot_cogmap: Whether to plot cogmap_full data on the same figure
        sample_counts_per_turn: Optional dict containing sample counts at each turn
    """
    if not group_configs:
        return

    # Prepare data - create a separate line for each configuration
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Define color palette and marker styles
    colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc05', '#ff6d00', '#795548', '#673ab7']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles =  ['-', ':', '--', '-.']

    # Collect data across models to determine the maximum number of turns
    max_turns = 0

    # Collect data for all models in this configuration group
    model_data = {}
    for model_name, model_configs in info_gains.items():
        for config in group_configs:
            if config in model_configs:
                infogain_list = model_configs[config]
                # Create a unique key for each model-config combination
                model_config_key = f"{model_name}_{config}"
                model_data[model_config_key] = {
                    'model_name': model_name,
                    'config': config,
                    'data': infogain_list
                }
                max_turns = max(max_turns, len(infogain_list))

    if max_turns == 0 and not (plot_cogmap and cogmap_full):
        return

    # x-axis positions represent step numbers (starting from 1)
    x_positions = range(1, max_turns + 1) if max_turns > 0 else []

    def _alpha_from_counts(sample_counts: Optional[List[int]], idx: int) -> float:
        if not sample_counts:
            return 0.9
        max_samples = max(sample_counts)
        count = sample_counts[idx] if idx < len(sample_counts) else sample_counts[-1]
        alpha = 0.1 + 0.9 * (count / max_samples)
        turn = idx + 1
        if 15 <= turn <= 17:
            alpha *= 1.0 - 0.2 * (turn - 15)
        elif turn > 17:
            alpha *= max(0.05, 0.5 ** (turn - 17))
        return alpha

    # Plot a line for each model-config combination
    line_index = 0
    legend_labels = []
    model_colors = {} 

    for model_config_key, data_info in model_data.items():
        model_name = data_info['model_name']
        config = data_info['config']
        infogain_list = data_info['data']

        # If plotting cogmap, only show GPT and Gemini models in infogain plot too
        if plot_cogmap and cogmap_full:
            if not ('gpt' in model_name.lower() or 'gemini' in model_name.lower()):
                continue

        if infogain_list and plot_info_gain:
            # Create legend label showing only the model short name
            model_short = model_name.split('/')[-1] if '/' in model_name else model_name
            model_short = model_short.replace("gemini-3-pro-preview", "gemini-3-pro")

            # Use a distinct color per plotted line
            line_color = colors[line_index % len(colors)]

            # Prepare x and y data limited to the available length
            x_data = x_positions[:len(infogain_list)]
            y_data = infogain_list

            # Get sample counts for this model-config to adjust alpha
            sample_counts = None
            if sample_counts_per_turn and model_name in sample_counts_per_turn:
                if config in sample_counts_per_turn[model_name]:
                    sample_counts = sample_counts_per_turn[model_name][config]

            # Plot the full line on ax1 with segments of varying alpha
            # Use solid line for infogain when plotting cogmap, otherwise use varied linestyles
            line_style = '-' if (plot_cogmap and cogmap_full) else linestyles[line_index % len(linestyles)]
            
            if sample_counts and len(sample_counts) > 0:
                # Plot line segments with varying alpha based on sample count
                for i in range(len(x_data) - 1):
                    alpha = _alpha_from_counts(sample_counts, i)
                    ax1.plot([x_data[i], x_data[i+1]], [y_data[i], y_data[i+1]],
                            color=line_color,
                            linestyle=line_style,
                            linewidth=2.5,
                            alpha=alpha)
                
                # Add label with a dummy plot
                ax1.plot([], [], color=line_color, linestyle=line_style,
                        linewidth=2.5, alpha=0.8, label=f'{model_short} (InfoGain)')
            else:
                # Fallback to original plotting if no sample counts available
                ax1.plot(x_data, y_data,
                        color=line_color,
                        linestyle=line_style,
                        linewidth=2.5,
                        alpha=0.8,
                        label=f'{model_short} (InfoGain)')

            # Show markers only on odd steps to reduce clutter
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                if x % 2 == 1:  # odd step numbers
                    # Adjust marker alpha based on sample count
                    marker_alpha = _alpha_from_counts(sample_counts, i) if sample_counts else 0.9
                    ax1.plot(x, y,
                            marker=markers[line_index % len(markers)],
                            color=line_color,
                            markersize=4,
                            markerfacecolor='white',
                            markeredgewidth=2,
                            alpha=marker_alpha)

            legend_labels.append(model_short)

            # Store color for this model for later reference
            model_colors[model_name] = line_color

            line_index += 1

    # Customize plot appearance for ax1
    if plot_cogmap and cogmap_full:
        if plot_info_gain:
            ax1.set_title('Accumulated Info Gain & Cognitive Map Coverage', fontsize=14, fontweight='bold', pad=20)
        else:
            ax1.set_title('Cognitive Map Coverage', fontsize=14, fontweight='bold', pad=20)
    else:
        ax1.set_title(f'Accumulated Info Gain', fontsize=14, fontweight='bold', pad=20)
    if plot_info_gain:
        ax1.set_ylabel('Info Gain', fontsize=14, fontweight='bold')
    else:
        ax1.set_ylabel('')
        ax1.set_yticks([])
    ax1.set_xlabel('Steps', fontsize=14, fontweight='bold')

    # Set x-axis ticks - show every other tick (1,3,5,...)
    x_ticks = [x for x in x_positions if x % 2 == 1]  # show 1,3,5,7...
    ax1.set_xticks(x_ticks)
    ax1.tick_params(axis='both', labelsize=14)

    # Add a light grid
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Draw vertical lines indicating sample end steps (only when not plotting cogmap)
    if plot_info_gain and not (plot_cogmap and cogmap_full):
        vline_positions = {}  # track counts per position to offset overlapping lines
        vline_added_to_legend = False  # ensure the line label is added to legend only once

        for model_name, model_end_steps in sample_end_steps.items():
            for config in group_configs:
                if config in model_end_steps:
                    end_steps_list = model_end_steps[config]
                    if end_steps_list:
                        # Compute the 80th percentile of end steps
                        end_steps_sorted = sorted(end_steps_list)
                        median_step = np.percentile(end_steps_sorted, 80)

                        # Compute a small offset to avoid overlapping vertical lines
                        if median_step in vline_positions:
                            vline_positions[median_step] += 1
                            offset = (vline_positions[median_step] - 1) * 0.1 - 0.05
                        else:
                            vline_positions[median_step] = 1
                            offset = 0

                        # Use the model's line color for the vertical marker
                        vline_color = model_colors.get(model_name, 'gray')

                        # Add the first vertical line to the legend as a gray dashed entry
                        if not vline_added_to_legend:
                            ax1.axvline(x=median_step + offset,
                                      color=vline_color,
                                      linestyle='--',
                                      alpha=0.7,
                                      linewidth=2)
                            # Create a separate gray legend entry for sample-end markers
                            ax1.plot([], [], color='gray', linestyle='--',
                                   linewidth=2, label='80% Samples Ends')
                            vline_added_to_legend = True
                        else:
                            ax1.axvline(x=median_step + offset,
                                      color=vline_color,
                                      linestyle='--',
                                      alpha=0.7,
                                      linewidth=2)

    # Compact the legend inside the plot (lower right) for a tighter layout
    # Only show legend if not plotting cogmap (cogmap will have combined legend)
    if plot_info_gain and not (plot_cogmap and cogmap_full):
        ax1.legend(loc='lower right', framealpha=0.85, fancybox=True, shadow=True,
                   fontsize=8, markerscale=0.7, handlelength=1.2, handletextpad=0.4,
                   borderpad=0.3, bbox_to_anchor=(0.98, 0.02), ncol=1)

    # Set y-axis limits and ticks (only if not plotting cogmap, as cogmap will set 0-1 range)
    if plot_info_gain and not (plot_cogmap and cogmap_full):
        all_values = []
        for model_configs in info_gains.values():
            for config in group_configs:
                if config in model_configs:
                    infogain_list = model_configs[config]
                    all_values.extend(infogain_list)

        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min

            # If all values are equal, set a sensible range
            if y_range == 0:
                if y_max == 0:
                    ax1.set_ylim(-0.1, 0.5)
                else:
                    ax1.set_ylim(max(0, y_max - 0.5), y_max + 0.5)
            else:
                ax1.set_ylim(max(0, y_min - y_range * 0.1), y_max + y_range * 0.1)

    # Plot cogmap_full data on the right y-axis if requested
    if plot_cogmap and cogmap_full:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cognitive Map Coverage', fontsize=14, fontweight='bold')
        # Collect cogmap data for the same configurations
        # Only include models with 'gpt' or 'gemini' in their name
        cogmap_model_data = {}
        max_cogmap_turns = 0
        
        for model_name, model_configs in cogmap_full.items():
            # Filter to only include GPT and Gemini models
            if not ('gpt' in model_name.lower() or 'gemini' in model_name.lower()):
                continue
                
            for config in (cogmap_configs or group_configs):
                if config in model_configs:
                    cogmap_list = model_configs[config]
                    model_config_key = f"{model_name}_{config}"
                    cogmap_model_data[model_config_key] = {
                        'model_name': model_name,
                        'config': config,
                        'data': cogmap_list
                    }
                    max_cogmap_turns = max(max_cogmap_turns, len(cogmap_list))
        
        if max_cogmap_turns > 0:
            x_positions_cogmap = range(1, max_cogmap_turns + 1)
            
            # Plot cogmap data using dashed lines with different styling
            cogmap_line_index = 0
            for model_config_key, data_info in cogmap_model_data.items():
                model_name = data_info['model_name']
                config = data_info['config']
                cogmap_list = data_info['data']
                
                if cogmap_list:
                    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                    model_short = model_short.replace("gemini-3-pro-preview", "gemini-3-pro")
                    # Use matching colors from the existing model_colors dict
                    if model_name not in model_colors:
                        model_colors[model_name] = colors[cogmap_line_index % len(colors)]
                    line_color = model_colors.get(model_name)
                    
                    x_data = x_positions_cogmap[:len(cogmap_list)]
                    y_data = cogmap_list
                    
                    # Get sample counts for this model-config to adjust alpha
                    sample_counts = None
                    if sample_counts_per_turn and model_name in sample_counts_per_turn:
                        if config in sample_counts_per_turn[model_name]:
                            sample_counts = sample_counts_per_turn[model_name][config]
                    
                    is_vision = "vision" in config.lower()
                    cogmap_linestyle = ':' if is_vision else '--'
                    cogmap_marker = '^' if is_vision else 'x'
                    cogmap_label = 'Vision' if is_vision else 'Text'
                    # Plot with dashed line on the right axis with varying alpha
                    if sample_counts and len(sample_counts) > 0:
                        # Plot line segments with varying alpha based on sample count
                        for i in range(len(x_data) - 1):
                            alpha = _alpha_from_counts(sample_counts, i)
                            ax2.plot([x_data[i], x_data[i+1]], [y_data[i], y_data[i+1]],
                                    color=line_color,
                                    linestyle=cogmap_linestyle,
                                    linewidth=2.0,
                                    alpha=alpha)
                        
                        # Add label with a dummy plot
                        ax2.plot([], [], color=line_color, linestyle=cogmap_linestyle,
                                linewidth=2.0, alpha=0.7, label=f'{model_short} (CogMap {cogmap_label})')
                    else:
                        # Fallback to original plotting if no sample counts available
                        ax2.plot(x_data, y_data,
                                color=line_color,
                                linestyle=cogmap_linestyle,
                                linewidth=2.0,
                                alpha=0.7,
                                label=f'{model_short} (CogMap {cogmap_label})')
                    
                    # Show markers with varying alpha
                    for i, (x, y) in enumerate(zip(x_data, y_data)):
                        if x % 2 == 1:
                            # Adjust marker alpha based on sample count
                            marker_alpha = _alpha_from_counts(sample_counts, i) if sample_counts else 0.7
                            ax2.plot(x, y,
                                    marker=cogmap_marker,
                                    color=line_color,
                                    markersize=5,
                                    markeredgewidth=1.5,
                                    alpha=marker_alpha)
                    
                    cogmap_line_index += 1
            
            # Set unified y-axis limits to 0-1
            ax2.set_ylim(0, 1.0)
            
            # Show combined legend
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            desired_order = ["gpt-5.2", "gemini-3-pro"]
            legend_items = []
            for handle, label in zip(handles1 + handles2, labels1 + labels2):
                model_name = label.split(" (", 1)[0].lower().replace("gemini-3-pro-preview", "gemini-3-pro")
                legend_items.append((handle, label, model_name))
            def _order_key(item):
                name = item[2]
                for idx, key in enumerate(desired_order):
                    if key in name:
                        return (idx, name)
                return (len(desired_order), name)
            legend_items.sort(key=_order_key)
            ax2.legend([h for h, _, _ in legend_items], [l for _, l, _ in legend_items], loc='lower right', framealpha=0.85, 
                      fancybox=True, shadow=True, fontsize=7, markerscale=0.6, 
                      handlelength=1.5, handletextpad=0.4, borderpad=0.3, 
                      bbox_to_anchor=(0.98, 0.02), ncol=1)

    # Save plot to file
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"save to: {save_path}")
    plt.show()

    # Print summary results
    print(f"\n{group_name} result:")
    for model_name, model_configs in info_gains.items():
        print(f"  {model_name}:")
        for config in group_configs:
            if config in model_configs:
                infogain_list = model_configs[config]
                final_gain = infogain_list[-1] if infogain_list else 0.0
                print(f"    {config}: {len(infogain_list)} turns, final={final_gain:.4f}")




def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description="Plot information gain visualization")
    parser.add_argument("--results_dir", type=str, default="result",
                        help="Directory containing experimental results (default: result)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save output plots (default: current directory)")
    parser.add_argument("--plot_cogmap", action="store_true", default=False,
                        help="Plot cognitive map overlay (default: False)")

    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir
    PLOT_COGMAP = args.plot_cogmap
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read information gain data and sample end steps
    info_gains, sample_end_steps, cogmap_full, sample_counts_per_turn = read_info_gain_from_models(results_dir)
    
    # Add Strategist model data (manually provided)
    strategist_data = [0.1045, 0.1903, 0.2669, 0.3584, 0.4007, 0.4582, 0.5107, 0.602, 0.6484, 0.6855, 0.7253, 0.7652, 0.8244, 0.8771, 0.9202, 0.9463, 0.9642, 0.976, 0.9821, 0.985, 0.9882]
    info_gains["Strategist"] = {"active_text": strategist_data}
    if not info_gains:
        return

    # Collect all configurations present in the data
    all_configs = set()
    for configs in info_gains.values():
        all_configs.update(configs.keys())
    for configs in cogmap_full.values():
        all_configs.update(configs.keys())

    # Define configuration groups
    config_groups = {
        "Active Text": [c for c in all_configs if "text" in c.lower() and "active" in c.lower()],
    }
    vision_cogmap_configs = [c for c in all_configs if "vision" in c.lower() and "active" in c.lower()]

    # Generate plots (line style) for each configuration group
    for group_name, group_configs in config_groups.items():
        save_path = os.path.join(output_dir, "info_gain_across_models.pdf")
        plot_config_group_line_style(
            info_gains,
            sample_end_steps,
            group_name,
            group_configs,
            save_path,
            cogmap_full=cogmap_full,
            plot_cogmap=PLOT_COGMAP,
            sample_counts_per_turn=sample_counts_per_turn,
            plot_info_gain=True,
            cogmap_configs=group_configs + vision_cogmap_configs,
        )


if __name__ == "__main__":
    main()