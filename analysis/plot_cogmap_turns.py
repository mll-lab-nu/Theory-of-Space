#!/usr/bin/env python3
"""
Visualize turn-by-turn cognitive map predictions vs ground truth.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
import hashlib

def get_object_color(obj_name):
    """
    Generate a consistent color for an object based on its name.
    """
    obj_name_lower = obj_name.lower()
    
    # Special colors
    if obj_name_lower == 'agent':
        return '#E74C3C'  # red
    elif 'init' in obj_name_lower or 'initial' in obj_name_lower:
        return '#000000'  # black
    
    # Generate color from hash of object name for consistency
    hash_object = hashlib.md5(obj_name.encode())
    hash_hex = hash_object.hexdigest()
    
    # Use first 6 characters as RGB color
    r = int(hash_hex[0:2], 16)
    g = int(hash_hex[2:4], 16)
    b = int(hash_hex[4:6], 16)
    
    # Ensure colors are not too dark (minimum brightness)
    r = max(r, 80)
    g = max(g, 80)
    b = max(b, 80)
    
    return f'#{r:02x}{g:02x}{b:02x}'


def plot_grid(ax, objects, title, x_range, y_range):
    """
    Plot objects on a grid.
    """
    # Set limits based on computed ranges
    ax.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)
    ax.set_ylim(y_range[0] - 0.5, y_range[1] + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Create ticks for grid
    x_ticks = np.arange(np.floor(x_range[0]) - 0.5, np.ceil(x_range[1]) + 1.5, 1)
    y_ticks = np.arange(np.floor(y_range[0]) - 0.5, np.ceil(y_range[1]) + 1.5, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4) # Reduced pad
    
    # Set background to light gray
    ax.set_facecolor('#F5F5F5')
    
    # Group objects by position to handle overlapping
    pos_to_objects = {}
    for obj in objects:
        name = obj.get('name', 'unknown')
        pos = obj.get('pos', [0, 0])
        
        # Skip doors/gates
        if 'door' in name.lower() or 'gate' in name.lower():
            continue
        
        pos_key = (float(pos[0]), float(pos[1]))
        if pos_key not in pos_to_objects:
            pos_to_objects[pos_key] = []
        pos_to_objects[pos_key].append(obj)
    
    # Always add initial position at (0, 0)
    init_pos_key = (0.0, 0.0)
    init_obj = {'name': 'initial_position', 'pos': [0.0, 0.0]}
    if init_pos_key not in pos_to_objects:
        pos_to_objects[init_pos_key] = []
    # Add initial position first
    pos_to_objects[init_pos_key].insert(0, init_obj)
    
    # Plot objects, handling overlapping
    for pos_key, objs in pos_to_objects.items():
        x, y = pos_key
        n_objs = len(objs)
        
        if n_objs == 1:
            # Single object
            obj = objs[0]
            color = get_object_color(obj['name'])
            rect = Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, 
                           facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
        elif n_objs == 2:
            # Two objects - split vertically
            for i, obj in enumerate(objs):
                color = get_object_color(obj['name'])
                if i == 0:
                    rect = Rectangle((x - 0.4, y - 0.4), 0.4, 0.8, 
                                   facecolor=color, edgecolor='black', linewidth=1)
                else:
                    rect = Rectangle((x, y - 0.4), 0.4, 0.8, 
                                   facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
        else:
            # More than 2 objects - pie chart
            for i, obj in enumerate(objs):
                color = get_object_color(obj['name'])
                angle_start = i * 360 / n_objs
                angle_end = (i + 1) * 360 / n_objs
                wedge = patches.Wedge((x, y), 0.4, angle_start, angle_end,
                                    facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(wedge)
        
        # Add orientation indicator for agent
        for obj in objs:
            if obj.get('name') == 'agent' and 'ori' in obj:
                ori = obj['ori']
                dx, dy = float(ori[0]) * 0.3, float(ori[1]) * 0.3
                ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.15, 
                       fc='black', ec='black', linewidth=2, zorder=10)


def plot_cogmap_comparison(json_file, output_file=None, max_turns=10):
    """
    Plot predicted vs ground truth cognitive maps for multiple turns.
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Filter turns
    turns_with_cogmap = [turn for turn in data if 'cogmap_log' in turn and 
                         'global' in turn['cogmap_log'] and 
                         turn['cogmap_log']['global']]
    
    turns_to_plot = turns_with_cogmap[:max_turns]
    n_turns = len(turns_to_plot)
    
    if n_turns == 0:
        print("No turns with cognitive map data found.")
        return
    
    # Calculate grid bounds
    min_x, max_x = 0, 0
    min_y, max_y = 0, 0
    
    all_object_names = set()
    
    for turn in turns_to_plot:
        cogmap = turn['cogmap_log']['global']
        for state_key in ['pred_room_state', 'gt_room_state']:
            if state_key in cogmap and 'objects' in cogmap[state_key]:
                for obj in cogmap[state_key]['objects']:
                    name = obj.get('name', 'unknown')
                    if 'door' not in name.lower() and 'gate' not in name.lower():
                        all_object_names.add(name)
                        pos = obj.get('pos', [0, 0])
                        x, y = float(pos[0]), float(pos[1])
                        min_x = min(min_x, x)
                        max_x = max(max_x, x)
                        min_y = min(min_y, y)
                        max_y = max(max_y, y)
    
    all_object_names.add('agent')
    all_object_names.add('initial_position')
    sorted_names = sorted(all_object_names)
    
    # Limits and Ranges
    x_range = (min_x, max_x)
    y_range = (min_y, max_y)
    
    # FIX: Add 1.0 to span to account for the +/- 0.5 grid padding
    x_span = max(max_x - min_x + 1.0, 1)
    y_span = max(max_y - min_y + 1.0, 1)
    
    # --- Figure Size Calculation ---
    # We fix the width of each subplot and calculate the necessary height
    # to satisfy the Aspect Ratio while respecting the top margin for the Legend.
    
    subplot_width = 3.2     # Inches per subplot width
    margin_top = 0.82       # Leave space for legend (0 to 1 scale)
    margin_bottom = 0.02    # Small bottom margin
    vertical_fraction_for_plots = margin_top - margin_bottom
    
    # Aspect Ratio of the data (Height / Width)
    data_aspect_ratio = y_span / x_span
    
    # Calculate required figure height
    # Height_pixels = Width_pixels * Aspect_Ratio
    # Total_Fig_Height * Vertical_Fraction = (2_Rows * Subplot_Width * Aspect_Ratio)
    required_fig_height = (2 * subplot_width * data_aspect_ratio) / vertical_fraction_for_plots
    
    # Ensure a minimum height so legend fits even if map is very flat
    required_fig_height = max(required_fig_height, 3.0) 
    
    total_fig_width = subplot_width * n_turns
    
    fig, axes = plt.subplots(2, n_turns, figsize=(total_fig_width, required_fig_height))
    
    if n_turns == 1:
        axes = axes.reshape(2, 1)
    
    # Plot each turn
    for i, turn in enumerate(turns_to_plot):
        turn_num = turn['turn_number']
        cogmap = turn['cogmap_log']['global']
        
        pred_objects = []
        if 'pred_room_state' in cogmap and 'objects' in cogmap['pred_room_state']:
            pred_objects = cogmap['pred_room_state']['objects']
        
        gt_objects = []
        if 'gt_room_state' in cogmap and 'objects' in cogmap['gt_room_state']:
            gt_objects = cogmap['gt_room_state']['objects']
        
        plot_grid(axes[0, i], pred_objects, f'Turn {turn_num} - Pred', x_range, y_range)
        plot_grid(axes[1, i], gt_objects, f'Turn {turn_num} - GT', x_range, y_range)
    
    # Legend
    legend_elements = []
    for name in sorted_names:
        color = get_object_color(name)
        legend_elements.append(patches.Patch(facecolor=color, edgecolor='black', label=name))
    
    ncol = (len(sorted_names) + 1) // 2 
    # Use global figure legend
    fig.legend(handles=legend_elements, loc='upper center', ncol=ncol, 
               bbox_to_anchor=(0.5, 0.995), frameon=True, fontsize=9, columnspacing=2.0)
    
    # --- Layout Adjustment ---
    # wspace=0.0 makes columns touch
    # hspace=0.2 makes rows have more spacing (pred vs gt)
    plt.subplots_adjust(
        left=0.01, 
        right=0.99, 
        top=margin_top, 
        bottom=margin_bottom, 
        wspace=0.0, 
        hspace=0.2
    )
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize cognitive map predictions vs ground truth over turns')
    parser.add_argument('json_file', type=str, 
                        help='Path to exploration_turn_logs.json')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (optional)')
    parser.add_argument('--max-turns', '-n', type=int, default=7,
                        help='Maximum number of turns to display')
    
    args = parser.parse_args()
    
    if not Path(args.json_file).exists():
        print(f"Error: Input file {args.json_file} not found.")
        return
    
    plot_cogmap_comparison(args.json_file, args.output, args.max_turns)


if __name__ == '__main__':
    main()