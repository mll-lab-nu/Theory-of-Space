"""
Scatter plot visualization for exploration cost vs performance

Generates four scatter plots:
1. Text (Active + Passive): Exploration cost vs evaluation accuracy
2. Vision (Active + Passive): Exploration cost vs evaluation accuracy
3. Active Text & Vision: Exploration cost vs evaluation accuracy
4. All (Active + Passive, Text + Vision): Exploration cost vs evaluation accuracy

Usage:
    python analysis/plot_cost_performance.py --results_dir /path/to/results --output_dir /path/to/output --data_dir /path/to/data
"""

import os
import json
import sys
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerBase
from PIL import Image

# Add vagen to path for importing get_exploration_history_stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vagen.env.spatial.Base.tos_base.managers.agent_proxy import get_exploration_history_stats


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


class ImageHandler(HandlerBase):
    """Custom legend handler for images"""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        if isinstance(orig_handle, OffsetImage):
            bb = AnnotationBbox(
                orig_handle, (width / 2, height / 2),
                frameon=False, xycoords=trans, box_alignment=(0.5, 0.5)
            )
            return [bb]
        return []


def _apply_alpha_scale(img: Image.Image, alpha_scale: float) -> Image.Image:
    """Apply alpha scaling to an RGBA image."""
    if alpha_scale >= 1.0:
        return img
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    r, g, b, a = img.split()
    a = a.point(lambda p: int(p * alpha_scale))
    return Image.merge('RGBA', (r, g, b, a))


def _load_rgba_cropped(img_path: str) -> Image.Image:
    img = Image.open(img_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


def preprocess_badge_legend_icon(img_path: str, target_size: int = 256) -> Image.Image:
    """
    For badge legend only:
    crop -> keep aspect ratio resize -> pad to square canvas (avoid deformation).
    """
    img = _load_rgba_cropped(img_path)

    # Keep aspect ratio
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    # Pad to square
    canvas = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
    x = (target_size - img.size[0]) // 2
    y = (target_size - img.size[1]) // 2
    canvas.paste(img, (x, y), img)
    return canvas


def preprocess_icon(
    img_path: str,
    target_size: int = 256,
    badge_path: Optional[str] = None,
    alpha_scale: float = 1.0
) -> Image.Image:
    """
    Load, crop transparent borders, resize icon, and optionally add a badge.

    NOTE: alpha_scale is applied to the *final composed* image so that the badge
    uses the same alpha as the main icon (e.g., passive points faded -> badge faded).
    """
    img = Image.open(img_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Keep original behavior: force square resize for model icons
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    composed = img

    if badge_path and os.path.exists(badge_path):
        badge = Image.open(badge_path).convert("RGBA")

        # Badge size: keep original behavior
        badge_size = int(target_size / 1.8)
        badge = badge.resize((badge_size, badge_size), Image.Resampling.LANCZOS)

        padding = int(target_size * 0.15)
        new_size = target_size + padding
        canvas = Image.new('RGBA', (new_size, new_size), (0, 0, 0, 0))

        canvas.paste(img, (0, padding))
        canvas.paste(badge, (new_size - badge_size, 0), badge)
        composed = canvas

    return _apply_alpha_scale(composed, alpha_scale)


def read_cost_performance_from_models(
    results_dir: str,
    data_dir: Optional[str] = None
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Read env_data.json files and extract exploration cost and evaluation accuracy."""
    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    scout_cost = 0.0
    strategist_cost = 0.0

    MODEL_MAPPING = {
        "claude-sonnet-4-5": "Claude-4.5-Sonnet",
        "gemini-3-pro-preview": "Gemini-3-Pro",
        "qwen/qwen3-vl-235b-a22b-thinking": "Qwen3-VL",
        "glm-4.6v": "GLM-4.6V",
        "gpt-5.2": "GPT-5.2",
    }

    EXCLUDE_MODEL_KEYWORDS = ["internvl"]

    if data_dir and os.path.exists(data_dir):
        try:
            print(f"Calculating baseline costs from {data_dir}...")
            scout_cost = get_exploration_history_stats(data_dir, 'scout').get('avg_action_cost', 0.0)
            strategist_cost = get_exploration_history_stats(data_dir, 'strategist').get('avg_action_cost', 0.0)
            print(f"Baseline costs: Scout={scout_cost:.2f}, Strategist={strategist_cost:.2f}")
        except Exception as e:
            print(f"Warning: Failed to load baseline costs: {e}")

    if not os.path.exists(results_dir):
        print(f"Warning: results_dir does not exist: {results_dir}")
        return data

    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path) or any(k in model_dir.lower() for k in EXCLUDE_MODEL_KEYWORDS):
            continue

        env_data_path = os.path.join(model_path, "env_data.json")
        if not os.path.exists(env_data_path):
            continue

        try:
            with open(env_data_path, 'r') as f:
                json_data = json.load(f)

            raw_name = json_data.get('meta_info', {}).get('model_name', model_dir)
            if any(k in str(raw_name).lower() for k in EXCLUDE_MODEL_KEYWORDS):
                continue

            model_name = MODEL_MAPPING.get(raw_name, raw_name)
            if any(k in str(model_name).lower() for k in EXCLUDE_MODEL_KEYWORDS):
                continue

            exp_summary = json_data.get('exp_summary', {}).get('group_performance', {})
            eval_summary = json_data.get('eval_summary', {}).get('group_performance', {})
            model_configs: Dict[str, Tuple[float, float]] = {}

            def passive_cost(config_name: str) -> Optional[float]:
                name = config_name.lower()
                if "text" in name and strategist_cost > 0:
                    return strategist_cost
                if "vision" in name and scout_cost > 0:
                    return scout_cost
                return exp_summary.get(config_name, {}).get('avg_action_cost')

            # Active (from exp_summary)
            for config_name, config_data in exp_summary.items():
                if "active" not in config_name.lower():
                    continue
                avg_cost = config_data.get('avg_action_cost')
                avg_acc = eval_summary.get(config_name, {}).get('avg_accuracy')
                if avg_cost is not None and avg_acc is not None:
                    model_configs[config_name] = (avg_cost, avg_acc)

            # Passive (from baseline or exp_summary fallback)
            for config_name, eval_data in eval_summary.items():
                if "passive" not in config_name.lower():
                    continue
                avg_cost = passive_cost(config_name)
                avg_acc = eval_data.get('avg_accuracy')
                if avg_cost is not None and avg_acc is not None:
                    model_configs[config_name] = (avg_cost, avg_acc)

            # Baselines
            if scout_cost > 0:
                for k, v in eval_summary.items():
                    if "scout" in k.lower() and "vision" in k.lower():
                        acc = v.get('avg_accuracy')
                        if acc is not None:
                            model_configs["Vision Scout"] = (scout_cost, acc)
                            break

            if strategist_cost > 0:
                for k, v in eval_summary.items():
                    if "strategist" in k.lower() and "text" in k.lower():
                        acc = v.get('avg_accuracy')
                        if acc is not None:
                            model_configs["Text Strategist"] = (strategist_cost, acc)
                            break

            if model_configs:
                data[str(model_name)] = model_configs

        except Exception as e:
            print(f"{env_data_path} error: {e}")

    return data


def plot_scatter(
    data: Dict[str, Dict[str, Tuple[float, float]]],
    config_filter: str,
    title: str,
    save_path: str,
    connect_points: bool = True,
    show_badges: bool = False,
    alpha_rules: Optional[List[Tuple[str, float]]] = None,
) -> None:
    """Create a scatter plot for cost vs performance."""
    _, ax = plt.subplots(figsize=(6.5, 5))

    icon_dir = os.path.join(os.path.dirname(__file__), "icon")
    model_icon_map = {
        'claude': 'claude.png',
        'gemini': 'gemini.png',
        'glm': 'glm.png',
        'gpt': 'gpt.png',
        'qwen': 'qwen.png'
    }

    ICON_ZOOM_DEFAULT = 0.14
    ICON_ZOOM_RULES = [
        ("claude-4.5-sonnet", 0.14),
        ("gemini 3 pro", 0.14),
        ("qwen3 vl", 0.14),
        ("glm-4.6v", 0.14),
        ("gpt-5.2", 0.14),
    ]

    def get_icon_zoom(name: str) -> float:
        s = name.lower()
        for key, z in ICON_ZOOM_RULES:
            if key in s:
                return z
        return ICON_ZOOM_DEFAULT

    def resolve_alpha(config_name: str) -> float:
        if not alpha_rules:
            return 1.0
        name = config_name.lower()
        for key, alpha in alpha_rules:
            if key in name:
                return alpha
        return 1.0

    colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc05', '#ff6d00', '#795548', '#673ab7', '#9c27b0']
    all_points: List[dict] = []
    model_index = 0

    for model_name, model_configs in sorted(data.items()):
        filtered = {
            k: v for k, v in model_configs.items()
            if config_filter.lower() in k.lower()
            and any(s in k.lower() for s in ["active", "passive"])
        }
        if not filtered:
            continue

        color = colors[model_index % len(colors)]
        icon_path = next(
            (os.path.join(icon_dir, f) for k, f in model_icon_map.items() if k in model_name.lower()),
            None
        )

        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        ax.scatter([], [], s=100, c=color, marker='o', edgecolors='black', linewidths=1.5, label=model_short)

        pts = []
        for config_name, (cost, acc) in filtered.items():
            badge_path = None
            if show_badges:
                if "vision" in config_name.lower():
                    badge_path = os.path.join(icon_dir, "image.png")
                elif "text" in config_name.lower():
                    badge_path = os.path.join(icon_dir, "text.png")

            all_points.append({
                'x': cost,
                'y': acc,
                'icon_path': icon_path,
                'icon_zoom': get_icon_zoom(model_name),
                'color': color,
                'badge_path': badge_path,
                'alpha_scale': resolve_alpha(config_name),
                'is_active': "active" in config_name.lower()
            })
            pts.append((cost, acc))

        if connect_points and len(pts) > 1:
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linestyle='--', linewidth=1.5, alpha=0.4)

        model_index += 1

    if not all_points:
        print(f"No points to plot for {config_filter}.")
        plt.close()
        return

    xs = [p['x'] for p in all_points]
    all_points.sort(key=lambda p: (p['x'], p['y'], 0 if not p['is_active'] else 1))

    for p in all_points:
        x, y = p['x'], p['y']
        if p['icon_path'] and os.path.exists(p['icon_path']):
            img = preprocess_icon(
                p['icon_path'],
                badge_path=p.get('badge_path'),
                alpha_scale=p.get('alpha_scale', 1.0)
            )
            imagebox = OffsetImage(img, zoom=p['icon_zoom'] * 0.6)
            ax.add_artist(AnnotationBbox(
                imagebox, (x, y),
                xybox=(x, y),
                boxcoords="data",
                frameon=False,
                pad=0,
                arrowprops=dict(arrowstyle="-", color="black", alpha=0.3, lw=0.5)
            ))
        else:
            ax.scatter(x, y, s=150, c=p['color'], marker='o', edgecolors='black', linewidths=1.5, zorder=2)

    # Style
    # ax.set_title(title, fontsize=16, fontweight='bold', pad=40)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=0, y = 1.06)
    ax.set_xlabel('Exploration Cost', fontsize=14, fontweight='bold')
    ax.set_ylabel('Evaluation Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    # ax.set_ylim([0, 1.05])
    ax.set_ylim([0, 0.65])

    if xs:
        margin = (max(xs) - min(xs)) * 0.15
        ax.set_xlim([max(0, min(xs) - margin), max(xs) + margin])

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '{:.0%}'.format(v)))

    # Legend (models)
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = []
    for h, lab in zip(handles, labels):
        icon_path = next(
            (os.path.join(icon_dir, f) for k, f in model_icon_map.items() if k in lab.lower()),
            None
        )
        if icon_path and os.path.exists(icon_path):
            img = preprocess_icon(icon_path)
            custom_handles.append(OffsetImage(img, zoom=get_icon_zoom(lab) * 0.35))
        else:
            custom_handles.append(h)

    model_legend = ax.legend(
        custom_handles, labels,
        loc='lower center',
        fontsize=10, framealpha=0.9,
        handler_map={OffsetImage: ImageHandler()},
        ncol=len(custom_handles), borderpad=0.25, labelspacing=0.25,
        handletextpad=0.35, columnspacing=0.6, borderaxespad=0.15,
        bbox_to_anchor=(0.5, 1.12),
        bbox_transform=ax.transAxes,
    )
    # Badge legend (top-left) for all_strategies (show_badges=True)
    if show_badges:
        badge_specs = [
            ("Image", os.path.join(icon_dir, "image.png")),
            ("Text", os.path.join(icon_dir, "text.png")),
        ]

        # Match the size ratio used when badge is placed on the model icon
        TARGET_SIZE = 256
        padding = int(TARGET_SIZE * 0.15)
        new_size = TARGET_SIZE + padding
        badge_size = int(TARGET_SIZE / 1.8)
        BADGE_TO_ICON_RATIO = badge_size / float(new_size)

        # Model legend zoom baseline ~ ICON_ZOOM_DEFAULT * 0.35; scale by badge ratio
        BADGE_LEGEND_ZOOM = (ICON_ZOOM_DEFAULT * 0.65) * BADGE_TO_ICON_RATIO

        badge_handles, badge_labels = [], []
        for lab, pth in badge_specs:
            if os.path.exists(pth):
                badge_img = preprocess_badge_legend_icon(pth, target_size=TARGET_SIZE)
                badge_handles.append(OffsetImage(badge_img, zoom=BADGE_LEGEND_ZOOM))
                badge_labels.append(lab)

        if badge_handles:
            ax.legend(
                badge_handles, badge_labels,
                loc='upper left', bbox_to_anchor=(0.0, 1.02),
                fontsize=10, framealpha=0.9,
                handler_map={OffsetImage: ImageHandler()},
                ncol=len(badge_handles), borderpad=0.25, labelspacing=0.25,
                handletextpad=0.35, columnspacing=0.6, borderaxespad=0.15
            )
            ax.add_artist(model_legend)

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    print(f"Saved scatter plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate cost-performance scatter plots.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Reading data from {args.results_dir}...")
    data = read_cost_performance_from_models(args.results_dir, args.data_dir)

    if not data:
        print("No data found!")
        return

    print("\nGenerating scatter plots...")

    # Text: active + passive, no badges, passive faded
    plot_scatter(
        data,
        "text",
        "Text: Evaluation Performance vs Exploration Cost",
        os.path.join(args.output_dir, "scatter_active_text.png"),
        connect_points=False,
        show_badges=False,
        alpha_rules=[("passive", 0.5)]
    )

    # Vision: active + passive, no badges, passive faded
    plot_scatter(
        data,
        "vision",
        "Vision: Evaluation Performance vs Exploration Cost",
        os.path.join(args.output_dir, "scatter_active_vision.png"),
        connect_points=False,
        show_badges=False,
        alpha_rules=[("passive", 0.5)]
    )

    # Text + Vision (active only), no badges, text faded
    plot_scatter(
        data,
        "active",
        "Text + Vision (Active): Evaluation Performance vs Exploration Cost",
        os.path.join(args.output_dir, "scatter_active_combined.png"),
        connect_points=False,
        show_badges=False,
        alpha_rules=[("text", 0.5)]
    )

    # All: active + passive, text + vision, badges, passive faded
    plot_scatter(
        data,
        "",
        "All: Evaluation Performance vs Exploration Cost",
        os.path.join(args.output_dir, "scatter_all_strategies.png"),
        connect_points=False,
        show_badges=True,
        alpha_rules=[("passive", 0.3)]
    )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()