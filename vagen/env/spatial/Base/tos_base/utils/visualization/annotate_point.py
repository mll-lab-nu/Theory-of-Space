#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw points or text labels on a top-down image using mask coordinates from meta_data.json.
The mapping is read from the `topdown_map` field inside `meta_data.json` (no separate topdown_map.json required).

Quick Debug:
python -m vagen.env.spatial.Base.tos_base.utils.visualization.annotate_point \
    --image vagen/env/spatial/room_data_3_room_new/run01/top_down_empty.png \
    --map vagen/env/spatial/room_data_3_room_new/run01/meta_data.json \
    --labels '{"0,1": "A", "0,2": "B"}' \
    --agent '2,3' \
    --out tmp.png
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
from PIL import Image, ImageDraw, ImageFont


def load_mapping_from_meta(meta_path: Path) -> tuple:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    topdown = data.get("topdown_map")
    if not topdown:
        raise ValueError("meta_data.json is missing the 'topdown_map' field")
    mapping = {}
    for item in topdown.get("mapping", []):
        r, c = item.get("row"), item.get("col")
        pixel = item.get("pixel", {})
        if r is None or c is None:
            continue
        if "x" not in pixel or "y" not in pixel:
            continue
        mapping[(r, c)] = (float(pixel["x"]), float(pixel["y"]))
    rows = topdown.get("rows", 0)
    cols = topdown.get("cols", 0)
    return mapping, rows, cols


def draw_point(img_path: Path, out_path: Path, mapping: Dict, label_dict: Dict, rows: int, cols: int, agent_pos: Optional[Tuple[int, int]] = None, output_size: Optional[Tuple[int, int]] = None):
    """
    Draw multiple annotations on the image.

    Args:
        img_path: Path to the input image
        out_path: Path where the output image will be saved
        mapping: Mapping from position to pixel coordinates, format {(row, col): (x, y)}
        label_dict: Mapping from position to label, format {(row, col): "A"} or {(row, col): None}
                    If the value is None, a red dot will be drawn instead of text.
        rows: Number of rows in the grid
        cols: Number of cols in the grid
        agent_pos: Agent position (row, col) to draw specifically.
        output_size: Optional size to resize the output image (width, height).
    """
    img = Image.open(img_path).convert("RGBA")
    
    # Calculate scaling factors
    base_img_size = 512
    base_grid_size = 15
    img_size = max(img.size)
    grid_size = max(rows, cols)
    scale = (img_size / base_img_size) * (base_grid_size / grid_size)
    
    # Scale parameters
    radius = int(4 * scale)
    font_size = int(18 * scale)
    offset_x = int(8 * scale)
    offset_y = int(20 * scale)
    
    draw = ImageDraw.Draw(img)
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    
    # 1. Draw all points in mapping as grey dots
    for pixel in mapping.values():
        x, y = pixel
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(128, 128, 128, 255))

    # 2. Draw labeled points (candidates) as red
    for position, label in label_dict.items():
        if position not in mapping:
            continue

        x, y = mapping[position]
        
        # Always draw the red point for candidates
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0, 255))
        
        if label:
            # Draw label next to the point (offset to top-right) to avoid covering the point
            draw.text((x + offset_x, y - offset_y), label, fill=(255, 0, 0, 255), font=font)

    # 3. Draw agent if provided
    if agent_pos and agent_pos in mapping:
        x, y = mapping[agent_pos]
        # Draw agent as a blue dot (slightly larger)
        agent_radius = radius + 1
        draw.ellipse((x - agent_radius, y - agent_radius, x + agent_radius, y + agent_radius), fill=(0, 0, 255, 255))
    
    if output_size:
        print(f"[DEBUG] resizing image to {output_size}")
        img = img.resize(output_size, Image.Resampling.LANCZOS)

    img.save(out_path)
    print(f"✅ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate a top-down image with dots or letters using mask coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", required=True, help="Path to top_down.png")
    parser.add_argument("--map", dest="map_path", required=True, help="Path to meta_data.json (must contain topdown_map)")
    parser.add_argument("--labels", type=str, help='Label dictionary JSON string, e.g. \'{"0,1": "A", "0,2": "B", "1,1": null}\'')
    parser.add_argument("--labels-file", type=str, help="Path to a JSON file containing the label dictionary")
    parser.add_argument("--agent", type=str, help="Agent position 'row,col', e.g. '2,3'")
    parser.add_argument("--out", default=None, help="Output path. Defaults to the same directory as the image with suffix _marked.png")
    args = parser.parse_args()

    image_path = Path(args.image)
    map_path = Path(args.map_path)

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"mapping not found: {map_path}")

    # Parse the labels dictionary
    if args.labels_file:
        label_data = json.loads(Path(args.labels_file).read_text(encoding="utf-8"))
    elif args.labels:
        label_data = json.loads(args.labels)
    else:
        # If no labels provided, use empty dict (valid if we just want to show agent + grey dots)
        label_data = {}
    
    # Convert string keys to tuple keys
    label_dict = {}
    for key, value in label_data.items():
        parts = key.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid position key: {key}. Expected 'row,col' format")
        row, col = int(parts[0]), int(parts[1])
        label_dict[(row, col)] = value

    mapping, rows, cols = load_mapping_from_meta(map_path)
    
    # Verify all label positions exist in the mapping
    for position in label_dict.keys():
        if position not in mapping:
            raise ValueError(f"coordinate {position} not found in mapping")

    # Parse agent
    agent_pos = None
    if args.agent:
        parts = args.agent.split(",")
        if len(parts) == 2:
            agent_pos = (int(parts[0]), int(parts[1]))

    out_path = Path(args.out) if args.out else image_path.with_name(image_path.stem + "_marked.png")
    draw_point(image_path, out_path, mapping, label_dict, rows, cols, agent_pos)


if __name__ == "__main__":
    main()
