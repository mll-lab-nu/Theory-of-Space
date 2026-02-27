import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import Blues

# --------------------
# Data
# --------------------
models = ["GPT-5.2", "Gemini-3 Pro", "Claude-4.5 Sonnet", "GLM-4.6V", "Qwen3-VL"]

data = {
    ("Active",  "Text"):   [72.0, 81.5, 68.8, 18.9, 36.8],
    ("Passive", "Text"):   [90.4, 86.5, 75.4, 27.1, 45.0],
    ("Active",  "Vision"): [46.0, 57.3, 32.1, 16.0, 21.4],
    ("Passive", "Vision"): [57.1, 60.5, 44.8, 18.0, 24.9],
}
df = pd.DataFrame(data, index=models)
df01 = df / 100.0

# --------------------
# Highlight switch (choose ONE of: "Active", "Passive", "Text", "Vision")
# --------------------
highlight_mode = "Passive"
# highlight_mode = "Active"
# highlight_mode = "Text"
# highlight_mode = "Vision"
# highlight_mode = ""

dim_alpha = 0.1
label_dimmed = False

# --------------------
# Plot
# --------------------
x = np.arange(len(models))
bar_w = 0.18
offsets = [-1.5*bar_w, -0.5*bar_w, 0.5*bar_w, 1.5*bar_w]
series = list(df01.columns)

blue_levels = [0.85, 0.70, 0.55, 0.40]
colors = [Blues(lvl) for lvl in blue_levels]

fig, ax = plt.subplots(figsize=(13, 6))

def is_highlight(col, mode):
    # col = (Active/Passive, Text/Vision)
    if mode in ("Active", "Passive"):
        return col[0] == mode
    if mode in ("Text", "Vision"):
        return col[1] == mode
    return True

for i, col in enumerate(series):
    y = df01[col].values
    mask = ~np.isnan(y)

    hi = is_highlight(col, highlight_mode)
    alpha = 1.0 if hi else dim_alpha

    bars = ax.bar(
        x[mask] + offsets[i],
        y[mask],
        width=bar_w,
        color=colors[i],
        edgecolor="none",
        alpha=alpha,
        label=f"{col[0]} · {col[1]}"
    )

    if hi or label_dimmed:
        for rect, val in zip(bars, y[mask]):
            ax.text(
                rect.get_x() + rect.get_width()/2,
                rect.get_height() + 0.015,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=10, alpha=alpha
            )

# Styling
ax.set_title("Avg Performance across models (Active/Passive × Text/Vision)", fontsize=20)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=True, fontsize=12)
ax.set_xlabel("Models", fontsize=16)
ax.set_ylabel("Avg Performance", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.35)

# Make legend entries match highlight/dim state
if highlight_mode:
    for handle, text in zip(leg.legend_handles, leg.get_texts()):
        t = text.get_text()
        mode0, mode1 = [s.strip() for s in t.split("·")]
        hi = (highlight_mode in ("Active", "Passive") and mode0 == highlight_mode) or \
             (highlight_mode in ("Text", "Vision") and mode1 == highlight_mode)
        a = 1.0 if hi else dim_alpha
        handle.set_alpha(a)
        text.set_alpha(a)
else:
    # no highlight -> all opaque
    for handle, text in zip(leg.legend_handles, leg.get_texts()):
        handle.set_alpha(1.0)
        text.set_alpha(1.0)

plt.tight_layout()
plt.savefig(f"avg_perform_{highlight_mode.lower()}.pdf", dpi=200, bbox_inches="tight")
plt.show()





# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # --------------------
# # 1. Data Preparation
# # --------------------
# # X-axis categories
# categories = ["GPT-5.2\n(Text)", "GPT-5.2\n(Vision)", "Gemini-3 Pro\n(Text)", "Gemini-3 Pro\n(Vision)"]

# # Data mapping: (Room Type, Mode) -> [Scores per category]
# data = {
#     ("2-room", "Passive"): [92.3, 59.3, 86.7, 58.3],
#     ("2-room", "Active"):  [77.8, 51.5, 80.6, 57.8],
#     ("4-room", "Passive"): [86.5, 52.6, 81.2, 56.2],
#     ("4-room", "Active"):  [66.0, 40.3, 77.7, 51.5],
# }

# df = pd.DataFrame(data, index=categories)
# df01 = df / 100.0  # Convert to 0.0 - 1.0 scale

# # --------------------
# # 2. Plot Configuration
# # --------------------
# # Professional color palette (Teal for 2-room, Royal Blue for 4-room)
# color_palette = [
#     "#4DB6AC", # 2-room Passive (Teal Light)
#     "#00796B", # 2-room Active  (Teal Dark)
#     "#64B5F6", # 4-room Passive (Blue Light)
#     "#1565C0"  # 4-room Active  (Blue Dark)
# ]

# x = np.arange(len(categories))
# bar_w = 0.18
# # Offsets to group bars: 2-room group on left, 4-room group on right
# offsets = [-1.6*bar_w, -0.6*bar_w, 0.6*bar_w, 1.6*bar_w]
# series = list(df01.columns)

# fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)

# # --------------------
# # 3. Core Plotting Logic
# # --------------------
# bars_groups = []

# for i, col in enumerate(series):
#     y_values = df01[col].values
    
#     # Plot bars
#     bars = ax.bar(
#         x + offsets[i],
#         y_values,
#         width=bar_w,
#         color=color_palette[i],
#         edgecolor="white",
#         linewidth=0.6,
#         zorder=3,
#         label=f"{col[0]} · {col[1]}"
#     )
#     bars_groups.append(bars)

#     # Add value labels above bars
#     for rect, val in zip(bars, y_values):
#         height = rect.get_height()
#         ax.text(
#             rect.get_x() + rect.get_width()/2,
#             height + 0.01,
#             f"{val:.1%}",
#             ha="center", va="bottom", 
#             fontsize=8, fontweight='semibold', color='#333333'
#         )

# # --------------------
# # 4. Gap Widening Annotation
# # --------------------
# def annotate_gap_widening(idx):
#     """
#     Visualizes the increasing gap between Passive and Active modes
#     when moving from 2-room to 4-room environments.
#     """
#     # Get bar objects for the current model group
#     b2p = bars_groups[0][idx] # 2-room Passive
#     b2a = bars_groups[1][idx] # 2-room Active
#     b4p = bars_groups[2][idx] # 4-room Passive
#     b4a = bars_groups[3][idx] # 4-room Active

#     # Calculate Gap percentages
#     gap_2r = (b2p.get_height() - b2a.get_height())
#     gap_4r = (b4p.get_height() - b4a.get_height())
    
#     # X coordinates
#     x_2r = (b2p.get_x() + b2a.get_x() + bar_w) / 2
#     x_4r = (b4p.get_x() + b4a.get_x() + bar_w) / 2
    
#     # Y coordinates (midpoint of the gap)
#     y_2r_mid = (b2p.get_height() + b2a.get_height()) / 2
#     y_4r_mid = (b4p.get_height() + b4a.get_height()) / 2

#     annotation_color = '#C62828' # Professional Dark Red

#     # 1. Draw vertical gap indicators
#     # 2-room
#     ax.plot([x_2r, x_2r], [b2a.get_height(), b2p.get_height()], 
#             color=annotation_color, lw=1.5, linestyle='-', alpha=0.8, zorder=4)
#     # 4-room
#     ax.plot([x_4r, x_4r], [b4a.get_height(), b4p.get_height()], 
#             color=annotation_color, lw=1.5, linestyle='-', alpha=0.8, zorder=4)

#     # 2. Draw connecting arrow (Gap Widens)
#     # flattened arc to fit within y=1.0 limit
#     arrow = mpatches.FancyArrowPatch(
#         (x_2r + bar_w*0.5, y_2r_mid), 
#         (x_4r - bar_w*0.5, y_4r_mid),
#         arrowstyle='-|>,head_width=3,head_length=6', 
#         color=annotation_color, lw=1.5, zorder=5, 
#         connectionstyle="arc3,rad=-0.1" 
#     )
#     ax.add_patch(arrow)
    
#     # 3. Add Text Label
#     # Positioned between the bars to save vertical space
#     mid_x = (x_2r + x_4r) / 2
#     mid_y = max(y_2r_mid, y_4r_mid) + 0.02
    
#     label_text = f"Gap Widens:\n{gap_2r:.1%} $\\rightarrow$ {gap_4r:.1%}"
    
#     ax.text(mid_x, mid_y, label_text, 
#             color=annotation_color, fontsize=7.5, fontweight='bold',
#             ha='center', va='bottom', 
#             bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.9, ec="none"))

# # Apply annotation only to GPT models (Index 0 and 1)
# annotate_gap_widening(0) # GPT Text
# annotate_gap_widening(1) # GPT Vision

# # --------------------
# # 5. Styling and Layout
# # --------------------
# # Title and Labels
# ax.set_title("Performance Degradation & Gap Amplification in Complex Environments", 
#              fontsize=16, fontweight='bold', pad=35, color='#212121')
# ax.set_ylabel("Success Rate", fontsize=12, fontweight='medium', color='#424242')

# # Axis formatting
# ax.set_xticks(x)
# ax.set_xticklabels(categories, fontsize=11, fontweight='medium', color='#212121')
# ax.set_ylim(0, 1.0) # Strictly constrained to 1.0 as requested

# # Grid and Spines
# ax.grid(axis="y", linestyle="--", linewidth=0.5, color='#BDBDBD', alpha=0.5, zorder=0)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_linewidth(0.8)
# ax.spines['bottom'].set_linewidth(0.8)
# ax.tick_params(axis='both', which='major', labelsize=10, color='#424242')

# # Legend Layout
# # Placed between title and plot to avoid overlap and save vertical space
# leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), 
#                 ncol=4, frameon=False, fontsize=12, 
#                 handletextpad=0.5, columnspacing=1.5)

# plt.tight_layout()
# plt.savefig("results.png", dpi=300, bbox_inches="tight")