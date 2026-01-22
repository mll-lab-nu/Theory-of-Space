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