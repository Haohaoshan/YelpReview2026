#%% 
import pandas as pd
import pandas as pd

df = pd.DataFrame({
    "feature_category": [
        "Lexical","Lexical","Lexical","Lexical","Lexical","Lexical",
        "Syntactic","Syntactic","Syntactic","Syntactic","Syntactic",
        "Readability","Readability","Readability","Readability","Readability",
        "Sentiment",
        "Structural","Structural","Structural","Structural","Structural",
        "Semantic","Semantic"
    ],
    "feature": [
        "avg_word_len","hapax_ratio","token_count","type_lemma_ratio","unique_lemmas","unique_tokens",
        "adj_count","adv_count","noun_count","verb_count","avg_dep_depth",
        "ari","coleman_liau","dale_chall","flesch_reading_ease","smog_index",
        "sentiment_proxy",
        "avg_sent_len","burstiness","char_count","punctuation_count","sent_count",
        "action_ratio","concreteness_proxy"
    ],
    "rank_overall": [
        4,12,21,7,1,3,
        22,23,15,18,6,
        14,13,9,8,24,
        10,
        11,20,5,2,16,
        17,19
    ],
    "rank_grp1": [
        5,13,15,9,2,4,
        19,24,17,22,7,
        12,14,8,10,21,
        3,
        11,18,6,1,23,
        16,20
    ],
    "rank_grp2": [
        8,11,19,5,1,2,
        23,16,14,15,6,
        13,18,12,9,20,
        7,
        10,21,4,3,22,
        24,17
    ],
    "rank_grp3": [
        6,20,23,9,1,3,
        8,18,2,24,11,
        17,19,7,14,21,
        4,
        12,13,5,10,22,
        15,16
    ]
})

df


# %%
import matplotlib.pyplot as plt 
# Helper function for scatter plots
def plot_scatter(x, y, xlabel, ylabel, title, color):
    plt.figure(figsize=(6,5))
    plt.scatter(df[x], df[y], s=70, color=color)
    plt.plot([0,25], [0,25], 'r--', linewidth=1.5)  # diagonal reference line
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot 1: Overall vs Group 1
plot_scatter(
    x="rank_overall",
    y="rank_grp1",
    xlabel="Overall Rank",
    ylabel="Group 1 Rank",
    title="Overall vs Group 1 Rank",
    color="steelblue"
)

# Plot 2: Overall vs Group 2
plot_scatter(
    x="rank_overall",
    y="rank_grp2",
    xlabel="Overall Rank",
    ylabel="Group 2 Rank",
    title="Overall vs Group 2 Rank",
    color="darkgreen"
)

# Plot 3: Overall vs Group 3
plot_scatter(
    x="rank_overall",
    y="rank_grp3",
    xlabel="Overall Rank",
    ylabel="Group 3 Rank",
    title="Overall vs Group 3 Rank",
    color="purple"
)



# %%
# --- equal-size 3×1 layout using GridSpec ---
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 

fig = plt.figure(figsize=(6, 18))   # taller figure for 3 rows
gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.25) 

# Plot definitions
plots = [
    ("rank_grp1", "Group 1 Rank", "steelblue"),
    ("rank_grp2", "Group 2 Rank", "darkgreen"),
    ("rank_grp3", "Group 3 Rank", "purple")
]

for i, (grp_col, ylabel, color) in enumerate(plots):
    ax = fig.add_subplot(gs[i, 0])
    ax.scatter(df["rank_overall"], df[grp_col], s=70, color=color)
    ax.plot([0, 25], [0, 25], "r--", linewidth=1.5)
    ax.set_xlabel("Overall Rank")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Overall vs {ylabel}")

plt.tight_layout()

# --- save as PNG ---
plt.savefig("feature_rank_overall_vs_groups_3x1.png", dpi=300)

plt.show()



# %% 
""" ================ 2 by 2 =========================""" 

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 12))
gs = GridSpec(2, 2, wspace=0.25, hspace=0.30)

plots = [
    ("rank_grp1", "Group 1 Rank", "steelblue"),
    ("rank_grp2", "Group 2 Rank", "darkgreen"),
    ("rank_grp3", "Group 3 Rank", "purple")
]

# --- individual scatter plots ---
for i, (grp_col, ylabel, color) in enumerate(plots):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    ax.scatter(df["rank_overall"], df[grp_col], s=70, color=color, alpha=0.9)
    ax.plot([0, 25], [0, 25], "r--", linewidth=1.5)
    ax.set_xlabel("Overall Rank")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Overall vs {ylabel}")

# --- combined overlay plot in bottom-right ---
ax_combined = fig.add_subplot(gs[1, 1])

for grp_col, ylabel, color in plots:
    ax_combined.scatter(df["rank_overall"], df[grp_col], 
                        s=60, alpha=0.6, label=ylabel, color=color)

ax_combined.plot([0, 25], [0, 25], "r--", linewidth=1.5)
ax_combined.set_xlabel("Overall Rank")
ax_combined.set_ylabel("Segmented Rank")
ax_combined.set_title("Overlay: All Groups")
ax_combined.legend()

plt.tight_layout(pad=0.2)
plt.savefig(
    "feature_rank_overall_vs_groups_2x2_overlay.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.05
)
plt.show()


