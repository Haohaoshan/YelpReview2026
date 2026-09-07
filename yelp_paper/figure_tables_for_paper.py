
#%% 
import matplotlib.pyplot as plt

# ---- Data  ----
features = [
    "action_ratio","adj_count","adv_count","ari","avg_dep_depth","avg_sent_len",
    "avg_word_len","burstiness","char_count","coleman_liau","concreteness_proxy",
    "dale_chall","flesch_reading_ease","hapax_ratio","noun_count","punctuation_count",
    "sent_count","sentiment_proxy","smog_index","token_count","type_lemma_ratio",
    "unique_lemmas","unique_tokens","verb_count"
]

lr_rank = [8,14,24,19,6,15,18,20,2,21,23,9,16,11,13,3,7,17,22,5,12,1,10,4]
shap_rank = [17,22,23,14,6,11,4,20,5,13,19,9,8,12,15,2,16,10,24,21,7,1,3,18]

#%% 
# ---- Plot ----
plt.figure(figsize=(10,8))

for x, y, label in zip(lr_rank, shap_rank, features):
    diff = abs(x - y)

    # ---- Color logic ----
    if x <= 8 and y <= 8:
        color = "green"          # strong agreement, top-tier in both
    elif diff > 8:
        color = "red"            # strong disagreement
    elif diff > 6:
        color = "orange"         # moderate disagreement (amber)
    else:
        color = "steelblue"      # normal case

    plt.scatter(x, y, s=80, color=color)
    plt.text(x + 0.1, y + 0.1, label, fontsize=11)

plt.xlabel("Logistic Coeff Rank")
plt.ylabel("XGB SHAP Rank")
plt.title("LR vs XGB Feature Importance Rank")
plt.grid(True, linestyle="--", alpha=0.4)

# ---- Add 45-degree diagonal reference line ----
min_rank = min(min(lr_rank), min(shap_rank))
max_rank = max(max(lr_rank), max(shap_rank))
plt.plot([min_rank, max_rank], [min_rank, max_rank],
         color="gray", linestyle="--", linewidth=1)

plt.tight_layout()
plt.savefig("fig_lr_shap_rank.png")




# %%
"""============================= updated plot ====================================="""

import matplotlib.pyplot as plt
import numpy as np

# ---- Data from your table ----
features = [
    "action_ratio","adj_count","adv_count","ari","avg_dep_depth","avg_sent_len",
    "avg_word_len","burstiness","char_count","coleman_liau","concreteness_proxy",
    "dale_chall","flesch_reading_ease","hapax_ratio","noun_count","punctuation_count",
    "sent_count","sentiment_proxy","smog_index","token_count","type_lemma_ratio",
    "unique_lemmas","unique_tokens","verb_count"
]

lr_rank = np.array([8,14,24,19,6,15,18,20,2,21,23,9,16,11,13,3,7,17,22,5,12,1,10,4])
shap_rank = np.array([17,22,23,14,6,11,4,20,5,13,19,9,8,12,15,2,16,10,24,21,7,1,3,18])

# ---- Compute perpendicular distance to y=x ----
distances = np.abs(lr_rank - shap_rank) / np.sqrt(2)
print("Distances from y=x line:")
distances_sorted_indices = np.argsort(distances) 
print("Sorted distances (ascending):")
for idx in distances_sorted_indices:
    print(f"{features[idx]:20s}  distance = {distances[idx]:.3f}")


# Print distances with labels
for f, d in zip(features, distances):
    print(f"{f:20s}  distance = {d:.3f}")


 
#%%
# ---- Plot ----
plt.figure(figsize=(10,8))

for x, y, label, d in zip(lr_rank, shap_rank, features, distances):

    # ---- Your exact color logic ----
    if x <= 8 and y <= 8:
        color = "green"
    elif d > 8:
        color = "red"
    elif d > 5:
        color = "orange"
    else:
        color = "steelblue"

    plt.scatter(x, y, s=80, color=color)
    plt.text(x + 0.1, y + 0.1, label, fontsize=10)   # <-- no distance printed

plt.xlabel("Logistic Coeff Rank")
plt.ylabel("XGB SHAP Rank")
plt.xlim(0, 30)
plt.ylim(0, 30) 
plt.title("LR vs XGB Feature Importance Rank (distance-colored)")
plt.grid(True, linestyle="--", alpha=0.4)

# ---- Add 45-degree diagonal reference line ----
plt.plot([0, 30], [0, 30],
         color="gray", linestyle="--", linewidth=1)

plt.tight_layout()
plt.savefig("fig_lr_shap_rank_distance_colored.png")


# %%
