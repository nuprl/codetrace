import pandas as pd
import matplotlib.pyplot as plt
import ast
import sys
from collections import Counter

# Example DataFrames
scatter_df = pd.read_csv(sys.argv[1])
bubble_df = pd.read_csv(sys.argv[2], encoding="us-ascii")
outfile = sys.argv[3]
bubble_df["types"] = bubble_df["change"].apply(lambda x: ast.literal_eval(x)[0])

# Parse list-like columns
scatter_df["prob_steering_success"] = scatter_df["prob_steering_success"].apply(ast.literal_eval)
scatter_df["prob_typechecks_before"] = scatter_df["prob_typechecks_before"].apply(ast.literal_eval)
scatter_df["mutations"] = scatter_df["mutations"].apply(ast.literal_eval)

# Count occurrences of each type for the same (model, lang, mutation)
bubble_grouped = bubble_df.groupby(["model", "lang", "mutation", "types"]).size().reset_index(name="count")
# max_indices = (
#     bubble_grouped.groupby(["model", "lang", "mutation"])["count"]
#     .transform(max) == bubble_grouped["count"]
# )

bubble_grouped = (
    bubble_grouped.groupby(["model", "lang", "mutation"], group_keys=False)
    .apply(lambda group: group.nlargest(3, "count"))
)
# bubble_grouped = bubble_grouped.loc[top_3_rows]
print(bubble_grouped)
# print(bubble_grouped.loc[max_indices])

# Generate a color map for each unique "type"
unique_types = set(list(bubble_grouped["types"]))

color_map = {t: plt.cm.Paired(i % 10) for i, t in enumerate(unique_types)}

# Overlay the plots
plt.figure(figsize=(16, 10))

# Scatter plot for the first DataFrame
for _, row in scatter_df.iterrows():
    x_values = row["prob_typechecks_before"]
    y_values = row["prob_steering_success"]
    for i, mutation in enumerate(row["mutations"]):
        # Ignore negative x or y values
        if x_values[i] >= 0 and y_values[i] >= 0:
            print(row["model"], row["lang"], mutation, x_values[i], y_values[i])
            plt.scatter(x_values[i], y_values[i], color="black", alpha=0.5, marker="x", label=(row["model"],row["lang"]) if i == 0 else "")
            plt.annotate(mutation, (x_values[i], y_values[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')

print(bubble_grouped)
# Bubble plot for the most common "types"
for _, row in bubble_grouped.iterrows():
    # Find matching data points in the scatter plot
    matching_row = scatter_df[(scatter_df["model"] == row["model"]) & (scatter_df["lang"] == row["lang"])]
    if not matching_row.empty:
        x_values = matching_row.iloc[0]["prob_typechecks_before"]
        y_values = matching_row.iloc[0]["prob_steering_success"]
        if row["mutation"] in matching_row.iloc[0]["mutations"]:
            idx = matching_row.iloc[0]["mutations"].index(row["mutation"])
            # Ignore negative x or y values
            if x_values[idx] >= 0 and y_values[idx] >= 0:
                print(row["model"], row["lang"], row["mutation"], x_values[idx], y_values[idx])
                # Bubble plot using "count" as the size, scaled down
                plt.scatter(
                    x_values[idx], y_values[idx],
                    s=row["count"] * 30, alpha=0.5,  # Adjust bubble size multiplier
                    color=color_map[row["types"]],
                    label=f"{row['types']} (count={row['count']})" if idx == 0 else ""
                )

# Set axis limits and labels
plt.title("Scatter and Bubble Plot with Most Common Types", fontsize=16)
plt.xlabel("Probability typechecks_before", fontsize=14)
plt.ylabel("Probability steering_success", fontsize=14)
plt.xticks([0, 1], fontsize=12)
plt.yticks([0, 1], fontsize=12)
plt.grid(alpha=0.5)


# Sort bubble_grouped by 'count' in descending order
bubble_grouped_sorted = bubble_grouped.sort_values(by="count", ascending=False)

print([row["types"] for _,row in bubble_grouped_sorted.iterrows()])
# Create handles sorted by the most common types
handles = []
seen = set()
for _, row in bubble_grouped_sorted.iterrows():
    if not row['types'] in seen:
        handles.append(
            plt.Line2D([0], [0], marker='o', color=color_map[row["types"]], linestyle='', 
                    label=(row['types'] if row["types"] != "__" else "\__", row['count']))
        )
        seen.add(row['types'])

# Dynamically calculate the number of columns for the legend
n_items = len(handles)
ncol = min(1, n_items)  # Increase max columns to flatten the legend

print([h.get_label() for h in handles])
# Place the legend on the side
lgd = plt.legend(
    handles=handles,
    title="Types",
    loc="center left",  # Place legend on the left of bbox_to_anchor
    bbox_to_anchor=(1.05, 0.5),  # Position legend outside the plot on the right
    fontsize=10,
    ncol=ncol,
    title_fontsize=12,
    borderaxespad=0.5  # Add some padding between the legend and plot
)

# Add space on the right for the legend
plt.tight_layout(rect=[0, 0, 0, 1]) 

plt.savefig(outfile, bbox_extra_artists=(lgd,))
