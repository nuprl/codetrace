import matplotlib.pyplot as plt
import pandas as pd
import ast
import sys

# Read the CSV file
csv_file = sys.argv[1]
outfile = sys.argv[2]
df = pd.read_csv(csv_file)

# Extracting the data for plotting
df["prob_steering_success"] = df["prob_steering_success"].apply(ast.literal_eval)
df["prob_typechecks_before"] = df["prob_typechecks_before"].apply(ast.literal_eval)
df["mutations"] = df["mutations"].apply(ast.literal_eval)

# Assign a unique color to each model-language combination
unique_combinations = df[["model", "lang"]].drop_duplicates()
color_map = {f"{row.model}_{row.lang}": plt.cm.tab10(i) 
             for i, row in enumerate(unique_combinations.itertuples(index=False))}

# Plotting
plt.figure(figsize=(12, 8))

# Loop through each row of the DataFrame to plot data
for _, row in df.iterrows():
    model_lang = f"{row['model']}_{row['lang']}"
    color = color_map[model_lang]
    x_values = row["prob_typechecks_before"]
    y_values = row["prob_steering_success"]
    
    # Scatter plot for the current model-language
    plt.scatter(x_values, y_values, color=color, label=model_lang, alpha=0.7)
    
    # Annotating mutations
    for i, mutation in enumerate(row["mutations"]):
        plt.annotate(mutation, (x_values[i], y_values[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')

plt.xlim(0, 1)
plt.ylim(0, 1)

handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=label) 
           for label, color in color_map.items()]
plt.legend(handles=handles, title="Model_Language", loc="best")

# Customizing the plot
plt.xlabel("Probability typechecks_before")
plt.ylabel("Probability steering_success")
plt.grid(alpha=0.5)
plt.savefig(outfile)
plt.legend()
plt.grid()