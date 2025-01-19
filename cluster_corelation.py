import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load the Dataset ===
file_path = "combined_metrics.csv"  # Replace with the actual file path
data = pd.read_csv(file_path)

# === Define Metrics for Analysis ===
popularity_metrics = [
    "stars", "forks", "pull_requests", "subscribers", "Tags Count", 
    "Open Issues Count", "Closed Issues Count", "Contributors Count", 
    "Commits Count", "Dependencies", "Usage", "popularity_1_year", 
    "ReleaseFrequency", "Release Count", "Vulnerabilities"
]

# Handle missing values by filling with 0
popularity_data = data[popularity_metrics].fillna(0)

# === Compute Correlation Matrix ===
correlation_matrix = popularity_data.corr()

# Save the correlation matrix to a CSV file
correlation_matrix.to_csv("correlation_matrix.csv")

# === Highlight Specific Metric Pairs ===
highlight_pairs = [
    ("stars", "subscribers", 0.82),
    ("pull_requests", "Closed Issues Count", 0.88),
]

# Plot the heatmap
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,                 # Show correlation values
    fmt=".2f",                  # Format values to 2 decimal places
    cmap="coolwarm",            # Diverging colormap for positive/negative correlations
    cbar=True,                  # Include color bar
    annot_kws={"size": 14},     # Adjust annotation font size
)

# Annotate the specified metric pairs
for metric1, metric2, corr_value in highlight_pairs:
    # Find the positions of the metrics in the matrix
    x = correlation_matrix.columns.tolist().index(metric2)
    y = correlation_matrix.index.tolist().index(metric1)
    
    # Add annotation directly on the heatmap
    plt.text(
        x + 0.5, y + 0.5,  # Position (center of cell)
        f"↔ {corr_value:.2f}",  # Show correlation value
        color="black",       # Text color
        ha="center", va="center", fontsize=12, weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="yellow")  # Highlight with a yellow background
    )

# Add title and labels
# plt.title("Correlation Matrix Heatmap (Highlighted Metrics)", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=14)  # Rotate x-axis labels for clarity
plt.yticks(fontsize=14)  # Adjust y-axis label size
plt.tight_layout()

# Save the heatmap as an image
plt.savefig("correlation_matrix_heatmap_highlighted.png")
plt.show()

# Outputs
print("\n=== Outputs ===")
print("- Correlation matrix saved to: correlation_matrix.csv")
print("- Highlighted heatmap saved to: correlation_matrix_heatmap_highlighted.png")
