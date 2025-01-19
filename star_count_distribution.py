import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "combined_metrics.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Define the ranges for the stars count
bins = [0, 10, 100, 1000, 10000, float('inf')]
labels = ['1-10', '11-100', '101-1000', '1001-10000', '10000+']

# Map the stars count to the defined ranges
df['Stars Range'] = pd.cut(df['stars'], bins=bins, labels=labels, right=False)

# Group by the ranges and count the occurrences
grouped_data = df['Stars Range'].value_counts().sort_index()

# Calculate percentages
total_count = grouped_data.sum()
percentages = (grouped_data / total_count) * 100

# Plot the data
plt.figure(figsize=(8, 6))
ax = grouped_data.plot(kind='bar', color='lightblue', edgecolor='black', width=0.8)  # Adjust the width parameter

# Annotate the bars with percentage values
for idx, (count, percentage) in enumerate(zip(grouped_data, percentages)):
    ax.text(idx, count, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

plt.xlabel('Ranges of Star Count', fontsize=12)
plt.ylabel('Number of Maven Packages', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
