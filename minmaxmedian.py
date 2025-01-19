import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datetime import datetime
# Data preparation
def unique_word_count(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    count = 0
    unique_word = {}
    for word in filtered_words:
        if word not in unique_word:
            unique_word[word] = 1
            count += 1
    return count

# Initialize lists to store data
star_count = []
fork_count = []
pull_requests = []
subscribers = []
license_info = []
tags_count = []
open_issues_count = []
closed_issues_count = []
contributors_count = []
commits_count = []
readme_exists = []
about_info = []
upstream_dependency_count = []
downstream_dependency_count = []
closed_issues_percentage = []
popularity_1_year = []
release_frequency = []
release_count = []
vulnerabilities = []

resolved_file_path = "combined_metrics.csv"

# Read CSV file
df_resolved = pd.read_csv(resolved_file_path, low_memory=False)

# Parse and append data
for index, row in df_resolved.iterrows():
    current_date_obj = datetime.strptime('2024-12-06', '%Y-%m-%d')
    created_at_obj = datetime.strptime(row['created_at'], '%Y-%m-%d')
    date_diff_years = abs((current_date_obj - created_at_obj).days / 365)
    if date_diff_years >= 2:
        star_count.append(row.get('stars', 0))
        fork_count.append(row.get('forks', 0))
        pull_requests.append(row.get('pull_requests', 0))
        subscribers.append(row.get('subscribers', 0))
        license_info.append(0 if 'No' in row.get('license', 'No') else 1)
        tags_count.append(row.get('Tags Count', 0))
        open_issues_count.append(row.get('Open Issues Count', 0))
        closed_issues_count.append(row.get('Closed Issues Count', 0))
        contributors_count.append(row.get('Contributors Count', 0))
        commits_count.append(row.get('Commits Count', 0))
        readme_exists.append(1 if row.get('README Exists', False) else 0)
        about_info.append(unique_word_count(str(row.get('About Info', ''))))
        upstream_dependency_count.append(row.get('Dependencies', 0))
        downstream_dependency_count.append(row.get('Usage', 0))
        total_count = (row['Closed Issues Count'] + row['Open Issues Count']) or 1
        closed_issues_percentage.append(row['Closed Issues Count'] / total_count)
        popularity_1_year.append(row.get('popularity_1_year', 0))
        release_frequency.append(row.get('ReleaseFrequency', 0))
        release_count.append(row.get('Release Count', 0))
        vulnerabilities.append(row.get('Vulnerabilities', 0))

# Create DataFrame
data = pd.DataFrame({
    'star_count': star_count,
    'fork_count': fork_count,
    'pull_requests': pull_requests,
    'subscribers': subscribers,
    'license': license_info,
    'tags_count': tags_count,
    'open_issues_count': open_issues_count,
    'closed_issues_count': closed_issues_count,
    'contributors_count': contributors_count,
    'commits_count': commits_count,
    'readme_exists': readme_exists,
    'about_info': about_info,
    'Dependencies': upstream_dependency_count,
    'Usage': downstream_dependency_count,
    'closed_issues_percentage': closed_issues_percentage,
    'popularity_1_year': popularity_1_year,
    'release_frequency': release_frequency,
    'release_count': release_count,
    'vulnerabilities': vulnerabilities
})

# Feature selection
features = data.columns.tolist()

# Sort data by star_count
data_sorted = data.sort_values(by='star_count', ascending=False)
# Calculate the number of rows to label
n_rows = len(data_sorted)
print(n_rows)
top_20_percent = int(n_rows * 0.2)
bottom_20_percent = int(n_rows * 0.2)

# Label the top 20% as 1 and bottom 20% as 0
data_sorted['final_label'] = np.nan  # Initialize with NaN
data_sorted.iloc[:top_20_percent, data_sorted.columns.get_loc('final_label')] = 1
data_sorted.iloc[-bottom_20_percent:, data_sorted.columns.get_loc('final_label')] = 0

# Split data into top 50% and bottom 50%
midpoint = len(data_sorted) // 2
top_50 = data_sorted.iloc[:midpoint]
bottom_50 = data_sorted.iloc[midpoint:]

# Analysis: Descriptive statistics and statistical tests
results = {}
for feature in features:
    # Descriptive statistics
    top_stats = {
        "min": top_50[feature].min(),
        "max": top_50[feature].max(),
        "mean": top_50[feature].mean(),
        "median": top_50[feature].median(),
    }
    bottom_stats = {
        "min": bottom_50[feature].min(),
        "max": bottom_50[feature].max(),
        "mean": bottom_50[feature].mean(),
        "median": bottom_50[feature].median(),
    }

    # Mann-Whitney U Test
    u_stat, p_value = mannwhitneyu(top_50[feature], bottom_50[feature], alternative='two-sided')

    # Cohen's d
    top_mean = np.mean(top_50[feature])
    bottom_mean = np.mean(bottom_50[feature])
    top_std = np.std(top_50[feature], ddof=1)
    bottom_std = np.std(bottom_50[feature], ddof=1)
    pooled_std = np.sqrt(((len(top_50) - 1) * top_std**2 + (len(bottom_50) - 1) * bottom_std**2) / (len(top_50) + len(bottom_50) - 2))
    cohen_d = (top_mean - bottom_mean) / pooled_std

    # Store results
    results[feature] = {
        "top_stats": top_stats,
        "bottom_stats": bottom_stats,
        "mannwhitneyu": {"u_stat": u_stat, "p_value": p_value},
        "cohen_d": cohen_d,
    }

# Display results
for feature, stats in results.items():
    print(f"Feature: {feature}")
    print("Top 50% Stats:", stats["top_stats"])
    print("Bottom 50% Stats:", stats["bottom_stats"])
    print("Mann-Whitney U Test:", stats["mannwhitneyu"])
    print(f"Cohen's d: {stats['cohen_d']:.4f}")
    print("-" * 50)
