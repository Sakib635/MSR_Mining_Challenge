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


def unique_word_count(text):
    text_data = text

    words = text_data.lower().split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    # print(filtered_words)
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
open_issue_percentage = []
closed_issues_percentage = []
popularity_1_year = []
release_frequency = []
release_count = []
vulnerabilities = []

count1 = 0

# Define file paths
resolved_file_path = "combined_metrics.csv"

# Read the CSV file
df_resolved = pd.read_csv(resolved_file_path, low_memory=False)


# Append data to lists
for index, row in df_resolved.iterrows():
    count1 += 1
    # print(count1)
    current_date_obj = datetime.strptime('2024-12-06', '%Y-%m-%d')
    created_at_obj = datetime.strptime(row['created_at'], '%Y-%m-%d')
    date_diff_years = abs((current_date_obj - created_at_obj).days / 365)
    if date_diff_years >= 2:
        star_count.append(row.get('stars', 0))
        fork_count.append(row.get('forks', 0))
        pull_requests.append(row.get('pull_requests', 0))
        subscribers.append(row.get('subscribers', 0))
        if 'No' in row.get('license', 'No'):
            license_info.append(0)
        else:
            license_info.append(1)
        tags_count.append(row.get('Tags Count', 0))
        open_issues_count.append(row.get('Open Issues Count', 0))
        closed_issues_count.append(row.get('Closed Issues Count', 0))
        contributors_count.append(row.get('Contributors Count', 0))
        commits_count.append(row.get('Commits Count', 0))
        readme_exists.append(1 if row.get('README Exists', False) else 0)
        text = str(row.get('About Info', ''))

        about_info.append(unique_word_count(text))
        upstream_dependency_count.append(row.get('Dependencies', 0))
        downstream_dependency_count.append(row.get('Usage', 0))
        total_count = (row['Closed Issues Count'] + row['Open Issues Count']) if (row['Closed Issues Count'] + row['Open Issues Count']) != 0 else 1
        closed_issues_percentage.append(row['Closed Issues Count'] / total_count)
        
        # Additional metrics
        popularity_1_year.append(row.get('popularity_1_year', 0))
        release_frequency.append(row.get('ReleaseFrequency', 0))
        release_count.append(row.get('Release Count', 0))
        vulnerabilities.append(row.get('Vulnerabilities', 0))

# Create a DataFrame
data = pd.DataFrame({
    'star_count': star_count,
    # 'fork_count': fork_count,
    # 'pull_requests': pull_requests,
    # 'subscribers': subscribers,
    'license': license_info,
    # 'tags_count': tags_count,
    # 'open_issues_count': open_issues_count,
    # 'closed_issues_count': closed_issues_count,
    # 'contributors_count': contributors_count,
    'commits_count': commits_count,
    'readme_exists': readme_exists,
    'about_info': about_info,
    'Dependencies': upstream_dependency_count,
    'Usage': downstream_dependency_count,
    'closed_issues_percentage': closed_issues_percentage,
    # 'popularity_1_year': popularity_1_year,
    'release_frequency': release_frequency,
    # 'release_count': release_count,
    'vulnerabilities': vulnerabilities
})

# Feature selection (including new metrics)
features = [
    # 'star_count',
    # 'fork_count',
    # 'pull_requests',
    # 'subscribers',
    'license',
    # 'tags_count',
    # 'open_issues_count',
    # 'closed_issues_count',
    # 'contributors_count',
    'commits_count',
    'readme_exists',
    'about_info',
    'Dependencies',
    'Usage',
    'closed_issues_percentage',
    # 'popularity_1_year',
    'release_frequency',
    # 'release_count',
    'vulnerabilities'
]

# Sort by star_count in descending order
data_sorted = data.sort_values(by='star_count', ascending=False)


# Calculate the number of rows to label
n_rows = len(data_sorted)
# print(n_rows)
top_20_percent = int(n_rows * 0.2)
bottom_20_percent = int(n_rows * 0.2)

# Label the top 20% as 1 and bottom 20% as 0
data_sorted['final_label'] = np.nan  # Initialize with NaN
data_sorted.iloc[:top_20_percent, data_sorted.columns.get_loc('final_label')] = 1
data_sorted.iloc[-bottom_20_percent:, data_sorted.columns.get_loc('final_label')] = 0

# Drop the middle 60%
data_labeled = data_sorted.dropna(subset=['final_label'])

X = data_labeled.drop(columns=['final_label', 'star_count'])
y = data_labeled['final_label']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = sm.Logit(y, sm.add_constant(X))
result = model.fit()

# Print summary of the logistic regression model
print(result.summary())

# Extracting specific values for significant predictors
significant_predictors = result.pvalues[result.pvalues < 0.05].index
for predictor in significant_predictors:
    coef = result.params[predictor]
    std_err = result.bse[predictor]
    z_value = result.tvalues[predictor]
    p_value = result.pvalues[predictor]
    print(f'Predictor: {predictor}')
    print(f'  Coefficient: {coef}')
    print(f'  Standard Error: {std_err}')
    print(f'  z-value: {z_value}')
    print(f'  p-value: {p_value}')
    print()

# Calculate AUC
# Predict probabilities
y_probs = result.predict(sm.add_constant(X))
# Compute AUC score
auc = roc_auc_score(y, y_probs)
print(f'AUC: {auc}')