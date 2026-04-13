import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('Test-O_G_Equipment_Data.csv')
df['Fail'] = df['Fail'].fillna(0).astype('int')
vars_to_analyze = ['Temperature', 'Pressure', 'VibrationX', 'VibrationY', 'VibrationZ', 'Frequency']

# Basic descriptive statistics by Fail
print("=== Descriptive Statistics by Fail ===")
print(df.groupby('Fail')[vars_to_analyze].mean())

# Identify temporal patterns:
# Are there specific presets where the variables increase?
print("\n=== Mean values by Preset_1, Preset_2 and Fail ===")
print(df.groupby(['Preset_1', 'Preset_2', 'Fail'])[vars_to_analyze].mean().reset_index().to_string())

# Let's check for "trend before failure"
# We can find the indices where Fail == 1
fail_indices = df.index[df['Fail'] == 1].tolist()
print("\n=== Anomalies before Failure ===")
# Average value of sensors 1 to 5 periods before failure, compared to normal periods
before_fail = []
for idx in fail_indices:
    # get up to 3 previous cycles
    start = max(0, idx-3)
    if start < idx:
        window = df.iloc[start:idx]
        for _, row in window.iterrows():
            before_fail.append(row[vars_to_analyze].to_dict())

if before_fail:
    df_before_fail = pd.DataFrame(before_fail)
    print("Metrics average 1-3 cycles BEFORE failure:")
    print(df_before_fail.mean())
    print("Overall metrics for normal state (excluding close to failure):")
    normal_indices = set(df.index[df['Fail'] == 0])
    # remove indices close to failure
    for idx in fail_indices:
        for offset in range(1, 4):
            if (idx - offset) in normal_indices:
                normal_indices.remove(idx - offset)
    print(df.iloc[list(normal_indices)][vars_to_analyze].mean())

# Check correlations
print("\n=== Correlation with Fail ===")
print(df[vars_to_analyze + ['Fail']].corr()['Fail'])

