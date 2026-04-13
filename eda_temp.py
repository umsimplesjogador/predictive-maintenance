import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Test-O_G_Equipment_Data.csv')

# Preprocessing
print("Data Types before preprocessing:")
print(df.dtypes)
print("\nNull values:")
print(df.isnull().sum())

# Convert Fail to boolean/int (0 and 1)
df['Fail'] = df['Fail'].fillna(0).astype(int)

print("\nData Types after preprocessing:")
print(df.dtypes)

# Task 1: Calculate and plot the number of times the equipment failed
fail_counts = df['Fail'].value_counts()
print("\nFail Counts:")
print(fail_counts)

# Task 2: Categorize and create visualizations showing failure distribution by Preset_1 and Preset_2
# Distribution of failures by Preset_1
preset1_fail = df.groupby('Preset_1')['Fail'].sum()
print("\nFailures by Preset_1:")
print(preset1_fail)

# Distribution of failures by Preset_2
preset2_fail = df.groupby('Preset_2')['Fail'].sum()
print("\nFailures by Preset_2:")
print(preset2_fail)

# Combination of Preset_1 and Preset_2
comb_fail = df.groupby(['Preset_1', 'Preset_2'])['Fail'].agg(['count', 'sum'])
comb_fail['fail_rate'] = comb_fail['sum'] / comb_fail['count']
comb_fail = comb_fail.sort_values(by='fail_rate', ascending=False)
print("\nFailures by Combination of Preset_1 and Preset_2:")
print(comb_fail)
