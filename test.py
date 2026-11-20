
import numpy as np
import pandas as pd

# Load the messy CSV dataset
df = pd.read_csv('electricity_bill_prediction_dataset(1).csv.csv')

print("Original Dataset:")
print(df.head(10))


# 1. Remove duplicate Record IDs
df = df.drop_duplicates(subset=['Record_ID'], keep='first')


# 2. Clean Record_ID column
df['Record_ID'] = df['Record_ID'].astype(str).str.strip()


# 3. Clean Month column
df['Month'] = df['Month'].astype(str).str.strip().str.title()


# 4. Clean numeric columns
numeric_columns = [
    'Units_Consumed_kWh',
    'Rate_per_Unit_Rs',
    'Fixed_Charge_Rs',
    'Electricity_Bill_Rs'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# 5. Handle negative and invalid values
df.loc[df['Units_Consumed_kWh'] < 0, 'Units_Consumed_kWh'] = np.nan

df.loc[df['Rate_per_Unit_Rs'] < 0, 'Rate_per_Unit_Rs'] = np.nan

df.loc[df['Fixed_Charge_Rs'] < 0, 'Fixed_Charge_Rs'] = np.nan

df.loc[df['Electricity_Bill_Rs'] < 0, 'Electricity_Bill_Rs'] = np.nan


# 6. Handle missing values using median
for col in numeric_columns:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)


# 7. Check electricity bill calculation
expected_bill = (
    df['Units_Consumed_kWh'] * df['Rate_per_Unit_Rs']
    + df['Fixed_Charge_Rs']
)

df['Bill_Check'] = np.where(
    np.isclose(df['Electricity_Bill_Rs'], expected_bill, atol=0.01),
    'Correct',
    'Incorrect'
)

print("\nBill Calculation Check:")
print(df['Bill_Check'].value_counts())


# 8. Remove temporary Bill_Check column
df = df.drop(columns=['Bill_Check'])


# 9. Export cleaned dataset
df.to_csv('Data_Cleaning.csv', index=False)


# 10. Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Record IDs:")
print(df['Record_ID'].duplicated().sum())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nData Cleaning Completed Successfully!")
print("File Saved: Data_Cleaning.csv")
