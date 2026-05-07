from ydata_profiling import ProfileReport
from users.kczap.profiling_data.load_honey_dataset import load_honey_data

# Load data
df = load_honey_data()

print("\nDataset information:")
df.info()

# Categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nNumber of categorical columns: {len(categorical_cols)}")
print("Categorical columns:", categorical_cols)

# Numerical columns
numerical_cols = df.select_dtypes(exclude='object').columns.tolist()
print(f"\nNumber of categorical columns: {len(numerical_cols)}")
print("Categorical columns:", numerical_cols)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Unique values
print("\nNumber of unique values per column:")
print(df.nunique())

#
df_numeric = df.select_dtypes(include=['number'])

# Generate profiling report
report = ProfileReport(
    df_numeric,
    title='Profiling Report — Honey Purity',
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "kendall": {"calculate": True}
    }
)
# Save report to HTML file
report.to_file("my_report_honey.html")
print("\nReport saved as: my_report_honey.html")