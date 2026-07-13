import pandas as pd
from sqlalchemy import create_engine

# Load the existing cleaned CSV
df = pd.read_csv("data/processed/students_cleaned.csv")

# Add a simple StudentID back if it's not there (needed as a primary key for editing)
if 'StudentID' not in df.columns:
    df.insert(0, 'StudentID', range(1001, 1001 + len(df)))

# Create SQLite database
engine = create_engine("sqlite:///data/students.db")
df.to_sql("students", engine, if_exists="replace", index=False)

print(f"Database created with {len(df)} students.")