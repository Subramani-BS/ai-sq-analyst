import pandas as pd
from sqlalchemy import create_engine
import re


def clean_column_names(df):
    df.columns = [
        re.sub(r'[^a-zA-Z0-9_]', '_', col).strip('_').lower()
        for col in df.columns
    ]
    return df


def clean_data(df):
    report = []
    original_rows = len(df)
    original_cols = len(df.columns)

    # 1. Clean column names
    df = clean_column_names(df)
    report.append("✅ Column names cleaned and standardized")

    # 2. Remove duplicate rows
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        report.append(f"🗑️ Removed {dupes} duplicate rows")
    else:
        report.append("✅ No duplicate rows found")

    # 3. Remove completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        df = df.dropna(how='all')
        report.append(f"🗑️ Removed {empty_rows} completely empty rows")

    # 4. Remove completely empty columns
    empty_cols = df.isnull().all(axis=0).sum()
    if empty_cols > 0:
        df = df.dropna(axis=1, how='all')
        report.append(f"🗑️ Removed {empty_cols} completely empty columns")

    # 5. Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                filled = df[col].isnull().sum()
                df[col] = df[col].fillna(median_val)
                if filled > 0:
                    report.append(f"🔢 '{col}': filled {filled} missing with median ({median_val:.2f})")
            else:
                filled = df[col].isnull().sum()
                df[col] = df[col].fillna("Unknown")
                if filled > 0:
                    report.append(f"📝 '{col}': filled {filled} missing with 'Unknown'")
    else:
        report.append("✅ No missing values found")

    # 6. Strip whitespace from string columns
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    if len(str_cols) > 0:
        report.append(f"✂️ Stripped whitespace from {len(str_cols)} text columns")

    # 7. Fix data types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                report.append(f"🔄 '{col}': converted to numeric")
                continue
            except:
                pass
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True)
                df[col] = converted
                report.append(f"📅 '{col}': converted to datetime")
                continue
            except:
                pass

    # 8. Remove unnamed columns
    unnamed = [col for col in df.columns if col.startswith('unnamed')]
    if unnamed:
        df = df.drop(columns=unnamed)
        report.append(f"🗑️ Removed {len(unnamed)} unnamed columns")

    # 9. Summary
    report.append(f"📊 Original: {original_rows} rows × {original_cols} cols")
    report.append(f"📊 Cleaned:  {len(df)} rows × {len(df.columns)} cols")

    return df, report


def load_csv_to_sqlite(csv_file, table_name='data'):
    df = pd.read_csv(csv_file)
    df, cleaning_report = clean_data(df)
    engine = create_engine("sqlite:///analyst.db", echo=False)
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    columns_info = {col: str(df[col].dtype) for col in df.columns}
    return engine, table_name, df, columns_info, cleaning_report