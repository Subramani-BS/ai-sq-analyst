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

    df = clean_column_names(df)
    report.append("✅ Column names cleaned and standardized")

    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        report.append(f"🗑️ Removed {dupes} duplicate rows")
    else:
        report.append("✅ No duplicate rows found")

    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        df = df.dropna(how='all')
        report.append(f"🗑️ Removed {empty_rows} completely empty rows")

    empty_cols = df.isnull().all(axis=0).sum()
    if empty_cols > 0:
        df = df.dropna(axis=1, how='all')
        report.append(f"🗑️ Removed {empty_cols} completely empty columns")

    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                filled = df[col].isnull().sum()
                df[col] = df[col].fillna(median_val)
                if filled > 0:
                    report.append(
                        f"🔢 '{col}': filled {filled} nulls with median ({median_val:.2f})")
            else:
                filled = df[col].isnull().sum()
                df[col] = df[col].fillna("Unknown")
                if filled > 0:
                    report.append(
                        f"📝 '{col}': filled {filled} nulls with 'Unknown'")
    else:
        report.append("✅ No missing values found")

    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    if len(str_cols) > 0:
        report.append(
            f"✂️ Stripped whitespace from {len(str_cols)} text columns")

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                report.append(f"🔄 '{col}': converted to numeric")
                continue
            except Exception:
                pass
            try:
                df[col] = pd.to_datetime(
                    df[col], infer_datetime_format=True)
                report.append(f"📅 '{col}': converted to datetime")
                continue
            except Exception:
                pass

    unnamed = [col for col in df.columns if col.startswith('unnamed')]
    if unnamed:
        df = df.drop(columns=unnamed)
        report.append(f"🗑️ Removed {len(unnamed)} unnamed columns")

    report.append(
        f"📊 Original: {original_rows} rows x {original_cols} cols")
    report.append(
        f"📊 Cleaned:  {len(df)} rows x {len(df.columns)} cols")

    return df, report


def read_file(uploaded_file):
    filename = uploaded_file.name.lower()

    # ── CSV ───────────────────────────────────────────────
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        return df, None

    # ── Excel .xlsx ───────────────────────────────────────
    elif filename.endswith('.xlsx'):
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        if len(sheet_names) == 1:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
            return df, None
        else:
            return None, sheet_names

    # ── Excel .xls ────────────────────────────────────────
    elif filename.endswith('.xls'):
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        if len(sheet_names) == 1:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
            return df, None
        else:
            return None, sheet_names

    # ── Excel .xlsm ───────────────────────────────────────
    elif filename.endswith('.xlsm'):
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        if len(sheet_names) == 1:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
            return df, None
        else:
            return None, sheet_names

    else:
        raise ValueError(f"Unsupported file type: {filename}")


def load_file_to_sqlite(uploaded_file, table_name='data',
                        auto_clean=False, sheet_name=None):
    filename = uploaded_file.name.lower()

    # Read file
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
    else:
        df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name if sheet_name else 0
        )

    cleaning_report = []

    if auto_clean:
        df, cleaning_report = clean_data(df)
    else:
        df = clean_column_names(df)
        cleaning_report.append(
            "ℹ️ Auto cleaning OFF — only column names standardized")
        cleaning_report.append(
            f"📊 Dataset: {len(df)} rows x {len(df.columns)} cols")
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            cleaning_report.append(
                f"⚠️ Found {null_count} null values — enable auto clean to fix")

    engine = create_engine("sqlite:///analyst.db", echo=False)
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    columns_info = {col: str(df[col].dtype) for col in df.columns}
    return engine, table_name, df, columns_info, cleaning_report


# Keep backward compatibility
def load_csv_to_sqlite(csv_file, table_name='data', auto_clean=False):
    return load_file_to_sqlite(
        csv_file, table_name=table_name, auto_clean=auto_clean)