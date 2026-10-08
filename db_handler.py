import pandas as pd
from sqlalchemy import create_engine
import re
from deep_translator import GoogleTranslator


def translate_text(text, target="en"):
    """Translates text if it is a non-empty string."""
    if not isinstance(text, str) or not text.strip() or text.strip().isdigit():
        return text
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


def translate_dataframe(df):
    """Translates column names and non-English text cell values to English."""
    report = []
    
    # 1. Translate column headers
    translated_cols = {}
    for col in df.columns:
        trans_col = translate_text(str(col))
        if trans_col != col:
            translated_cols[col] = trans_col
    
    if translated_cols:
        df = df.rename(columns=translated_cols)
        report.append(f"🌐 Translated {len(translated_cols)} column name(s) to English")

    # 2. Translate string/text cell values
    str_cols = df.select_dtypes(include="object").columns
    total_translated_cells = 0
    
    for col in str_cols:
        unique_vals = [v for v in df[col].dropna().unique() if isinstance(v, str) and v.strip()]
        trans_map = {}
        for val in unique_vals:
            if not val.isascii():
                trans_map[val] = translate_text(val)
        
        if trans_map:
            df[col] = df[col].replace(trans_map)
            total_translated_cells += len(trans_map)
            
    if total_translated_cells > 0:
        report.append(f"🌐 Translated non-English values across {len(str_cols)} text column(s)")
    else:
        report.append("🌐 No non-English cell values detected")

    return df, report


def clean_column_names(df):
    """Standardizes column names and deduplicates identical or empty names."""
    new_cols = []
    seen = {}
    
    for i, col in enumerate(df.columns):
        col_str = str(col).strip()
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', col_str).strip('_').lower()
        if not cleaned:
            cleaned = f"col_{i}"
        
        # Deduplicate repeated names (e.g., date, date_1, date_2)
        if cleaned in seen:
            seen[cleaned] += 1
            cleaned = f"{cleaned}_{seen[cleaned]}"
        else:
            seen[cleaned] = 0
            
        new_cols.append(cleaned)
        
    df.columns = new_cols
    return df


def clean_data(df):
    """Applies cleaning operations: duplicates, missing values, typing."""
    report = []
    original_rows = len(df)
    original_cols = len(df.columns)

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
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                report.append(f"📅 '{col}': converted to datetime")
                continue
            except Exception:
                pass

    unnamed = [col for col in df.columns if col.startswith('unnamed')]
    if unnamed:
        df = df.drop(columns=unnamed)
        report.append(f"🗑️ Removed {len(unnamed)} unnamed columns")

    report.append(f"📊 Original: {original_rows} rows x {original_cols} cols")
    report.append(f"📊 Cleaned:  {len(df)} rows x {len(df.columns)} cols")

    return df, report


def read_file(uploaded_file):
    """Inspects file type and returns (DataFrame, sheet_names_list)."""
    filename = uploaded_file.name.lower()

    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        return df, None

    elif filename.endswith(('.xlsx', '.xls', '.xlsm')):
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
                        auto_clean=False, sheet_name=None,
                        translate_to_english=False):
    filename = uploaded_file.name.lower()

    # 1. Read file
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

    # 2. Optional translation
    if translate_to_english:
        df, trans_report = translate_dataframe(df)
        cleaning_report.extend(trans_report)

    # 3. Clean and deduplicate column names
    df = clean_column_names(df)

    # 4. Data cleaning
    if auto_clean:
        df, clean_rep = clean_data(df)
        cleaning_report.extend(clean_rep)
    else:
        cleaning_report.append(
            "ℹ️ Auto cleaning OFF — only column names standardized")
        cleaning_report.append(
            f"📊 Dataset: {len(df)} rows x {len(df.columns)} cols")
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            cleaning_report.append(
                f"⚠️ Found {null_count} null values — enable auto clean to fix")

    # 5. Persist to SQLite
    engine = create_engine("sqlite:///analyst.db", echo=False)
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    columns_info = {col: str(df[col].dtype) for col in df.columns}
    return engine, table_name, df, columns_info, cleaning_report


def load_csv_to_sqlite(csv_file, table_name='data', auto_clean=False,
                       translate_to_english=False):
    """Backwards compatibility wrapper for load_csv_to_sqlite."""
    return load_file_to_sqlite(
        csv_file,
        table_name=table_name,
        auto_clean=auto_clean,
        translate_to_english=translate_to_english
    )