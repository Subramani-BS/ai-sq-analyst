import pandas as pd
from sqlalchemy import create_engine
import re
from deep_translator import GoogleTranslator


def translate_text(text, target="en"):
    """Translates single short string if needed[cite: 3]."""
    if not isinstance(text, str) or not text.strip() or text.strip().isdigit():
        return text
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


def translate_dataframe(df):
    """
    Fast in-place translation for foreign languages (e.g. Hindi).
    Translates unique values in batches of 50 to avoid question mark encoding issues
    and speed up execution by 10-20x[cite: 3].
    """
    report = []
    translator = GoogleTranslator(source="auto", target="en")

    # 1. Translate column names in-place
    translated_cols = {}
    for col in df.columns:
        col_str = str(col).strip()
        if not col_str.isascii():
            try:
                translated_cols[col] = translator.translate(col_str)
            except Exception:
                pass

    if translated_cols:
        df = df.rename(columns=translated_cols)
        report.append(f"🌐 Translated {len(translated_cols)} non-English column header(s)")

    # 2. Translate text cell values in batches
    str_cols = df.select_dtypes(include="object").columns
    total_translated = 0

    for col in str_cols:
        raw_vals = [
            str(v).strip() for v in df[col].dropna().unique() 
            if isinstance(v, str) and not v.isascii() and v.strip()
        ]
        
        if not raw_vals:
            continue

        trans_map = {}
        batch_size = 50

        for i in range(0, len(raw_vals), batch_size):
            chunk = raw_vals[i:i + batch_size]
            try:
                translated_chunk = translator.translate_batch(chunk)
                for orig, trans in zip(chunk, translated_chunk):
                    if trans:
                        trans_map[orig] = trans
            except Exception:
                for item in chunk:
                    try:
                        trans_map[item] = translator.translate(item)
                    except Exception:
                        pass

        if trans_map:
            df[col] = df[col].astype(str).replace(trans_map)
            total_translated += len(trans_map)

    if total_translated > 0:
        report.append(f"🌐 Translated {total_translated} unique non-English value(s) in-place")
    else:
        report.append("🌐 No foreign language values detected")

    return df, report


def handle_large_integers(df):
    """
    Fixes OverflowError: Python int too large to convert to SQLite INTEGER.
    SQLite only supports signed 64-bit ints (-9223372036854775808 to 9223372036854775807)[cite: 3].
    Converts numbers exceeding this boundary to strings (TEXT)[cite: 3].
    """
    sqlite_max_int = 9223372036854775807
    sqlite_min_int = -9223372036854775808

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                if (df[col] > sqlite_max_int).any() or (df[col] < sqlite_min_int).any():
                    df[col] = df[col].astype(str)
            except Exception:
                df[col] = df[col].astype(str)
        elif df[col].dtype == 'object':
            def is_overflow_int(val):
                if isinstance(val, int):
                    return val > sqlite_max_int or val < sqlite_min_int
                return False
            if df[col].apply(is_overflow_int).any():
                df[col] = df[col].astype(str)
    return df


def clean_column_names(df):
    """
    Standardizes column names and deduplicates identical or empty names.
    Supports Unicode/Hindi/Devanagari characters without wiping them to col_0, col_1!
    """
    new_cols = []
    seen = {}
    
    for i, col in enumerate(df.columns):
        col_str = str(col).strip()
        # Preserve word characters (including Hindi \w) and remove invalid symbols
        cleaned = re.sub(r'[^\w\s]', '', col_str, flags=re.UNICODE)
        # Replace spaces with underscores
        cleaned = re.sub(r'\s+', '_', cleaned).strip('_')
        
        # Only fallback if header is genuinely empty
        if not cleaned:
            cleaned = f"col_{i}"
        
        if cleaned in seen:
            seen[cleaned] += 1
            cleaned = f"{cleaned}_{seen[cleaned]}"
        else:
            seen[cleaned] = 0
            
        new_cols.append(cleaned)
        
    df.columns = new_cols
    return df


def clean_data(df):
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
                    report.append(f"🔢 '{col}': filled {filled} nulls with median ({median_val:.2f})")
            else:
                filled = df[col].isnull().sum()
                df[col] = df[col].fillna("Unknown")
                if filled > 0:
                    report.append(f"📝 '{col}': filled {filled} nulls with 'Unknown'")
    else:
        report.append("✅ No missing values found")

    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    if len(str_cols) > 0:
        report.append(f"✂️ Stripped whitespace from {len(str_cols)} text columns")

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                converted = pd.to_numeric(df[col])
                if (converted > 9223372036854775807).any() or (converted < -9223372036854775808).any():
                    pass
                else:
                    df[col] = converted
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
    filename = uploaded_file.name.lower()

    if filename.endswith('.csv'):
        for enc in ['utf-8-sig', 'utf-8', 'utf-16', 'cp1252']:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                if "???" not in df.head(5).to_string():
                    return df, None
            except Exception:
                continue
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
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

    if filename.endswith('.csv'):
        df = None
        for enc in ['utf-8-sig', 'utf-8', 'utf-16', 'cp1252']:
            try:
                uploaded_file.seek(0)
                temp_df = pd.read_csv(uploaded_file, encoding=enc)
                if "???" not in temp_df.head(5).to_string():
                    df = temp_df
                    break
            except Exception:
                continue
        if df is None:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
    else:
        df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name if sheet_name else 0
        )

    cleaning_report = []

    # 1. Translate in-place
    if translate_to_english:
        df, trans_report = translate_dataframe(df)
        cleaning_report.extend(trans_report)

    # 2. Clean and deduplicate headers (retaining Hindi/Unicode names)
    df = clean_column_names(df)

    # 3. Clean contents
    if auto_clean:
        df, clean_rep = clean_data(df)
        cleaning_report.extend(clean_rep)
    else:
        cleaning_report.append("ℹ️ Auto cleaning OFF — only column names standardized")
        cleaning_report.append(f"📊 Dataset: {len(df)} rows x {len(df.columns)} cols")
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            cleaning_report.append(f"⚠️ Found {null_count} null values — enable auto clean to fix")

    # 4. Handle large integers before storing in SQLite
    df = handle_large_integers(df)

    engine = create_engine("sqlite:///analyst.db", echo=False)
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    columns_info = {col: str(df[col].dtype) for col in df.columns}
    return engine, table_name, df, columns_info, cleaning_report


def load_csv_to_sqlite(csv_file, table_name='data', auto_clean=False,
                       translate_to_english=False):
    return load_file_to_sqlite(
        csv_file,
        table_name=table_name,
        auto_clean=auto_clean,
        translate_to_english=translate_to_english
    )