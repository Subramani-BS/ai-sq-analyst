import os
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from deep_translator import GoogleTranslator
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import reverse_geocoder as rg

# Warmup reverse_geocoder KD-tree into RAM once on startup to eliminate freezing
try:
    _ = rg.search([(0.0, 0.0)], mode=1)
except Exception:
    pass


def translate_text(text, target="en"):
    """Translates single short string if needed."""
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
    and speed up execution by 10-20x.
    """
    report = []
    translator = GoogleTranslator(source="auto", target="en")

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
    Converts numbers exceeding 64-bit boundaries to strings (TEXT).
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
    Preserves Unicode/Hindi characters.
    """
    new_cols = []
    seen = {}
    
    for i, col in enumerate(df.columns):
        col_str = str(col).strip()
        cleaned = re.sub(r'[^\w\s]', '', col_str, flags=re.UNICODE)
        cleaned = re.sub(r'\s+', '_', cleaned).strip('_')
        
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
    """
    Cleans duplicates, empty rows/columns, trims strings, and cleans types.
    Preserves raw null values intact without filling.
    """
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

    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        report.append(f"ℹ️ Preserved {missing_count} null value(s) without filling")
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
    """Reads CSV/Excel files preserving Unicode/Hindi encodings."""
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

    if translate_to_english:
        df, trans_report = translate_dataframe(df)
        cleaning_report.extend(trans_report)

    df = clean_column_names(df)

    if auto_clean:
        df, clean_rep = clean_data(df)
        cleaning_report.extend(clean_rep)
    else:
        cleaning_report.append("ℹ️ Auto cleaning OFF — only column names standardized")
        cleaning_report.append(f"📊 Dataset: {len(df)} rows x {len(df.columns)} cols")
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            cleaning_report.append(f"ℹ️ Preserved {null_count} null values")

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


# ════════════════════════════════════════════════════════════
# GEOCODING & REVERSE GEOCODING HELPERS
# ════════════════════════════════════════════════════════════
def get_geolocator():
    """Initializes Nominatim geocoder with a custom user-agent."""
    return Nominatim(user_agent="ai_sql_analyst_geo_app_v4", timeout=12)


def clean_location_token(text):
    """
    Cleans government census codes, bracketed numbers, and unwanted symbols.
    e.g. 'Veerta [50]' -> 'Veerta'
         'Ward No. 4 (Urban)' -> 'Ward No. 4'
    """
    if not isinstance(text, str):
        return ""
    # 1. Remove text inside brackets/parentheses like [50], (102), etc.
    cleaned = re.sub(r'\[.*?\]', '', text)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    # 2. Remove stray symbols like slashes, underscores, quotes
    cleaned = re.sub(r'[\\/#_]', ' ', cleaned)
    # 3. Collapse multiple whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def forward_geocode_dataframe(df, address_cols, batch_delay=1.0):
    """
    Forward geocodes address strings with regex cleaning and cascading fallback.
    Prevents empty coordinates caused by census codes like 'Veerta [50]'.
    """
    if isinstance(address_cols, str):
        address_cols = [address_cols]

    geolocator = get_geolocator()
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=batch_delay, max_retries=2)

    # 1. Build cleaned address query per row
    def build_address_parts(row):
        parts = []
        for c in address_cols:
            val = str(row[c]).strip() if pd.notnull(row[c]) else ""
            if val and val.lower() not in ['nan', 'none', 'null', 'unknown', '']:
                clean_val = clean_location_token(val)
                if clean_val:
                    parts.append(clean_val)
        return parts

    address_parts_series = df.apply(build_address_parts, axis=1)

    primary_queries = address_parts_series.apply(lambda p: ", ".join(p) if p else "")
    unique_queries = [q for q in primary_queries.unique() if q]

    geo_cache = {}

    # 2. Geocode with automatic fallback
    for full_q in unique_queries:
        try:
            # Attempt 1: Exact cleaned string (e.g. Village, District, State)
            loc = geocode(f"{full_q}, India")
            if loc:
                geo_cache[full_q] = (loc.latitude, loc.longitude)
                continue

            # Attempt 2: Fallback — drop the micro-village and try (District, State)
            tokens = [t.strip() for t in full_q.split(',') if t.strip()]
            if len(tokens) > 1:
                broader_q = ", ".join(tokens[1:]) + ", India"
                loc = geocode(broader_q)
                if loc:
                    geo_cache[full_q] = (loc.latitude, loc.longitude)
                    continue

            geo_cache[full_q] = (None, None)
        except Exception:
            geo_cache[full_q] = (None, None)

    # 3. Map coordinates back to dataframe
    lat_list = []
    lon_list = []
    for q in primary_queries:
        coords = geo_cache.get(q, (None, None))
        lat_list.append(coords[0])
        lon_list.append(coords[1])

    df['latitude'] = lat_list
    df['longitude'] = lon_list

    successful_count = len([k for k, v in geo_cache.items() if v[0] is not None])
    return df, len(unique_queries), successful_count


def classify_settlement(addr_dict=None, raw_name="", cc=""):
    """
    Infers area type (Urban, Peri-Urban, Rural) based on OSM administrative tags or GeoNames entities.
    """
    if addr_dict:
        if any(addr_dict.get(k) for k in ['city', 'municipality', 'borough', 'commercial_area']):
            return "Urban"
        if any(addr_dict.get(k) for k in ['suburb', 'neighbourhood', 'town']):
            return "Peri-Urban" if addr_dict.get('suburb') and not addr_dict.get('city') else "Urban"
        if any(addr_dict.get(k) for k in ['census_town', 'outgrowth', 'industrial']):
            return "Peri-Urban"
        if any(addr_dict.get(k) for k in ['village', 'hamlet', 'farm', 'isolated_dwelling', 'locality']):
            return "Rural"
        return "Peri-Urban"

    raw_lower = raw_name.lower()
    if any(term in raw_lower for term in ['city', 'nagar', 'pur', 'cantt', 'metro']):
        return "Urban"
    elif any(term in raw_lower for term in ['rural', 'kalan', 'khurd', 'gaon', 'village']):
        return "Rural"
    elif any(term in raw_lower for term in ['outgrowth', 'estate', 'colony', 'sector']):
        return "Peri-Urban"
    
    return "Peri-Urban"


def reverse_geocode_dataframe(df, lat_col, lon_col, detailed_online=False, progress_callback=None, *args, **kwargs):
    """
    Reverse geocodes coordinates into separate hierarchical columns:
    [village_ward, city_suburb, district, state, country, area_type, geocoded_address]
    Supports a progress_callback(current, total) to report UI progress.
    """
    if 'detailed_online' in kwargs:
        detailed_online = kwargs['detailed_online']
    elif len(args) > 0:
        detailed_online = args[0]

    numeric_lat = pd.to_numeric(df[lat_col], errors='coerce')
    numeric_lon = pd.to_numeric(df[lon_col], errors='coerce')
    valid_mask = numeric_lat.notnull() & numeric_lon.notnull()

    columns_to_add = ['village_ward', 'city_suburb', 'district', 'state', 'country', 'area_type', 'geocoded_address']

    if not valid_mask.any():
        for col in columns_to_add:
            df[col] = None
        return df, 0, 0

    coord_series = list(zip(
        numeric_lat[valid_mask].round(5),
        numeric_lon[valid_mask].round(5)
    ))
    unique_coords = list(set(coord_series))

    lookup_map = {}

    if detailed_online:
        geolocator = get_geolocator()
        reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1.0, max_retries=2)
        total_unique = len(unique_coords)

        for i, pair in enumerate(unique_coords):
            if progress_callback:
                progress_callback(i + 1, total_unique)
            try:
                loc = reverse(pair, language="en", addressdetails=True, zoom=18)
                if loc and 'address' in loc.raw:
                    addr = loc.raw['address']

                    v_ward = (
                        addr.get('village') or 
                        addr.get('hamlet') or 
                        addr.get('ward') or 
                        addr.get('suburb') or 
                        addr.get('neighbourhood') or 
                        addr.get('quarter') or 
                        addr.get('residential') or 
                        addr.get('locality') or 
                        addr.get('isolated_dwelling') or 
                        ''
                    )

                    v_city = (
                        addr.get('city') or 
                        addr.get('town') or 
                        addr.get('municipality') or 
                        addr.get('subdistrict') or 
                        addr.get('city_district') or 
                        addr.get('tehsil') or 
                        ''
                    )

                    v_dist = (
                        addr.get('state_district') or 
                        addr.get('county') or 
                        addr.get('district') or 
                        ''
                    )
                    v_state = addr.get('state') or ''
                    v_country = addr.get('country') or ''

                    full_addr = loc.address
                    if not v_ward and full_addr:
                        tokens = [t.strip() for t in full_addr.split(',') if t.strip()]
                        if tokens:
                            v_ward = tokens[0]

                    if not v_city and v_dist:
                        v_city = v_dist

                    area_classification = classify_settlement(addr_dict=addr)
                    lookup_map[pair] = (v_ward, v_city, v_dist, v_state, v_country, area_classification, full_addr)
                else:
                    lookup_map[pair] = ('', '', '', '', '', 'Unknown', 'Not Found')
            except Exception:
                lookup_map[pair] = ('', '', '', '', '', 'Unknown', 'Not Found')
    else:
        results = rg.search(unique_coords, mode=1)
        for (lat, lon), r in zip(unique_coords, results):
            place_name = r.get('name', '')
            admin2 = r.get('admin2', '')
            admin1 = r.get('admin1', '')
            cc = r.get('cc', '')

            v_ward = place_name
            v_city = admin2 if admin2 else place_name
            v_dist = admin2 if admin2 else admin1
            v_state = admin1
            v_country = cc

            area_classification = classify_settlement(raw_name=place_name, cc=cc)
            full_addr = ", ".join([p for p in [place_name, admin2, admin1, cc] if p])
            lookup_map[(lat, lon)] = (v_ward, v_city, v_dist, v_state, v_country, area_classification, full_addr)

    for col in columns_to_add:
        df[col] = None

    valid_indices = df[valid_mask].index

    for idx, pair in zip(valid_indices, coord_series):
        w, c, d, s, co, atype, f = lookup_map.get(pair, (None, None, None, None, None, None, None))
        df.at[idx, 'village_ward'] = w if w else None
        df.at[idx, 'city_suburb'] = c if c else None
        df.at[idx, 'district'] = d if d else None
        df.at[idx, 'state'] = s if s else None
        df.at[idx, 'country'] = co if co else None
        df.at[idx, 'area_type'] = atype
        df.at[idx, 'geocoded_address'] = f

    total_processed = len(df)
    successful_count = sum(1 for f in df['geocoded_address'] if f and f != "Not Found")
    return df, total_processed, successful_count