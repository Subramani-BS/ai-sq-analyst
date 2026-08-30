import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ✅ Load API key from Streamlit secrets or .env
try:
    groq_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_key
except Exception:
    groq_key = os.getenv("GROQ_API_KEY")

# ✅ Stop app if no key found
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found. Please add it in Streamlit Cloud Secrets.")
    st.stop()

from db_handler import load_csv_to_sqlite
from agent import create_agent, run_query, extract_sql_and_run
from visualizer import auto_visualize
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="AI SQL Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .sql-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        color: #79c0ff;
        font-size: 14px;
        white-space: pre-wrap;
    }
    .answer-box {
        background: #0d2137;
        border-left: 4px solid #00b4d8;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        font-size: 16px;
        color: #caf0f8;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI SQL Analyst")
    st.caption("Upload a CSV and ask questions in plain English")
    st.divider()
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
    model_choice = st.selectbox(
        "🤖 LLM Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"]
    )
    show_debug = st.checkbox("🐛 Show Debug Info", value=False)
    st.divider()
    st.caption("Powered by Groq + LangChain + SQLite")

st.title("AI SQL Data Analyst Agent")

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    st.stop()

# ── Session State ─────────────────────────────────────────────
if "transformed_df" not in st.session_state:
    st.session_state.transformed_df = None
if "transform_history" not in st.session_state:
    st.session_state.transform_history = []

# ── Load CSV ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading and cleaning CSV...")
def setup_db(file):
    return load_csv_to_sqlite(file)

engine, table_name, original_df, columns_info, cleaning_report = setup_db(
    uploaded_file)

# Use transformed df if exists
df = st.session_state.transformed_df \
    if st.session_state.transformed_df is not None \
    else original_df.copy()

# ── Data Preview ──────────────────────────────────────────────
with st.expander("📋 Data Preview", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Null Values", f"{df.isnull().sum().sum():,}")

# ── Cleaning Report ───────────────────────────────────────────
with st.expander("🧹 Auto Data Cleaning Report"):
    for item in cleaning_report:
        st.markdown(f"- {item}")

# ── Column Info ───────────────────────────────────────────────
with st.expander("🗂️ Column Names & Types"):
    col_df = pd.DataFrame(
        {col: str(df[col].dtype) for col in df.columns}.items(),
        columns=["Column Name", "Data Type"]
    )
    st.dataframe(col_df, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════
# DATA TRANSFORMATION
# ════════════════════════════════════════════════════════════
st.subheader("🔧 Data Transformation")
st.caption("Type anything you want to change — the AI understands plain English")

with st.expander("💡 Example instructions"):
    st.markdown("""
- `Fill null values in salary column with mean value`
- `Fill all null values in numeric columns with their mean`
- `Change date format to 10 July 2025`
- `Add new column profit = revenue - cost`
- `Remove rows where age is less than 0`
- `Convert price column from string to number`
- `Rename column cust_nm to customer_name`
- `Extract year from date column into new column called year`
- `Replace negative values in quantity with 0`
- `Capitalize all values in name column`
- `Remove all rows where salary is null`
- `Fill missing city values with Mumbai`
    """)

transform_prompt = st.text_area(
    "✏️ What do you want to change?",
    placeholder="e.g. Fill null values in salary column with mean value",
    height=80
)

col_t1, col_t2 = st.columns([1, 1])
with col_t1:
    transform_btn = st.button("⚡ Apply Transformation", type="primary")
with col_t2:
    if st.button("↩️ Reset to Original Data"):
        st.session_state.transformed_df = None
        st.session_state.transform_history = []
        st.success("✅ Reset to original!")
        st.rerun()

# ── Transform History ─────────────────────────────────────────
if st.session_state.transform_history:
    with st.expander(
            f"📜 Transform History ({len(st.session_state.transform_history)} applied)"):
        for i, h in enumerate(st.session_state.transform_history, 1):
            st.markdown(f"**{i}.** {h['prompt']}")
            st.code(h['code'], language="python")

# ── Apply Transformation ──────────────────────────────────────
if transform_btn and transform_prompt:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_choice,
        temperature=0
    )

    col_context = ", ".join(
        [f"{col} ({str(df[col].dtype)})" for col in df.columns])
    sample_data = df.head(3).to_string()

    transform_code_prompt = f"""
You are a Python Pandas expert. You have a DataFrame called 'df'.

Current columns and types: {col_context}

Sample rows:
{sample_data}

The user wants this transformation:
"{transform_prompt}"

Write ONLY raw executable Python code. No explanation. No markdown. No imports.

RULES:
- pandas is imported as pd
- numpy is imported as np
- Modify 'df' directly
- Handle edge cases safely
- If user says mean/median/mode use correct pandas function
- If user says fill null use fillna()
- If user says remove rows use boolean filtering
- If user says add column add it to df
- If user says change format modify the column
- Output only Python code nothing else
"""

    with st.spinner("🤔 Understanding your instruction..."):
        response = llm.invoke(transform_code_prompt)
        generated_code = response.content.strip()
        generated_code = (generated_code
                          .replace("```python", "")
                          .replace("```", "")
                          .strip())

    st.subheader("📝 Generated Code")
    st.code(generated_code, language="python")

    try:
        exec_globals = {"df": df.copy(), "pd": pd, "np": np}
        exec(generated_code, exec_globals)
        df = exec_globals["df"]

        st.session_state.transformed_df = df
        st.session_state.transform_history.append({
            "prompt": transform_prompt,
            "code": generated_code
        })

        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        st.success("✅ Transformation applied successfully!")

        with st.expander("👀 Updated Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", len(df.columns))
            c3.metric("Null Values", f"{df.isnull().sum().sum():,}")

        null_cols = df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        if len(null_cols) > 0:
            st.warning(
                f"⚠️ Still has nulls in: {', '.join(null_cols.index.tolist())}")
        else:
            st.success("🎉 No null values remaining!")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("💡 Try rephrasing your instruction more clearly")

st.divider()

# ════════════════════════════════════════════════════════════
# DOWNLOAD SECTION
# ════════════════════════════════════════════════════════════
st.subheader("⬇️ Download Dataset")

dl1, dl2 = st.columns(2)

with dl1:
    current_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned & Transformed CSV",
        data=current_csv,
        file_name="cleaned_transformed_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption(f"Current: {len(df):,} rows × {len(df.columns)} cols")

with dl2:
    original_csv = original_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Original CSV",
        data=original_csv,
        file_name="original_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption(
        f"Original: {len(original_df):,} rows × {len(original_df.columns)} cols")

st.divider()

# ════════════════════════════════════════════════════════════
# QUERY SECTION
# ════════════════════════════════════════════════════════════
st.subheader("💬 Ask Questions About Your Data")

question = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is the total sales by region?"
)

if st.button("🔍 Analyze", type="primary") and question:

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_choice,
        temperature=0
    )

    agent_executor, _ = create_agent(engine)
    col_context = ", ".join(
        [f"{col} ({str(df[col].dtype)})" for col in df.columns])
    enriched_question = f"{question}\n\n[Table: data | Columns: {col_context}]"

    with st.spinner("🤔 Thinking..."):
        result = run_query(agent_executor, enriched_question)
        try:
            sql_query, result_df = extract_sql_and_run(engine, question, llm)
        except Exception as e:
            sql_query = f"-- Could not generate SQL:\n-- {e}"
            result_df = None

    if show_debug:
        with st.expander("🐛 Debug Info"):
            st.write("**Columns:**", list(df.columns))
            st.write("**Question sent:**", enriched_question)
            st.write("**Raw result:**", result)

    col_answer, col_sql = st.columns([1, 1])
    with col_answer:
        st.subheader("💡 Answer")
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True
        )
    with col_sql:
        st.subheader("🗃️ Generated SQL")
        st.markdown(
            f'<div class="sql-box">{sql_query}</div>',
            unsafe_allow_html=True
        )

    st.divider()
    if result_df is not None and not result_df.empty:
        st.subheader("📊 Visualization")
        tab1, tab2 = st.tabs(["Chart", "Result Table"])
        with tab1:
            fig = auto_visualize(result_df, question)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Chart not applicable for this result.")
        with tab2:
            st.dataframe(result_df, use_container_width=True)
    elif result_df is not None and result_df.empty:
        st.warning("⚠️ Query returned no results. Try rephrasing.")