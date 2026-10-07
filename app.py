import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found!")
    st.stop()

from db_handler import load_csv_to_sqlite
from agent import create_agent, run_query, extract_sql_and_run
from visualizer import auto_visualize
from reporter import generate_summary, generate_pdf_report
from auth import get_authenticator, show_login
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="AI SQL Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e6edf3; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #161b22;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #8b949e;
        font-weight: 600;
        padding: 8px 20px;
        font-size: 14px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1f3864, #2e75b6) !important;
        color: white !important;
        border-radius: 8px;
    }
    .metric-card {
        background: linear-gradient(135deg, #161b22, #1f2937);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #00b4d8;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }
    .answer-box {
        background: linear-gradient(135deg, #0d2137, #1a2f4a);
        border-left: 4px solid #00b4d8;
        padding: 20px 24px;
        border-radius: 0 12px 12px 0;
        font-size: 15px;
        color: #caf0f8;
        line-height: 1.6;
    }
    .sql-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-top: 3px solid #f78166;
        border-radius: 0 0 12px 12px;
        padding: 16px;
        font-family: 'Courier New', monospace;
        color: #79c0ff;
        font-size: 13px;
        white-space: pre-wrap;
    }
    .sql-header {
        background: #21262d;
        border: 1px solid #30363d;
        border-bottom: none;
        border-radius: 12px 12px 0 0;
        padding: 8px 16px;
        font-size: 12px;
        color: #8b949e;
        font-family: monospace;
    }
    .chat-q {
        background: linear-gradient(135deg, #1f3864, #2e3f5c);
        border-radius: 12px 12px 0 12px;
        padding: 12px 16px;
        margin-bottom: 4px;
        color: #caf0f8;
        font-weight: 600;
        font-size: 14px;
    }
    .chat-a {
        background: #161b22;
        border-left: 3px solid #00b4d8;
        border-radius: 0 12px 12px 12px;
        padding: 12px 16px;
        margin-bottom: 16px;
        color: #e6edf3;
        font-size: 14px;
        line-height: 1.6;
    }
    .section-header {
        background: linear-gradient(135deg, #1f3864, #2e75b6);
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
        font-size: 16px;
        font-weight: 700;
        color: white;
    }
    .info-banner {
        background: linear-gradient(135deg, #0d2137, #1a3a5c);
        border: 1px solid #2e75b6;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13px;
        color: #90caf9;
        margin-bottom: 12px;
    }
    .success-banner {
        background: linear-gradient(135deg, #0d2b1a, #1a4a2e);
        border: 1px solid #2e7d46;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13px;
        color: #a5d6a7;
        margin-bottom: 12px;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f3864, #2e75b6) !important;
        border: none !important;
        color: white !important;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Authentication ────────────────────────────────────────────
authenticator = get_authenticator()
show_login(authenticator)

name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")

if authentication_status is False:
    st.error("❌ Incorrect username or password")
    st.stop()

if authentication_status is None:
    st.warning("👆 Please enter your username and password")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 16px 0;'>
        <div style='font-size:40px;'>🧠</div>
        <div style='font-size:18px; font-weight:700; color:#00b4d8;'>
            AI SQL Analyst
        </div>
        <div style='font-size:12px; color:#8b949e; margin-top:4px;'>
            👤 {name}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    auto_clean = st.checkbox(
        "🧹 Auto Clean Dataset",
        value=False,
        help="Removes duplicates, fills nulls, fixes data types"
    )

    if auto_clean:
        st.markdown(
            '<div class="success-banner">✅ Auto cleaning enabled</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-banner">ℹ️ Auto cleaning disabled</div>',
            unsafe_allow_html=True)

    model_choice = st.selectbox(
        "🤖 LLM Model",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b"
        ],
        help="gpt-oss-20b is fastest, 120b is smartest"
    )

    show_debug = st.checkbox("🐛 Debug Mode", value=False)
    st.divider()
    authenticator.logout("🚪 Logout", location="sidebar")
    st.markdown(
        '<div style="text-align:center; font-size:11px; color:#555; margin-top:8px;">Powered by Groq + LangChain + SQLite</div>',
        unsafe_allow_html=True)

# ── App Header ────────────────────────────────────────────────
st.markdown("""
<div style='padding: 20px 0 24px 0;'>
    <div style='font-size:28px; font-weight:800;
        background: linear-gradient(135deg, #00b4d8, #2e75b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;'>
        🧠 AI SQL Data Analyst
    </div>
    <div style='font-size:13px; color:#8b949e; margin-top:4px;'>
        Upload any CSV and ask questions in plain English
    </div>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px;'>
        <div style='font-size:64px;'>📂</div>
        <div style='font-size:22px; font-weight:700;
            color:#00b4d8; margin-top:16px;'>
            Upload a CSV to get started
        </div>
        <div style='font-size:14px; color:#8b949e; margin-top:8px;'>
            Use the sidebar to upload your dataset
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Session State ─────────────────────────────────────────────
for key, default in {
    "transformed_df": None,
    "transform_history": [],
    "chat_history": [],
    "chart_history": [],
    "ai_summary": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load CSV ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading CSV...")
def setup_db(file, clean):
    return load_csv_to_sqlite(file, auto_clean=clean)

engine, table_name, original_df, columns_info, cleaning_report = setup_db(
    uploaded_file, auto_clean)

df = st.session_state.transformed_df \
    if st.session_state.transformed_df is not None \
    else original_df.copy()

# ── Metric Cards ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Rows</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df.columns)}</div>
        <div class="metric-label">Columns</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.isnull().sum().sum():,}</div>
        <div class="metric-label">Null Values</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(st.session_state.chat_history)}</div>
        <div class="metric-label">Questions Asked</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Overview",
    "🔧 Transform",
    "💬 Query & Charts",
    "📊 AI Summary",
    "📄 Export Report"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<div class="section-header">📋 Dataset Preview</div>',
        unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '<div class="section-header">🗂️ Column Information</div>',
            unsafe_allow_html=True)
        col_df = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Nulls": [df[c].isnull().sum() for c in df.columns],
            "Unique": [df[c].nunique() for c in df.columns]
        })
        st.dataframe(col_df, use_container_width=True)

    with col_b:
        st.markdown(
            '<div class="section-header">🧹 Cleaning Report</div>',
            unsafe_allow_html=True)
        for item in cleaning_report:
            st.markdown(f"- {item}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">📈 Quick Stats</div>',
            unsafe_allow_html=True)
        try:
            st.dataframe(
                df.describe().round(2),
                use_container_width=True)
        except Exception:
            st.info("No numeric columns.")

# ════════════════════════════════════════════════════════════
# TAB 2 — TRANSFORM
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="section-header">🔧 Data Transformation</div>',
        unsafe_allow_html=True)
    st.caption("Tell the AI what to change — plain English")

    with st.expander("💡 Example instructions"):
        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown("""
- `Convert name column to lowercase`
- `Fill nulls in salary with mean`
- `Add column profit = revenue - cost`
- `Remove rows where age < 0`
- `Extract year from date column`
            """)
        with ex2:
            st.markdown("""
- `Convert price to numeric`
- `Rename cust_nm to customer_name`
- `Replace negative quantity with 0`
- `Capitalize values in city column`
- `Fill missing city with Mumbai`
            """)

    transform_prompt = st.text_area(
        "✏️ What do you want to change?",
        placeholder="e.g. Fill null values in salary with mean",
        height=100
    )

    col_t1, col_t2, col_t3 = st.columns([2, 2, 4])
    with col_t1:
        transform_btn = st.button(
            "⚡ Apply", type="primary", use_container_width=True)
    with col_t2:
        if st.button("↩️ Reset All", use_container_width=True):
            st.session_state.transformed_df = None
            st.session_state.transform_history = []
            st.success("✅ Reset to original!")
            st.rerun()

    if transform_btn and transform_prompt:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
            temperature=0
        )

        col_context = ", ".join(
            [f"{col} ({str(df[col].dtype)})" for col in df.columns])
        sample_data = df.head(3).to_string()

        transform_code_prompt = (
            "You are a Python Pandas expert. DataFrame is called df.\n"
            "Columns: " + col_context + "\n"
            "Sample:\n" + sample_data + "\n\n"
            "Task: " + transform_prompt + "\n\n"
            "Write ONLY Python code. No markdown. No explanation.\n"
            "- pandas=pd, numpy=np\n"
            "- Modify df directly\n"
            "- No imports, no prints\n"
            "Examples:\n"
            "df['col'] = df['col'].str.lower()\n"
            "df['col'] = df['col'].fillna(df['col'].mean())\n"
            "df['profit'] = df['revenue'] - df['cost']\n"
            "df = df[df['age'] >= 0]\n"
        )

        with st.spinner("🤔 Generating code..."):
            response = llm.invoke(transform_code_prompt)
            generated_code = response.content.strip()
            generated_code = generated_code.replace("```python", "")
            generated_code = generated_code.replace("```", "")
            generated_code = generated_code.strip()

            lines = generated_code.split('\n')
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('<') or stripped.startswith('>'):
                    continue
                if any(stripped.lower().startswith(w) for w in
                       ['thought', 'note', 'the ', 'this ',
                        'here', 'to ', 'we ', 'i ']):
                    continue
                if not any(c in stripped for c in
                           ['=', '(', '[', '.', 'df']):
                    continue
                clean_lines.append(line)
            generated_code = '\n'.join(clean_lines).strip()

        st.markdown(
            '<div class="section-header">📝 Generated Code</div>',
            unsafe_allow_html=True)
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

            df.to_sql(table_name, con=engine,
                      if_exists="replace", index=False)
            st.success("✅ Transformation applied!")

            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("Rows", f"{len(df):,}")
            col_p2.metric("Columns", len(df.columns))
            col_p3.metric("Nulls", f"{df.isnull().sum().sum():,}")
            st.dataframe(df.head(10), use_container_width=True)

            null_cols = df.isnull().sum()
            null_cols = null_cols[null_cols > 0]
            if len(null_cols) > 0:
                st.warning(
                    f"⚠️ Still has nulls in: {', '.join(null_cols.index.tolist())}")
            else:
                st.success("🎉 No null values remaining!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("💡 Try rephrasing your instruction")

    # ── Transform History with Time Travel ───────────────────
    if st.session_state.transform_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">📜 Transform History — Click Restore to Go Back</div>',
            unsafe_allow_html=True)
        st.caption(
            "💡 ↩️ Restore = go back to that step | 🗑️ Delete = remove that specific transform")

        for i, h in enumerate(st.session_state.transform_history):
            col_h1, col_h2, col_h3 = st.columns([5, 1, 1])

            with col_h1:
                with st.expander(
                        f"✅ Step {i + 1}: {h['prompt'][:70]}"):
                    st.code(h['code'], language="python")

            with col_h2:
                if st.button(
                        "↩️",
                        key=f"restore_{i}",
                        use_container_width=True,
                        help=f"Restore to Step {i + 1}"
                ):
                    restored_df = original_df.copy()
                    try:
                        for j in range(i + 1):
                            exec_globals = {
                                "df": restored_df,
                                "pd": pd,
                                "np": np
                            }
                            exec(
                                st.session_state.transform_history[j]['code'],
                                exec_globals
                            )
                            restored_df = exec_globals["df"]

                        st.session_state.transformed_df = restored_df
                        st.session_state.transform_history = \
                            st.session_state.transform_history[:i + 1]

                        restored_df.to_sql(
                            table_name,
                            con=engine,
                            if_exists="replace",
                            index=False
                        )
                        st.success(
                            f"✅ Restored to Step {i + 1}!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Could not restore: {e}")

            with col_h3:
                if st.button(
                        "🗑️",
                        key=f"delete_{i}",
                        use_container_width=True,
                        help=f"Delete Step {i + 1}"
                ):
                    st.session_state.transform_history.pop(i)
                    restored_df = original_df.copy()
                    try:
                        for h2 in st.session_state.transform_history:
                            exec_globals = {
                                "df": restored_df,
                                "pd": pd,
                                "np": np
                            }
                            exec(h2['code'], exec_globals)
                            restored_df = exec_globals["df"]

                        st.session_state.transformed_df = \
                            restored_df if st.session_state.transform_history \
                            else None
                        restored_df.to_sql(
                            table_name,
                            con=engine,
                            if_exists="replace",
                            index=False
                        )
                        st.success("✅ Step deleted!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    # ── Manual Column Rename ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">✏️ Rename Columns Manually</div>',
        unsafe_allow_html=True)
    st.caption(
        "Type new names directly — only changed names will be applied")

    rename_map = {}
    col_list = list(df.columns)
    cols_per_row = 3

    for row_start in range(0, len(col_list), cols_per_row):
        row_cols = col_list[row_start:row_start + cols_per_row]
        grid = st.columns(cols_per_row)
        for idx, col_name in enumerate(row_cols):
            with grid[idx]:
                new_name = st.text_input(
                    f"**{col_name}**",
                    value=col_name,
                    key=f"rename_{col_name}_{row_start}_{idx}",
                    placeholder=col_name
                )
                if new_name.strip() and new_name.strip() != col_name:
                    rename_map[col_name] = new_name.strip()

    col_r1, col_r2, col_r3 = st.columns([2, 2, 4])
    with col_r1:
        apply_rename = st.button(
            "✅ Apply Renames",
            type="primary",
            use_container_width=True
        )
    with col_r2:
        if rename_map:
            st.markdown(
                f"<div style='padding:8px; color:#00b4d8; font-size:13px;'>"
                f"🔄 {len(rename_map)} change(s) pending</div>",
                unsafe_allow_html=True
            )

    if apply_rename:
        if rename_map:
            try:
                df = df.rename(columns=rename_map)
                st.session_state.transformed_df = df

                # Add to history
                rename_code = (
                    "df = df.rename(columns=" + str(rename_map) + ")"
                )
                st.session_state.transform_history.append({
                    "prompt": f"Renamed columns: {rename_map}",
                    "code": rename_code
                })

                df.to_sql(
                    table_name,
                    con=engine,
                    if_exists="replace",
                    index=False
                )
                st.success(
                    f"✅ Renamed {len(rename_map)} column(s) successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error renaming: {e}")
        else:
            st.info("ℹ️ No changes detected — edit a column name first")

    # ── Download ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">⬇️ Download</div>',
        unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📥 Download Transformed CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="transformed_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl2:
        st.download_button(
            "📥 Download Original CSV",
            data=original_df.to_csv(index=False).encode('utf-8'),
            file_name="original_data.csv",
            mime="text/csv",
            use_container_width=True
        )

# ════════════════════════════════════════════════════════════
# TAB 3 — QUERY & CHARTS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="section-header">💬 Ask Questions About Your Data</div>',
        unsafe_allow_html=True)
    st.caption("AI converts your question to SQL and shows results with charts")

    question = st.text_input(
        "Your question:",
        placeholder="e.g. Show total sales by region as bar chart",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🔍 Analyze", type="primary")

    if analyze_btn and question:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_choice,
            temperature=0
        )

        col_context = ", ".join(
            [f"{col} ({str(df[col].dtype)})" for col in df.columns])
        enriched = f"{question}\n\n[Table: data | Columns: {col_context}]"

        agent_executor, _ = create_agent(engine, model_choice)

        with st.spinner("🤔 Analyzing..."):
            result = run_query(agent_executor, enriched)
            try:
                sql_query, result_df = extract_sql_and_run(
                    engine, question, llm)
            except Exception as e:
                sql_query = f"-- Error: {e}"
                result_df = None

        if show_debug:
            with st.expander("🐛 Debug"):
                st.write("Columns:", list(df.columns))
                st.write("Result:", result)

        st.markdown(
            '<div class="section-header">💡 Answer</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="sql-header">🗃️ Generated SQL</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="sql-box">{sql_query}</div>',
            unsafe_allow_html=True)

        fig = None
        if result_df is not None and not result_df.empty:
            fig = auto_visualize(result_df, question)

        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sql": sql_query
        })
        if fig:
            st.session_state.chart_history.append(
                (f"Q: {question}", fig))

        if result_df is not None and not result_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">📊 Results</div>',
                unsafe_allow_html=True)
            chart_tab, table_tab = st.tabs(["📈 Chart", "📋 Table"])
            with chart_tab:
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Chart not applicable for this result.")
            with table_tab:
                st.dataframe(result_df, use_container_width=True)
        elif result_df is not None and result_df.empty:
            st.warning("⚠️ No results found. Try rephrasing.")

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        col_hdr, col_clr = st.columns([4, 1])
        with col_hdr:
            st.markdown(
                f'<div class="section-header">💬 Chat History ({len(st.session_state.chat_history)})</div>',
                unsafe_allow_html=True)
        with col_clr:
            if st.button("🗑️ Clear"):
                st.session_state.chat_history = []
                st.session_state.chart_history = []
                st.rerun()

        for i, chat in enumerate(
                reversed(st.session_state.chat_history), 1):
            num = len(st.session_state.chat_history) - i + 1
            st.markdown(
                f'<div class="chat-q">Q{num}: {chat["question"]}</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="chat-a">{chat["answer"]}</div>',
                unsafe_allow_html=True)
            with st.expander("View SQL"):
                st.code(chat["sql"], language="sql")

# ════════════════════════════════════════════════════════════
# TAB 4 — AI SUMMARY
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div class="section-header">📊 AI Generated Summary</div>',
        unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([1, 4])
    with col_s1:
        if st.button("🤖 Generate", type="primary",
                     use_container_width=True):
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name=model_choice,
                temperature=0.3
            )
            with st.spinner("🔍 Analyzing..."):
                st.session_state.ai_summary = generate_summary(df, llm)
    with col_s2:
        if st.session_state.ai_summary:
            if st.button("🗑️ Clear"):
                st.session_state.ai_summary = None
                st.rerun()

    if st.session_state.ai_summary:
        st.markdown(
            f'<div class="answer-box">{st.session_state.ai_summary}</div>',
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_stat, col_null = st.columns(2)
        with col_stat:
            st.markdown(
                '<div class="section-header">📈 Statistics</div>',
                unsafe_allow_html=True)
            try:
                st.dataframe(
                    df.describe().round(2),
                    use_container_width=True)
            except Exception:
                st.info("No numeric columns.")
        with col_null:
            st.markdown(
                '<div class="section-header">🔍 Null Analysis</div>',
                unsafe_allow_html=True)
            null_df = pd.DataFrame({
                "Column": df.columns,
                "Nulls": df.isnull().sum().values,
                "Null %": (
                    df.isnull().sum().values / len(df) * 100
                ).round(1)
            })
            st.dataframe(null_df, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:40px;'>
            <div style='font-size:48px;'>🤖</div>
            <div style='font-size:16px; color:#8b949e; margin-top:12px;'>
                Click Generate to get AI insights
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — EXPORT REPORT
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        '<div class="section-header">📄 Export PDF Report</div>',
        unsafe_allow_html=True)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.chat_history)}</div>
            <div class="metric-label">Questions</div>
        </div>""", unsafe_allow_html=True)
    with col_info2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.chart_history)}</div>
            <div class="metric-label">Charts</div>
        </div>""", unsafe_allow_html=True)
    with col_info3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{"✅" if st.session_state.ai_summary else "❌"}</div>
            <div class="metric-label">Summary</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📄 Generate PDF", type="primary"):
        if not st.session_state.ai_summary:
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name=model_choice,
                temperature=0.3
            )
            with st.spinner("Generating summary..."):
                st.session_state.ai_summary = generate_summary(df, llm)

        with st.spinner("📄 Building PDF..."):
            try:
                pdf_bytes = generate_pdf_report(
                    df=df,
                    chat_history=st.session_state.chat_history,
                    summary=st.session_state.ai_summary,
                    figures=st.session_state.chart_history
                )
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name="ai_sql_analyst_report.pdf",
                    mime="application/pdf"
                )
                st.success("✅ PDF ready!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">📋 Report Contents</div>',
        unsafe_allow_html=True)
    for num, title, desc in [
        ("1.", "Dataset Overview", "Rows, columns, types"),
        ("2.", "AI Summary", "Auto-generated insights"),
        ("3.", "Key Statistics", "Min, max, mean per column"),
        ("4.", "Null Analysis", "Null counts and percentages"),
        ("5.", "Q&A History", "All questions, answers and SQL"),
        ("6.", "Visualizations", "All charts from session"),
    ]:
        st.markdown(f"**{num} {title}** — {desc}")