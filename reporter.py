import pandas as pd
from fpdf import FPDF
import os
import tempfile


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    return text.encode('latin-1', 'replace').decode('latin-1')


class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(31, 56, 100)
        self.rect(0, 0, 210, 18, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.set_xy(0, 4)
        self.cell(0, 10, "AI SQL Data Analyst Report", align="C")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_fill_color(46, 117, 182)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, clean_text(title), fill=True, ln=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", size=9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, clean_text(text))
        self.ln(1)


def generate_summary(df, llm):
    col_info = ", ".join(
        [f"{col} ({str(df[col].dtype)})" for col in df.columns])

    try:
        stats = df.describe().round(2).to_string()
    except Exception:
        stats = "Not available"

    null_info = df.isnull().sum().to_string()

    prompt = (
        "You are a data analyst. Given this dataset info:\n"
        "Columns: " + col_info + "\n"
        "Statistics:\n" + stats + "\n"
        "Null counts:\n" + null_info + "\n\n"
        "Write a comprehensive data summary including:\n"
        "1. What this dataset is about\n"
        "2. Key statistics and insights\n"
        "3. Data quality observations\n"
        "4. Top 3 recommendations for analysis\n"
        "Keep it under 300 words. Use plain text only, no special characters."
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_pdf_report(df, chat_history, summary, figures):
    pdf = PDFReport()
    pdf.set_margins(15, 25, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── 1. Dataset Overview ───────────────────────────────
    pdf.section_title("1. Dataset Overview")
    pdf.body_text(f"Total Rows: {len(df):,}")
    pdf.body_text(f"Total Columns: {len(df.columns)}")
    pdf.body_text("Column Names: " + clean_text(
        ", ".join(df.columns.tolist())))
    pdf.ln(3)

    # ── 2. AI Summary ─────────────────────────────────────
    pdf.section_title("2. AI Generated Summary")
    pdf.body_text(clean_text(summary))
    pdf.ln(3)

    # ── 3. Statistics (text format) ───────────────────────
    pdf.section_title("3. Key Statistics")
    try:
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            for col in num_cols[:6]:  # max 6 columns
                col_data = df[col].dropna()
                stat_text = (
                    f"{col}: "
                    f"Min={col_data.min():.2f}, "
                    f"Max={col_data.max():.2f}, "
                    f"Mean={col_data.mean():.2f}, "
                    f"Median={col_data.median():.2f}, "
                    f"Nulls={df[col].isnull().sum()}"
                )
                pdf.body_text(stat_text)
        else:
            pdf.body_text("No numeric columns found.")
    except Exception as e:
        pdf.body_text(f"Statistics error: {str(e)}")
    pdf.ln(3)

    # ── 4. Null Analysis ──────────────────────────────────
    pdf.section_title("4. Null Value Analysis")
    try:
        null_counts = df.isnull().sum()
        has_nulls = False
        for col in df.columns:
            count = null_counts[col]
            pct = round(count / len(df) * 100, 2)
            pdf.body_text(f"{col}: {count} nulls ({pct}%)")
            if count > 0:
                has_nulls = True
        if not has_nulls:
            pdf.body_text("No null values found in dataset.")
    except Exception as e:
        pdf.body_text(f"Null analysis error: {str(e)}")
    pdf.ln(3)

    # ── 5. Chat History ───────────────────────────────────
    if chat_history:
        pdf.section_title("5. Questions and Answers")
        for i, chat in enumerate(chat_history, 1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(46, 117, 182)
            pdf.multi_cell(
                0, 5,
                clean_text(f"Q{i}: {chat['question']}")
            )
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(50, 50, 50)
            answer = str(chat['answer'])[:600]
            pdf.multi_cell(
                0, 5,
                clean_text(f"Answer: {answer}")
            )
            if chat.get('sql') and not str(
                    chat['sql']).startswith('--'):
                pdf.set_font("Courier", size=7)
                pdf.set_text_color(100, 100, 100)
                sql_text = clean_text(
                    "SQL: " + str(chat['sql'])[:400])
                pdf.multi_cell(0, 4, sql_text)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
    else:
        pdf.section_title("5. Questions and Answers")
        pdf.body_text("No questions asked yet.")
        pdf.ln(3)

    # ── 6. Charts ─────────────────────────────────────────
    if figures:
        pdf.section_title("6. Visualizations")
        for title, fig in figures:
            try:
                with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False) as tmp:
                    fig.write_image(
                        tmp.name,
                        width=550,
                        height=320,
                        scale=2
                    )
                    tmp_path = tmp.name

                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(46, 117, 182)
                pdf.cell(0, 6, clean_text(str(title)[:80]), ln=True)
                pdf.set_text_color(0, 0, 0)
                pdf.image(tmp_path, w=160)
                pdf.ln(5)
                os.unlink(tmp_path)
            except Exception as e:
                pdf.set_font("Helvetica", size=8)
                pdf.set_text_color(150, 0, 0)
                pdf.body_text(f"Chart error: {str(e)}")
                pdf.ln(2)
    else:
        pdf.section_title("6. Visualizations")
        pdf.body_text(
            "No charts generated yet. Ask questions to generate charts.")

    return bytes(pdf.output())