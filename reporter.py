import pandas as pd
from fpdf import FPDF
import os
import tempfile


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
        self.cell(0, 8, title, fill=True, ln=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", size=9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, text)
        self.ln(1)


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    return text.encode('latin-1', 'replace').decode('latin-1')


def generate_summary(df, llm):
    col_info = ", ".join(
        [f"{col} ({str(df[col].dtype)})" for col in df.columns])
    
    try:
        stats = df.describe().to_string()
    except Exception:
        stats = "Statistics not available"
    
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
        "Keep it professional and concise under 300 words."
    )
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_pdf_report(df, chat_history, summary, figures):
    pdf = PDFReport()
    pdf.set_margins(15, 20, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Dataset Overview ──────────────────────────────────
    pdf.section_title("1. Dataset Overview")
    pdf.body_text(f"Rows: {len(df):,}   |   Columns: {len(df.columns)}")
    pdf.body_text("Columns: " + clean_text(", ".join(df.columns.tolist())))
    pdf.ln(3)

    # ── AI Summary ────────────────────────────────────────
    pdf.section_title("2. AI Generated Summary")
    pdf.body_text(clean_text(summary))
    pdf.ln(3)

    # ── Statistics ────────────────────────────────────────
    pdf.section_title("3. Dataset Statistics")
    try:
        num_df = df.select_dtypes(include='number')
        if not num_df.empty:
            stats = num_df.describe().round(2)
            pdf.set_font("Courier", size=7)
            pdf.set_text_color(30, 30, 30)

            # Header row
            cols = ['Stat'] + list(stats.columns)
            col_width = min(25, 170 // len(cols))
            for col in cols:
                pdf.cell(col_width, 5,
                         clean_text(str(col))[:12],
                         border=1, align="C")
            pdf.ln()

            # Data rows
            for idx in stats.index:
                pdf.cell(col_width, 5,
                         clean_text(str(idx))[:12],
                         border=1, align="C")
                for col in stats.columns:
                    val = str(round(stats.loc[idx, col], 2))
                    pdf.cell(col_width, 5,
                             clean_text(val)[:12],
                             border=1, align="C")
                pdf.ln()
        else:
            pdf.body_text("No numeric columns found for statistics.")
    except Exception as e:
        pdf.body_text(f"Statistics could not be generated: {str(e)}")
    pdf.ln(4)

    # ── Chat History ──────────────────────────────────────
    if chat_history:
        pdf.section_title("4. Questions and Answers")
        for i, chat in enumerate(chat_history, 1):
            # Question
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(46, 117, 182)
            pdf.multi_cell(
                0, 5,
                clean_text(f"Q{i}: {chat['question']}")
            )
            # Answer
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(50, 50, 50)
            answer = chat['answer'][:500]
            pdf.multi_cell(
                0, 5,
                clean_text(f"A: {answer}")
            )
            # SQL
            if chat.get('sql') and not chat['sql'].startswith('--'):
                pdf.set_font("Courier", size=7)
                pdf.set_text_color(100, 100, 100)
                sql_clean = clean_text(chat['sql'][:300])
                pdf.multi_cell(0, 4, f"SQL: {sql_clean}")
            pdf.ln(3)

    # ── Charts ────────────────────────────────────────────
    if figures:
        pdf.section_title("5. Visualizations")
        for title, fig in figures:
            try:
                with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False) as tmp:
                    fig.write_image(
                        tmp.name,
                        width=600,
                        height=350,
                        scale=1.5
                    )
                    tmp_path = tmp.name

                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(46, 117, 182)
                pdf.cell(0, 6, clean_text(title), ln=True)
                pdf.image(tmp_path, w=170)
                pdf.ln(4)
                os.unlink(tmp_path)
            except Exception as e:
                pdf.set_font("Helvetica", size=8)
                pdf.set_text_color(150, 0, 0)
                pdf.cell(0, 5,
                         clean_text(f"Chart could not be rendered: {str(e)}"),
                         ln=True)
                pdf.ln(2)

    return bytes(pdf.output())