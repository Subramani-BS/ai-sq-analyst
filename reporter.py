import pandas as pd
from fpdf import FPDF
import plotly.express as px
import io
import os
import tempfile


class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(31, 56, 100)
        self.rect(0, 0, 210, 20, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 12)
        self.set_xy(0, 0)
        self.cell(0, 20, "AI SQL Data Analyst Report", align="C")
        self.ln(25)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_fill_color(46, 117, 182)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 10, title, fill=True, ln=True)
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", size=9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6, text)
        self.ln(2)


def generate_summary(df, llm):
    col_info = ", ".join(
        [f"{col} ({str(df[col].dtype)})" for col in df.columns])
    stats = df.describe().to_string()
    null_info = df.isnull().sum().to_string()

    prompt = (
        "You are a data analyst. Given this dataset info:\n"
        "Columns: " + col_info + "\n"
        "Statistics:\n" + stats + "\n"
        "Null counts:\n" + null_info + "\n\n"
        "Write a comprehensive data summary report including:\n"
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
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Dataset Summary ───────────────────────────────────
    pdf.section_title("1. Dataset Overview")
    pdf.body_text(f"Rows: {len(df):,}   |   Columns: {len(df.columns)}")
    pdf.body_text(f"Columns: {', '.join(df.columns.tolist())}")
    pdf.ln(3)

    # ── AI Summary ────────────────────────────────────────
    pdf.section_title("2. AI Generated Summary")
    clean_summary = summary.encode('latin-1', 'replace').decode('latin-1')
    pdf.body_text(clean_summary)
    pdf.ln(3)

    # ── Statistics ────────────────────────────────────────
    pdf.section_title("3. Dataset Statistics")
    stats = df.describe()
    pdf.set_font("Courier", size=7)
    pdf.set_text_color(30, 30, 30)
    stats_text = stats.to_string()
    clean_stats = stats_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, clean_stats)
    pdf.ln(3)

    # ── Chat History ──────────────────────────────────────
    if chat_history:
        pdf.section_title("4. Questions & Answers")
        for i, chat in enumerate(chat_history, 1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(46, 117, 182)
            q = f"Q{i}: {chat['question']}"
            clean_q = q.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, clean_q)
            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(50, 50, 50)
            a = f"A: {chat['answer']}"
            clean_a = a.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, clean_a)
            pdf.ln(3)

    # ── Charts ────────────────────────────────────────────
    if figures:
        pdf.section_title("5. Visualizations")
        for i, (title, fig) in enumerate(figures):
            try:
                with tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False) as tmp:
                    fig.write_image(tmp.name, width=700, height=400)
                    tmp_path = tmp.name

                clean_title = title.encode(
                    'latin-1', 'replace').decode('latin-1')
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(46, 117, 182)
                pdf.cell(0, 6, clean_title, ln=True)
                pdf.image(tmp_path, w=180)
                pdf.ln(5)
                os.unlink(tmp_path)
            except Exception:
                pass

    return bytes(pdf.output())