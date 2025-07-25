# streamlit_app.py


import os
os.environ["CREWAI_STORAGE_BACKEND"] = "no"
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from ai_analyst_crew.crew import AiAnalystCrew
from pathlib import Path
from fpdf import FPDF
import re
import tempfile

def render_report(report_text: str):
    st.markdown("### ✅ Final Recommendation Report")

    # פיצול החלקים לפי כותרות Markdown
    sections = re.split(r"^##+\s+", report_text, flags=re.MULTILINE)
    titles = re.findall(r"^##+\s+(.+)", report_text, flags=re.MULTILINE)

    if len(sections) > 1:
        # החלק הראשון הוא כותרת כללית
        st.subheader(sections[0].strip())

        for i, section in enumerate(sections[1:]):
            with st.expander(f"📌 {titles[i]}", expanded=True):
                st.markdown(section.strip())
    else:
        st.markdown(report_text)

def export_pdf(report_text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in report_text.split("\n"):
        pdf.multi_cell(0, 10, txt=line, align="L")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        return Path(tmp_file.name).read_bytes()

# הגדרות עמוד
st.set_page_config(page_title="AI Stock Recommender", layout="centered")
st.title("📈 AI Stock Recommender")

# קלט מהמשתמש
symbol = st.text_input("Enter Stock Symbol (e.g., NVDA, AAPL, TSLA)", "NVDA").upper()
days_choice = st.radio("Forecast Horizon (Days)", options=[1, 5, 21], horizontal=True)

# כפתור להרצה
if st.button("🔍 Run Analysis"):
    with st.spinner("Running crew... please wait..."):
        try:
            result = AiAnalystCrew().crew().kickoff(inputs={
                "symbol": symbol,
                "days_ahead": days_choice
            })

            final_output = result.output if hasattr(result, "output") else str(result)
            render_report(final_output)

            # שמירה לקובץ מקומי
            output_dir = Path(__file__).resolve().parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{symbol}_{days_choice}_report.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_output)

            st.success(f"Markdown saved to: {output_path}")

            # כפתור להורדת PDF
            pdf_bytes = export_pdf(final_output)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{symbol}_forecast_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"❌ Error occurred while running the crew:\n{e}")
