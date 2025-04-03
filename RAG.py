import streamlit as st
import PyPDF2
import pandas as pd
import os
import re
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline
import tempfile

# Configure app page
st.set_page_config(page_title="PDF QA Extractor", layout="wide")

# Extract text from PDF (supports both text-based and OCR)
def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        
        text = ""
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        
        if not text.strip():  # If no text was extracted, use OCR
            images = convert_from_path(temp_file_path)
            text = "\n".join([pytesseract.image_to_string(img) for img in images])

        os.unlink(temp_file_path)
        return text.strip()
    
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Parse text into structured questions and answers
def parse_text_to_data(text):
    try:
        lines = text.split('\n')
        data = []
        bookname = "Book" 
        chapter = "Chapter"
        question_pattern = re.compile(r"(Q\d+\.|.*\?)")  # Supports questions in Q1. format or ending with ?

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Book:"):
                bookname = line.replace("Book:", "").strip()

            elif line.startswith("Chapter:"):
                chapter = line.replace("Chapter:", "").strip()

            elif question_pattern.match(line):
                qid = f"Q{len(data) + 1}"
                question_data = {
                    "Bookname": bookname,
                    "Chapter": chapter,
                    "QId": qid,
                    "Question": line,
                    "Choices": "",
                    "Correct Answer": "",
                    "Explanation": ""
                }
                data.append(question_data)

            elif re.match(r"^[A-D]\)", line):  # Recognize choices
                if data:
                    data[-1]["Choices"] = data[-1]["Choices"] + " | " + line if data[-1]["Choices"] else line

            elif line.startswith("Answer:"):
                if data:
                    data[-1]["Correct Answer"] = line.replace("Answer:", "").strip()

            elif line.startswith("Explanation:"):
                if data:
                    data[-1]["Explanation"] = line.replace("Explanation:", "").strip()

        return data

    except Exception as e:
        st.error(f"Error parsing text data: {e}")
        return []

# AI model for answering questions
def answer_question(question, context):
    try:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        result = qa_pipeline(question=question, context=context)
        return result['answer'], result['score']
    except Exception as e:
        st.error(f"Error in question answering: {e}")
        return "Unable to generate answer", 0

# Main application interface
def main():
    st.title("📄 PDF Question Extractor & AI Q&A")
    st.markdown("🧐 **Extract structured data from PDF files and answer questions using AI**.")

    uploaded_file = st.file_uploader("📂 **Upload PDF file**", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("⏳ **Extracting text...**"):
            pdf_text = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            st.success("✅ **Text extracted successfully!**")

            with st.expander("📜 **View extracted raw text**"):
                st.text_area("", pdf_text, height=200)

            with st.spinner("⏳ **Analyzing data...**"):
                parsed_data = parse_text_to_data(pdf_text)

            if parsed_data:
                df = pd.DataFrame(parsed_data)

                st.subheader("📋 **Extracted Data**")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 **Download as CSV**", data=csv, file_name="extracted_data.csv", mime="text/csv")

                st.subheader("💡 **Ask AI about the content**")
                context = " ".join([f"{row['Question']} {row['Choices']} {row['Correct Answer']} {row['Explanation']}" for row in parsed_data])

                question = st.text_input("✍️ **Enter your question here:**")
                if question:
                    with st.spinner("🤖 **Generating answer...**"):
                        answer, confidence = answer_question(question, context)
                    st.markdown(f"**🔹 Answer:** {answer}  \n**🔹 Confidence:** {confidence:.2%}")

            else:
                st.warning("⚠️ **No structured data found. The PDF format may not be supported.**")
        else:
            st.error("❌ **Failed to extract text. Please ensure the PDF is valid.**")

if __name__ == '__main__':
    main()
