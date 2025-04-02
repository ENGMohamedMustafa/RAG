import pdfplumber
import pandas as pd
import re
import streamlit as st
import tempfile
from transformers import pipeline

# Load NLP model for better text extraction
qa_pipeline = pipeline("question-answering")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Function to parse questions using NLP model
def parse_questions(text, book_name):
    questions = []
    chapters = re.split(r'Chapter\s+\d+', text)  # Split by chapters
    chapter_num = 1
    
    for chapter in chapters[1:]:  # Skip the first part before the first chapter
        chapter_lines = chapter.strip().split("\n")
        qid = 1
        current_question = {}
        
        for i, line in enumerate(chapter_lines):
            if re.match(r'Q\d+[:.]?', line, re.IGNORECASE):  # Question identifier
                if current_question:  # Save previous question before starting a new one
                    questions.append(current_question)
                
                current_question = {
                    "Bookname": book_name,
                    "Chapter": f"Chapter {chapter_num}",
                    "QId": qid,
                    "Question": line,
                    "Choices": "",
                    "Correct answer": "",
                    "Explanation": ""
                }
                qid += 1
            elif re.match(r'[A-D][.)]', line):  # Choices (A, B, C, D)
                current_question["Choices"] += line + "\n"
            elif "Correct Answer:" in line:
                current_question["Correct answer"] = line.split(":")[-1].strip()
            elif "Explanation:" in line:
                current_question["Explanation"] = line.split(":")[-1].strip()
            
            # Use NLP model to extract additional insights
            elif current_question and len(chapter_lines) > i + 1:
                context = " ".join(chapter_lines[max(0, i - 2): i + 3])
                qa_result = qa_pipeline(question=current_question["Question"], context=context)
                current_question["Explanation"] = qa_result['answer']
        
        if current_question:
            questions.append(current_question)
        chapter_num += 1
    
    return questions

# Streamlit UI
def main():
    st.title("PDF to CSV Question Extractor with NLP")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name
        
        book_name = uploaded_file.name.replace(".pdf", "")
        text = extract_text_from_pdf(temp_pdf_path)
        questions = parse_questions(text, book_name)
        
        if questions:
            df = pd.DataFrame(questions)
            st.write(df)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="extracted_questions.csv",
                mime="text/csv"
            )
        else:
            st.warning("No questions were found in the PDF.")

if __name__ == "__main__":
    main()

