
import streamlit as st
import PyPDF2
import pandas as pd
import re
from transformers import pipeline
import io

# Initialize QA pipeline
qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def parse_book_data(text):
    """Parse text to extract book-related information"""
    # Initialize empty lists to store data
    book_data = []
    
    # Regular expressions for parsing
    patterns = {
        'book_name': r'Book:\s*(.*?)(?=Chapter|\n|$)',
        'chapter': r'Chapter\s*(\d+)',
        'question': r'Question\s*\d+:\s*(.*?)(?=A\)|$)',
        'choices': r'([A-D]\).*?)(?=[A-D]\)|$)',
        'correct_answer': r'Correct Answer:\s*([A-D])',
        'explanation': r'Explanation:\s*(.*?)(?=\n|$)'
    }
    
    # Split text into sections (assuming each question is separated by some delimiter)
    sections = text.split('\n\n')
    
    for section in sections:
        data = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, section)
            if match:
                data[key] = match.group(1).strip()
            else:
                data[key] = ''
                
        if any(data.values()):  # Only add if at least one field was found
            book_data.append(data)
            
    return pd.DataFrame(book_data)

def answer_question(question, context):
    """Get answer for user question using QA model"""
    result = qa_model(question=question, context=context)
    return result['answer']

# Streamlit UI
st.title("Book Content Analyzer")

# File upload
uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Parse book data
    df = parse_book_data(text)
    
    # Display data table
    st.subheader("Extracted Book Data")
    st.dataframe(df)
    
    # Question answering section
    st.subheader("Ask Questions About the Content")
    user_question = st.text_input("Enter your question:")
    if user_question:
        answer = answer_question(user_question, text)
        st.write("Answer:", answer)
    
    # Download button for CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="book_data.csv",
        mime="text/csv"
    )
    # Download button for PDF
    pdf = io.BytesIO()
    pdf.write(text.encode('utf-8'))
    pdf.seek(0)
    st.download_button(
        label="Download PDF",
        data=pdf,
        file_name="book_content.pdf",
        mime="application/pdf"
    ) 
    