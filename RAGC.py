
import streamlit as st
import PyPDF2
import pandas as pd
from transformers import pipeline
import io
import base64

# Set page config
st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")

# Initialize the Q&A pipeline
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_model = load_qa_model()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

# Function to get answer from model
def get_answer(question, context):
    try:
        result = qa_model(question=question, context=context)
        return result['answer']
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

# Function to create download link
def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="extracted_data.csv">Download CSV File</a>'
    return href

# Main app
def main():
    st.title("PDF Question & Answer System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        # Extract text
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
            
        if extracted_text:
            st.success("Text extracted successfully!")
            
            # Create basic structured data
            data = {
                'content': [extracted_text],
                'length': [len(extracted_text)],
                'words': [len(extracted_text.split())]
            }
            df = pd.DataFrame(data)
            
            # Display extracted text
            with st.expander("View extracted text"):
                st.text(extracted_text)
            
            # Question input
            question = st.text_input("Ask a question about the document:")
            
            if question:
                with st.spinner("Generating answer..."):
                    answer = get_answer(question, extracted_text)
                    if answer:
                        st.write("Answer:", answer)
            
            # Download option
            st.markdown(get_download_link(df), unsafe_allow_html=True)
            
    else:
        st.info("Please upload a PDF file to begin.")
        st.stop()


if __name__ == "__main__":
    main()
