import streamlit as st
import PyPDF2
import pandas as pd
import io
import base64
import re

def extract_quiz_data(text):
    """Extract quiz questions and related information from text"""
    quiz_data = []
    
    # Regular expressions for identifying different components
    chapter_pattern = r'Chapter\s+(\d+|[IVX]+)'
    question_pattern = r'(?:Q|Question)\s*\.?\s*(\d+)\s*[:.]?\s*([^\n]+)'
    choice_pattern = r'(?:[A-D])\s*\.\s*([^\n]+)'
    answer_pattern = r'(?:Answer|Correct Answer)\s*[:.]?\s*([A-D])'
    explanation_pattern = r'(?:Explanation|Solution)\s*[:.]?\s*([^\n]+)'
    
    # Split text into sections (you might need to adjust this based on your PDF structure)
    sections = text.split('\n\n')
    
    current_chapter = ""
    current_bookname = ""
    
    for section in sections:
        # Try to identify chapter
        chapter_match = re.search(chapter_pattern, section)
        if chapter_match:
            current_chapter = chapter_match.group(1)
            continue
            
        # Look for questions
        question_match = re.search(question_pattern, section)
        if question_match:
            qid = question_match.group(1)
            question = question_match.group(2).strip()
            
            # Extract choices
            choices = re.findall(choice_pattern, section)
            choices_dict = {}
            for i, choice in enumerate(['A', 'B', 'C', 'D'][:len(choices)]):
                choices_dict[choice] = choices[i].strip()
            
            # Find answer
            answer_match = re.search(answer_pattern, section)
            correct_answer = answer_match.group(1) if answer_match else ""
            
            # Find explanation
            explanation_match = re.search(explanation_pattern, section)
            explanation = explanation_match.group(1).strip() if explanation_match else ""
            
            # Create question data dictionary
            question_data = {
                "Bookname": current_bookname,
                "Chapter": current_chapter,
                "QId": qid,
                "Question": question,
                "Choices": choices_dict,
                "Correct_Answer": correct_answer,
                "Explanation": explanation
            }
            
            quiz_data.append(question_data)
    
    return quiz_data

def process_text_to_structured_data(text):
    if not text:
        return pd.DataFrame()
    
    # Extract quiz data
    quiz_data = extract_quiz_data(text)
    
    # Convert to DataFrame
    df = pd.DataFrame(quiz_data)
    
    # Clean and format the DataFrame
    df['Choices'] = df['Choices'].apply(lambda x: str(x))  # Convert choices dict to string
    
    return df

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text += f"\n--- Page {page_num} ---\n"
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="quiz_data.csv">Download CSV file</a>'
    return href

def display_quiz_statistics(df):
    st.subheader("Quiz Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Questions", len(df))
        
    with col2:
        if 'Chapter' in df.columns:
            st.metric("Number of Chapters", df['Chapter'].nunique())
            
    with col3:
        if 'Correct_Answer' in df.columns:
            st.metric("Questions with Answers", df['Correct_Answer'].notna().sum())

def main():
    st.title("Quiz PDF Extraction System")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            # Process text into structured data
            with st.spinner("Processing quiz data..."):
                df = process_text_to_structured_data(text)
            
            if not df.empty:
                # Display statistics
                display_quiz_statistics(df)
                
                # Display data
                st.subheader("Extracted Quiz Data")
                
                # Filter options
                st.sidebar.subheader("Filters")
                if 'Chapter' in df.columns:
                    selected_chapters = st.sidebar.multiselect(
                        "Select Chapters",
                        options=sorted(df['Chapter'].unique()),
                        default=sorted(df['Chapter'].unique())
                    )
                    df = df[df['Chapter'].isin(selected_chapters)]
                
                # Display questions
                for _, row in df.iterrows():
                    with st.expander(f"Question {row['QId']}: {row['Question'][:100]}..."):
                        st.write("**Question:**", row['Question'])
                        st.write("**Choices:**")
                        choices = eval(row['Choices']) if isinstance(row['Choices'], str) else row['Choices']
                        for choice, text in choices.items():
                            st.write(f"{choice}. {text}")
                        st.write("**Correct Answer:**", row['Correct_Answer'])
                        if row['Explanation']:
                            st.write("**Explanation:**", row['Explanation'])
                
                # Download link for CSV
                st.markdown(get_download_link(df), unsafe_allow_html=True)
                
                # Display raw text in expandable section
                with st.expander("View Raw Extracted Text"):
                    st.text_area("", text, height=300)
            else:
                st.warning("No quiz data could be extracted from the PDF.")
    else:
        st.info("Please upload a PDF file containing quiz questions to begin.")

if __name__ == "__main__":
    main()
