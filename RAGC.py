import streamlit as st
import PyPDF2
import pandas as pd
import tempfile
import re
import os
from pdf2image import convert_from_path
from PIL import Image
import io
import numpy as np
import concurrent.futures
import logging
import cv2
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 200
OCR_RESOLUTION = 600  # DPI
MIN_TEXT_LENGTH = 200  # Threshold for OCR fallback

def preprocess_image(image):
    """Enhanced image preprocessing for better OCR accuracy"""
    try:
        # Convert to grayscale
        img = image.convert('L')
        
        # Contrast stretching
        img_array = np.array(img)
        p2, p98 = np.percentile(img_array, (2, 98))
        img_array = np.clip((img_array - p2) * (255.0 / (p98 - p2)), 0, 255)
        
        # Denoising
        img_array = cv2.fastNlMeansDenoising(img_array.astype(np.uint8), None, 10, 7, 21)
        
        return Image.fromarray(img_array)
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {str(e)}")
        return image

def process_page(img):
    """Process a single page with OCR"""
    try:
        processed_img = preprocess_image(img)
        return pytesseract.image_to_string(processed_img, config='--oem 3 --psm 6')
    except Exception as e:
        logger.error(f"Page processing failed: {str(e)}")
        return ""

def extract_text_with_ocr(pdf_path):
    """Parallel OCR processing for faster extraction"""
    try:
        images = convert_from_path(pdf_path, dpi=OCR_RESOLUTION)
        text_chunks = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_page, img) for img in images]
            for future in concurrent.futures.as_completed(futures):
                text_chunks.append(future.result())
        
        return "\n".join([text for text in text_chunks if text.strip()])
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_file):
    """Optimized PDF text extraction with fallback to OCR"""
    try:
        if pdf_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        
        # Try direct text extraction first
        text = ""
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        # Fallback to OCR if needed
        if len(text.strip()) < MIN_TEXT_LENGTH:
            logger.info("Insufficient text, switching to OCR")
            ocr_text = extract_text_with_ocr(temp_file_path)
            text += "\n" + ocr_text
        
        return text.strip()
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise
    finally:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass

def parse_questions(text):
    """Advanced question parsing with regex"""
    questions = []
    question_pattern = re.compile(
        r'(?P<prefix>Q\d+[:.]|Question\s+\d+[:.]?|\d+[.)])\s*(?P<text>.*?\?)',
        re.IGNORECASE
    )
    choice_pattern = re.compile(r'^[A-D][.)]\s*(?P<text>.*)$', re.MULTILINE)
    answer_pattern = re.compile(r'Answer[:.]?\s*(?P<letter>[A-D])', re.IGNORECASE)
    explanation_pattern = re.compile(
        r'Explanation[:.]?\s*(?P<text>.*?)(?=(Q\d+|Question\s+\d+|\d+[.)]|\Z))', 
        re.IGNORECASE | re.DOTALL
    )
    
    for q_match in question_pattern.finditer(text):
        question = {
            'id': q_match.group('prefix').strip('.'),
            'text': q_match.group('text').strip(),
            'choices': [],
            'answer': '',
            'explanation': ''
        }
        
        # Extract choices
        choices_text = text[q_match.end():q_match.end()+500]
        question['choices'] = [
            c.group().strip() 
            for c in choice_pattern.finditer(choices_text)
        ][:4]  # Only take first 4 options (A-D)
        
        # Extract answer
        if ans_match := answer_pattern.search(choices_text):
            question['answer'] = ans_match.group('letter').upper()
        
        # Extract explanation
        if exp_match := explanation_pattern.search(text[q_match.end():]):
            question['explanation'] = exp_match.group('text').strip()
        
        questions.append(question)
    
    return questions

def create_structured_dataframe(questions):
    """Create a structured DataFrame with separate columns for options, answers, and explanations"""
    structured_data = []
    
    for q in questions:
        # Ensure we have exactly 4 options (pad with empty strings if needed)
        options = q['choices'] + [''] * (4 - len(q['choices']))
        
        row = {
            'ID': q['id'],
            'Question': q['text'],
            'Option A': options[0] if len(options) > 0 else '',
            'Option B': options[1] if len(options) > 1 else '',
            'Option C': options[2] if len(options) > 2 else '',
            'Option D': options[3] if len(options) > 3 else '',
            'Correct Answer': q['answer'],
            'Explanation': q['explanation']
        }
        structured_data.append(row)
    
    return pd.DataFrame(structured_data)

def main():
    st.set_page_config(
        page_title="PDF QA Extractor",
        layout="wide",
        page_icon="📚"
    )
    
    st.title("📚 PDF QA Extractor")
    st.markdown("""
    **Extract questions from textbooks and exams**
    - Supports both digital and scanned PDFs
    - Organizes questions with options, answers, and explanations
    """)
    
    uploaded_file = st.file_uploader(
        "Upload PDF (max 200MB)",
        type=["pdf"]
    )
    
    if uploaded_file:
        try:
            with st.spinner("Processing document..."):
                with st.status("Extracting content...", expanded=True) as status:
                    status.write("📄 Reading PDF content...")
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    
                    if not extracted_text.strip():
                        st.warning("No text could be extracted")
                        return
                    
                    status.write("🔍 Identifying questions...")
                    questions = parse_questions(extracted_text)
                    
                    if not questions:
                        st.warning("No questions found. Showing raw text:")
                        st.text_area("Extracted Text", extracted_text, height=300)
                        return
                    
                    status.write("📊 Formatting results...")
                    df = create_structured_dataframe(questions)
                    
                    st.success(f"✅ Found {len(df)} questions!")
                    st.dataframe(df, use_container_width=True, height=600)
                    
                    # Export options
                    st.divider()
                    st.markdown("### Export Options")
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "💾 Download CSV",
                            data=csv,
                            file_name="questions.csv",
                            mime="text/csv",
                            help="Download as CSV file with all questions and answers"
                        )
                    with col2:
                        st.download_button(
                            "💾 Download Excel",
                            data=excel_buffer,
                            file_name="questions.xlsx",
                            mime="application/vnd.ms-excel",
                            help="Download as Excel file with all questions and answers"
                        )
                    
                    status.update(label="Processing complete!", state="complete")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()