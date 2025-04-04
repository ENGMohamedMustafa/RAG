import streamlit as st
import PyPDF2
import pandas as pd
import re
import io
import tempfile
import os
import google.generativeai as genai
import json
from google.api_core import exceptions

# Initialize page config
st.set_page_config(
    page_title="Medical Question Extractor Pro",
    layout="wide",
    page_icon="📚"
)

# API key input with secure handling
api_key = st.text_input("Enter Google Gemini API Key:", type="password", key="gemini_api_key")
if not api_key:
    st.warning("Please enter your Gemini API key to continue")
    st.stop()

# Initialize Gemini with automatic model detection
try:
    genai.configure(api_key=api_key)
    
    # Get available models and select the most capable one
    available_models = genai.list_models()
    model_priority = [
        'gemini-1.5-pro-latest',
        'gemini-1.5-pro',
        'gemini-pro',
        'models/gemini-pro'
    ]
    
    selected_model = None
    for model_name in model_priority:
        if any(model_name in m.name for m in available_models):
            selected_model = model_name
            break
    
    if not selected_model:
        st.error("No compatible model found. Available models:\n" + 
                "\n".join([m.name for m in available_models]))
        st.stop()
    
    model = genai.GenerativeModel(selected_model)
    st.session_state['gemini_model'] = selected_model
    
except exceptions.NotFound as e:
    st.error(f"Model configuration error: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"API initialization failed: {str(e)}")
    st.stop()

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with enhanced preprocessing"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        progress_bar = st.progress(0)
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            
            # Clean common PDF artifacts
            page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
            page_text = re.sub(r'-\n', '', page_text)   # Fix hyphenated words
            text += page_text + "\n"
            progress_bar.progress((i + 1) / len(pdf_reader.pages))
        
        progress_bar.empty()
        return text.strip()
    
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return None

def analyze_medical_text(text):
    """Analyze text with specialized medical question detection"""
    try:
        # Enhanced prompt for medical question detection
        prompt = f"""
        You are a medical education specialist analyzing textbook content.
        
        FOR PRETEST® FORMAT:
        1. Questions are numbered (1., 2., etc.)
        2. Each has exactly 4 options (A-D)
        3. Answers are marked with "Answer:" or "Correct Answer:"
        4. Explanations follow answers
        
        FOR BOARD REVIEW BOOKS:
        1. Look for "Practice Questions" sections
        2. Detect questions ending with ?
        3. Options are typically lettered (A) (B) (C) (D)
        
        RETURN ONLY VALID JSON with this structure:
        {{
            "content_type": "questions" | "toc" | "text_content",
            "questions": [
                {{
                    "id": number,
                    "question": "text?",
                    "options": ["A", "B", "C", "D"],
                    "answer": "A"|"B"|"C"|"D",
                    "explanation": "text"
                }}
            ],
            "message": "additional info"
        }}
        
        CONTENT TO ANALYZE:
        {text[:15000]}  # First 15k characters
        """
        
        response = model.generate_content(prompt)
        
        # Robust JSON parsing
        try:
            # Remove all markdown formatting
            clean_json = re.sub(r'```(json)?|```', '', response.text).strip()
            # Handle cases where response includes headers
            if clean_json.startswith('{'):
                return json.loads(clean_json)
            else:
                # Find the first valid JSON object
                json_match = re.search(r'\{.*\}', clean_json, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {
                    "content_type": "parse_error",
                    "message": "No valid JSON found",
                    "raw_response": response.text
                }
        except json.JSONDecodeError:
            return {
                "content_type": "parse_error",
                "message": "Invalid JSON format",
                "raw_response": response.text
            }
            
    except exceptions.InvalidArgument as e:
        st.error(f"API input error: {str(e)}")
        return None
    except exceptions.ResourceExhausted as e:
        st.error(f"API quota exceeded: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

def create_question_dataframe(analysis_result):
    """Create structured DataFrame from analysis results"""
    if not analysis_result or analysis_result.get("content_type") != "questions":
        return None
    
    questions = analysis_result.get("questions", [])
    if not questions:
        return None
    
    structured_data = []
    for q in questions:
        # Ensure exactly 4 options
        options = (q.get('options', []) + [''] * 4)[:4]
        
        structured_data.append({
            'Q#': q.get('id', len(structured_data) + 1),
            'Question': q.get('question', '').strip(),
            'A': options[0].strip() if options[0] else '',
            'B': options[1].strip() if options[1] else '',
            'C': options[2].strip() if options[2] else '',
            'D': options[3].strip() if options[3] else '',
            'Correct': q.get('answer', '').upper().strip(),
            'Explanation': q.get('explanation', '').strip()
        })
    
    return pd.DataFrame(structured_data)

def display_results(analysis_result):
    """Display analysis results appropriately"""
    content_type = analysis_result.get("content_type", "unknown")
    
    if content_type == "questions":
        df = create_question_dataframe(analysis_result)
        if df is not None and not df.empty:
            st.success(f"✅ Extracted {len(df)} practice questions!")
            
            # Display dataframe with optimized formatting
            st.dataframe(
                df,
                column_config={
                    "Q#": st.column_config.NumberColumn("Question #"),
                    "Question": "Question Text",
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D",
                    "Correct": "Correct Answer",
                    "Explanation": "Explanation"
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
            
            # Export options
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Download as CSV",
                data=csv,
                file_name="medical_questions.csv",
                mime="text/csv",
                help="Download all questions in CSV format"
            )
        else:
            st.warning("No questions found in this section")
    
    elif content_type == "toc":
        st.info("📖 Table of Contents Detected")
        st.write("""
        **For best results:**
        1. Navigate past the table of contents
        2. Upload pages containing:
           - Numbered questions (1., 2., etc.)
           - Multiple-choice options (A-D)
           - Clear answer explanations
        """)
    
    elif content_type == "text_content":
        st.info("📚 Textbook Content Detected")
        st.write("""
        **This appears to be regular textbook content.**
        Try uploading pages from:
        - Question banks
        - Practice exam sections
        - Chapter review questions
        """)
    
    else:
        st.warning("Unable to determine content type")
        if "raw_response" in analysis_result:
            with st.expander("Technical Details"):
                st.text(analysis_result["raw_response"])

def main():
    st.title("📚 Medical Question Extractor Pro")
    st.caption(f"Using model: {st.session_state.get('gemini_model', 'Unknown')}")
    
    st.markdown("""
    **Upload PDF pages containing:**
    - PreTest® question banks
    - Board review questions
    - Medical practice exams
    """)
    
    uploaded_file = st.file_uploader(
        "Select PDF file (max 5MB)",
        type=["pdf"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File exceeds 5MB size limit")
            return
        
        with st.spinner("Analyzing medical content..."):
            # Use temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Step 1: Extract text
                text = extract_text_from_pdf(tmp_path)
                if not text:
                    st.error("Failed to extract text from PDF")
                    return
                
                # Step 2: Analyze content
                analysis = analyze_medical_text(text)
                if not analysis:
                    st.error("Content analysis failed")
                    return
                
                # Step 3: Display results
                display_results(analysis)
            
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

if __name__ == "__main__":
    main()