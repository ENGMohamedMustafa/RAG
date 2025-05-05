import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# Page title
st.set_page_config(page_title="Document Q&A RAG App")
st.title("📚 RAG-powered Document Q&A")

# Set up OpenAI API key
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

with st.sidebar:
    st.subheader("API Configuration")
    api_key = st.text_input("Enter your OpenAI API key:", value=st.session_state["OPENAI_API_KEY"], type="password")
    st.session_state["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    
    st.subheader("Model Settings")
    model_name = st.selectbox("Select Model:", ["gpt-3.5-turbo", "gpt-4"], index=0)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    st.subheader("Document Settings")
    chunk_size = st.slider("Chunk Size:", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap:", min_value=0, max_value=500, value=100, step=10)

# Initialize session states
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Function to process the document
def process_documents(uploaded_files):
    # Create a temporary directory to store uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        documents = []
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file to the temporary directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Load and process the document based on file extension
                if file_path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                else:
                    st.warning(f"Unsupported file format: {uploaded_file.name}")
            except ImportError as e:
                st.error(f"Error loading {uploaded_file.name}: Missing PDF dependencies. Please install PyPDF2 with 'pip install pypdf2'")
                return False
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                return False
        
        if not documents:
            st.error("No documents were successfully loaded. Please check your files and try again.")
            return False
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Display number of chunks created
        st.write(f"✅ Created {len(chunks)} document chunks")
        
        # Create embeddings and vector store
        if api_key:
            try:
                embeddings = OpenAIEmbeddings()
                vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.vector_store = vector_store
                st.session_state.processed_docs = True
                return True
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                return False
        else:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return False

# File uploader
uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=["pdf", "txt"])

# Process documents when uploaded
if uploaded_files and not st.session_state.processed_docs:
    with st.spinner("Processing documents..."):
        process_documents(uploaded_files)

# Reset the conversation
if st.button("Reset Conversation"):
    st.session_state.chat_history = []
    st.session_state.processed_docs = False
    st.session_state.conversation = None
    st.session_state.vector_store = None
    st.experimental_rerun()

# Query input and response
if st.session_state.processed_docs and st.session_state.vector_store:
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        with st.spinner("Generating response..."):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                try:
                    # Create the chain on demand
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    
                    # Create template for the response generation
                    template = """
                    You are a helpful assistant for question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know.
                    Use three sentences maximum and keep the answer concise.
                    
                    Question: {question}
                    
                    Context: {context}
                    
                    Answer:
                    """
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    # Setup the chain
                    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
                    
                    rag_chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Get response
                    answer = rag_chain.invoke(query)
                    
                    # Add query and response to chat history
                    st.session_state.chat_history.append({"query": query, "response": answer})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for i, exchange in enumerate(st.session_state.chat_history):
            st.info(f"Question {i+1}: {exchange['query']}")
            st.success(f"Answer {i+1}: {exchange['response']}")
else:
    st.info("Please upload documents to start the conversation.")

