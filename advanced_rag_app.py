import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import StreamlitCallbackHandler
import tempfile
import pandas as pd
import uuid

# Page configuration
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
    }
    .stProgress > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>🧠 Advanced RAG Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("Retrieve information from your documents and have conversations with AI about them")

# Initialize session states
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_source" not in st.session_state:
    st.session_state.document_source = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

# Sidebar configuration
with st.sidebar:
    st.markdown("<h2 class='sub-header'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    # API Configuration
    st.subheader("API Settings")
    api_option = st.radio("Select API Provider:", ["OpenAI", "Hugging Face Hub"])
    
    if api_option == "OpenAI":
        openai_api_key = st.text_input("OpenAI API Key:", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        model_name = st.selectbox(
            "Select Model:",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
    else:
        hf_api_key = st.text_input("Hugging Face API Key:", type="password")
        if hf_api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
        
        model_name = st.selectbox(
            "Select Model:",
            ["google/flan-t5-xxl", "tiiuae/falcon-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.1"],
            index=2
        )
    
    # Model parameters
    st.subheader("Model Parameters")
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    # Document processing parameters
    st.subheader("Document Processing")
    chunk_size = st.slider("Chunk Size:", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap:", min_value=0, max_value=500, value=200, step=50)
    
    # Embedding model selection
    st.subheader("Embedding Model")
    embedding_option = st.radio("Select Embedding Model:", ["OpenAI", "Sentence Transformers"])
    
    if embedding_option == "OpenAI":
        embedding_model_name = "text-embedding-ada-002"
    else:
        embedding_model_name = st.selectbox(
            "Select Sentence Transformer Model:",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            index=0
        )
    
    # Vector store selection
    vector_store_type = st.radio("Vector Store:", ["FAISS", "Chroma"])
    
    # Save configuration
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap
    st.session_state.embedding_option = embedding_option
    st.session_state.embedding_model_name = embedding_model_name
    st.session_state.vector_store_type = vector_store_type
    st.session_state.api_option = api_option

# Function to load embeddings
def load_embeddings():
    if st.session_state.embedding_option == "OpenAI":
        return OpenAIEmbeddings()
    else:
        return HuggingFaceEmbeddings(model_name=st.session_state.embedding_model_name)

# Function to load LLM
def load_llm():
    if st.session_state.api_option == "OpenAI":
        return ChatOpenAI(
            model_name=st.session_state.model_name,
            temperature=st.session_state.temperature,
            streaming=True
        )
    else:
        return HuggingFaceHub(
            repo_id=st.session_state.model_name,
            model_kwargs={"temperature": st.session_state.temperature}
        )

# Function to process documents
def process_documents(uploaded_files):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        documents = []
        file_details = []
        
        with st.status("Processing documents...") as status:
            # Save and process each uploaded file
            for i, uploaded_file in enumerate(uploaded_files):
                status.update(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Save file to temp directory
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process file based on type
                try:
                    if file_path.lower().endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        file_details.append(f"📄 {uploaded_file.name}: {len(docs)} pages")
                    elif file_path.lower().endswith(".txt"):
                        loader = TextLoader(file_path)
                        docs = loader.load()
                        file_details.append(f"📝 {uploaded_file.name}: Text document")
                    elif file_path.lower().endswith(".csv"):
                        loader = CSVLoader(file_path)
                        docs = loader.load()
                        file_details.append(f"📊 {uploaded_file.name}: CSV data")
                    else:
                        st.error(f"Unsupported file type: {uploaded_file.name}")
                        continue
                        
                    # Add metadata about the source
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    
                    documents.extend(docs)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            # Split documents into chunks
            status.update("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            status.update("Creating embeddings and vector store...")
            embeddings = load_embeddings()
            
            # Create appropriate vector store
            if st.session_state.vector_store_type == "FAISS":
                vector_store = FAISS.from_documents(chunks, embeddings)
                persistence_type = "In-memory (FAISS)"
            else:
                # Create a persistent path for Chroma
                db_path = os.path.join(temp_dir, "chroma_db")
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=db_path
                )
                persistence_type = "Disk-based (Chroma)"
            
            # Create conversation chain
            status.update("Setting up the retrieval chain...")
            llm = load_llm()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
                memory=memory,
                return_source_documents=True
            )
            
            # Update session state
            st.session_state.conversation = conversation_chain
            st.session_state.processed_docs = True
            st.session_state.vector_store = vector_store
            st.session_state.document_source = file_details
            st.session_state.embedding_model = st.session_state.embedding_model_name
            
            status.update(label="✅ Processing complete!", state="complete")
            
            return chunks, file_details, persistence_type

# Main area tabs
tab1, tab2, tab3 = st.tabs(["Document Upload", "Chat Interface", "System Information"])

# Tab 1: Document Upload
with tab1:
    st.markdown("<h2 class='sub-header'>📁 Upload Documents</h2>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, TXT, CSV)",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"]
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Process Documents", type="primary", disabled=not uploaded_files):
            if not (st.session_state.api_option == "OpenAI" and "OPENAI_API_KEY" not in os.environ) and not (st.session_state.api_option == "Hugging Face Hub" and "HUGGINGFACEHUB_API_TOKEN" not in os.environ):
                chunks, file_details, persistence_type = process_documents(uploaded_files)
                
                # Display processing results
                st.markdown("<h3 class='sub-header'>Processing Results</h3>", unsafe_allow_html=True)
                st.markdown(f"✅ **Total chunks created:** {len(chunks)}")
                st.markdown(f"✅ **Embedding model:** {st.session_state.embedding_model_name}")
                st.markdown(f"✅ **Vector store:** {persistence_type}")
                
                # Display file details
                st.markdown("<h3 class='sub-header'>Files Processed</h3>", unsafe_allow_html=True)
                for detail in file_details:
                    st.markdown(f"- {detail}")
            else:
                st.error("Please enter your API key in the sidebar.")
    
    with col2:
        if st.button("Reset", type="secondary"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.processed_docs = False
            st.session_state.vector_store = None
            st.session_state.document_source = None
            st.session_state.embedding_model = None
            st.experimental_rerun()

# Tab 2: Chat Interface
with tab2:
    st.markdown("<h2 class='sub-header'>💬 Ask Questions About Your Documents</h2>", unsafe_allow_html=True)
    
    if not st.session_state.processed_docs:
        st.info("Please upload and process documents in the Document Upload tab first.")
    else:
        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                with st.chat_message("user"):
                    st.write(message.content)
            else:  # AI message
                with st.chat_message("assistant"):
                    st.write(message.content)
                    
                    # If this is an AI response and we have source documents, show them
                    if hasattr(message, 'source_documents') and message.source_documents:
                        with st.expander("View sources"):
                            for j, doc in enumerate(message.source_documents):
                                st.markdown(f"**Source {j+1}:** {doc.metadata.get('source', 'Unknown')}")
                                st.markdown(f"**Content:** {doc.page_content[:200]}...")
        
        # Handle new user question
        if user_question:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate AI response
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.conversation(
                            {"question": user_question},
                            callbacks=[st_callback]
                        )
                        answer = response["answer"]
                        source_docs = response.get("source_documents", [])
                        
                        # Store the response with sources
                        from langchain.schema.messages import AIMessage
                        ai_message = AIMessage(content=answer)
                        if source_docs:
                            ai_message.source_documents = source_docs
                        
                        # Add to chat history (first user message, then AI response)
                        from langchain.schema.messages import HumanMessage
                        st.session_state.chat_history.append(HumanMessage(content=user_question))
                        st.session_state.chat_history.append(ai_message)
                        
                        # Show the response
                        st.write(answer)
                        
                        # Show sources if available
                        if source_docs:
                            with st.expander("View sources"):
                                for j, doc in enumerate(source_docs):
                                    st.markdown(f"**Source {j+1}:** {doc.metadata.get('source', 'Unknown')}")
                                    st.markdown(f"**Content:** {doc.page_content[:200]}...")
                    
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

# Tab 3: System Information
with tab3:
    st.markdown("<h2 class='sub-header'>ℹ️ System Information</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='sub-header'>Configuration</h3>", unsafe_allow_html=True)
        
        config_info = f"""
        | Parameter | Value |
        | --- | --- |
        | API Provider | {st.session_state.get('api_option', 'Not set')} |
        | Model | {st.session_state.get('model_name', 'Not set')} |
        | Temperature | {st.session_state.get('temperature', 'Not set')} |
        | Chunk Size | {st.session_state.get('chunk_size', 'Not set')} |
        | Chunk Overlap | {st.session_state.get('chunk_overlap', 'Not set')} |
        | Embedding Model | {st.session_state.get('embedding_model_name', 'Not set')} |
        | Vector Store | {st.session_state.get('vector_store_type', 'Not set')} |
        | Session ID | {st.session_state.session_id} |
        """
        
        st.markdown(config_info)
    
    with col2:
        st.markdown("<h3 class='sub-header'>Document Status</h3>", unsafe_allow_html=True)
        
        if st.session_state.processed_docs and st.session_state.document_source:
            st.markdown("**Files processed:**")
            for detail in st.session_state.document_source:
                st.markdown(f"- {detail}")
        else:
            st.markdown("No documents processed yet.")
    
    # Show technical explanation of RAG
    with st.expander("How RAG Works"):
        st.markdown("""
        ### Retrieval-Augmented Generation (RAG) Explained
        
        RAG combines the power of retrieval systems with generative language models. Here's how it works:
        
        1. **Document Processing**: Documents are split into chunks and converted into vector embeddings
        2. **Retrieval**: When you ask a question, the system:
           - Converts your question into an embedding
           - Finds the most similar document chunks in the vector store
           - Retrieves the relevant information
        3. **Generation**: The language model receives:
           - Your question
           - The retrieved document chunks as context
           - Previous conversation history
           - Then generates a response using all this information
        
        This approach results in more accurate, factual responses that are grounded in your documents.
        """)
    
    # Show examples
    with st.expander("Example Questions"):
        st.markdown("""
        ### Try asking:
        
        - "What are the main themes discussed in the documents?"
        - "Can you summarize the information about [specific topic]?"
        - "What does the document say about [specific term or concept]?"
        - "Are there any contradictions in the information provided?"
        - "What are the key findings or conclusions in the documents?"
        """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using LangChain and Streamlit")