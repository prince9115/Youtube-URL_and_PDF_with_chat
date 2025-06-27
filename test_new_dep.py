import sys 
import asyncio
# Fix Playwright compatibility on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import tempfile
import validators
from urllib.parse import urlparse, parse_qs
import os

# LangChain imports for both functionalities
from langchain_community.document_loaders import YoutubeLoader, PlaywrightURLLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Audio processing imports
import openai
from youtube_transcript_api import YouTubeTranscriptApi
import re

import subprocess

try:
    subprocess.run(["playwright", "install", "chromium"], check=True)
except Exception as e:
    print("Playwright installation may have already been done or failed:", e)

import gc
import psutil

# Add this function to clean up memory
def cleanup_memory():
    """Clean up memory and temporary files"""
    gc.collect()
    
    # Clear large session state items if needed
    if 'vectordb' in st.session_state and st.session_state.vectordb:
        del st.session_state.vectordb
        st.session_state.vectordb = None
    
    gc.collect()

def show_memory_usage():
    """Display current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    st.sidebar.metric("Memory Usage", f"{memory_mb:.1f} MB")

# Add cleanup button in sidebar
if st.sidebar.button("🧹 Clear Memory"):
    cleanup_memory()
    st.sidebar.success("Memory cleared!")
    st.rerun()

# Page config
st.set_page_config(
    page_title="Assistant Summarizer",
    layout="wide"
)

st.title("Assistant Summarizer- PDF Chat & YouTube/Web Summarizer")

# --- YouTube/Web Functions ---
def clean_youtube_url(url):
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return f"https://www.youtube.com/watch?v={parsed.path.lstrip('/')}"
    elif "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        return f"https://www.youtube.com/watch?v={query.get('v', [''])[0]}"
    return url

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id, language='en'):
    """Get YouTube transcript using youtube-transcript-api"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        transcript = ' '.join([item['text'] for item in transcript_list])
        return transcript
    except Exception as e:
        raise Exception(f"Could not retrieve transcript: {str(e)}")

def download_audio_for_whisper(youtube_url):
    """Download audio for OpenAI Whisper API"""
    import tempfile
    import yt_dlp
    
    tmpdir = tempfile.mkdtemp()
    audio_path = os.path.join(tmpdir, "audio.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded_file = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
    
    return downloaded_file

def openai_whisper_transcribe(audio_file, api_key):
    """Transcribe audio using OpenAI Whisper API"""
    client = openai.OpenAI(api_key=api_key)
    
    with open(audio_file, 'rb') as audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    
    return transcript.text

# --- PDF Chat Functions ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create session history"""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def initialize_embeddings(api_key):
    """Initialize Cohere embeddings"""
    try:
        embeddings = CohereEmbeddings(
            cohere_api_key=api_key,
            model="embed-english-light-v3.0"  
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def setup_rag_chain(llm, retriever):
    """Setup the RAG chain with history"""
    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create history-aware retriever
    history_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # QA prompt
    system_prompt = (
        "You are a rude ai assistant who gives all answers in as rude way as possible."
        "Use the following pieces of retrieved context to answer in detailed and pointwise manner "
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create chains
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_retriever, qa_chain)
    
    # Create conversation chain with history
    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversation_rag_chain

# --- Sidebar Configuration ---
st.sidebar.title("🔧 Configuration")

# API Keys
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
cohere_api_key = st.sidebar.text_input("Cohere API Key (for PDF Chat)", type="password")

# Model settings
if groq_api_key:
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.6)
    
    # Different model options for different features
    st.sidebar.subheader("Model Selection")
    summarizer_model = st.sidebar.selectbox(
        "YouTube/Web Summarizer Model:", 
        ["meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen-qwq-32b", "mistral-saba-24b"]
    )
    
    pdf_chat_model = st.sidebar.selectbox(
        "PDF Chat Model:", 
        ["meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct", "qwen-qwq-32b", "mistral-saba-24b"]
    )

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'conversation_rag_chain' not in st.session_state:
    st.session_state.conversation_rag_chain = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None

# --- Main Interface ---
st.markdown("---")
st.subheader("Choose Your Assistant Mode")


col1, col2 = st.columns(2)

with col1:
    if st.button("PDF Chat Assistant", use_container_width=True, type="primary"):
        st.session_state.current_mode = "pdf_chat"

with col2:
    if st.button("YouTube/Web Summarizer", use_container_width=True, type="primary"):
        st.session_state.current_mode = "youtube_web"

st.markdown("---")

# --- PDF Chat Mode ---
if st.session_state.current_mode == "pdf_chat":
    st.header("PDF Assistant - Chat & Summarize")
    
    if not groq_api_key:
        st.warning("Please enter Groq API key in the sidebar to use PDF features.")
    else:
        # Add sub-mode selection for PDF features
        st.subheader("Choose PDF Operation")
        pdf_col1, pdf_col2 = st.columns(2)
        
        with pdf_col1:
            if st.button("Chat with PDF", use_container_width=True):
                st.session_state.pdf_mode = "chat"
        
        with pdf_col2:
            if st.button("Summarize PDF", use_container_width=True):
                st.session_state.pdf_mode = "summarize"
        
        # Initialize pdf_mode if not exists
        if 'pdf_mode' not in st.session_state:
            st.session_state.pdf_mode = None
        
        # PDF Chat Mode
        if st.session_state.pdf_mode == "chat":
            st.markdown("### Chat with Your PDF Documents")
            
            if not cohere_api_key:
                st.warning("Please enter Cohere API key in the sidebar to use PDF Chat.")
            else:
                # Session ID for PDF chat
                session_id = st.text_input("Session ID", value="Session_1")
                
                # Initialize LLM for PDF chat
                llm = ChatGroq(
                    model=pdf_chat_model,
                    api_key=groq_api_key,
                    temperature=temperature
                )
                
                # Initialize embeddings
                embeddings = initialize_embeddings(cohere_api_key)
                
                if embeddings:
                    st.success("Embeddings initialized successfully!")
                    
                    # File upload
                    uploaded_files = st.file_uploader(
                        "Choose PDF files", 
                        type="pdf", 
                        accept_multiple_files=True,
                        help="Upload one or more PDF files to chat with their content"
                    )
                    
                    if uploaded_files:
                        # Process PDFs
                        with st.spinner("Processing PDFs..."):
                            documents = []
                            
                            for uploaded_file in uploaded_files:
                                # Create temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name
                                
                                try:
                                    # Load and process PDF
                                    loader = PyPDFLoader(tmp_path)
                                    docs = loader.load()
                                    documents.extend(docs)
                                    
                                    st.success(f"Processed: {uploaded_file.name}")
                                except Exception as e:
                                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                                finally:
                                    # Clean up temporary file
                                    os.unlink(tmp_path)
                            
                            if documents:
                                try:
                                    # Split documents
                                    splitter = RecursiveCharacterTextSplitter(
                                        chunk_size=1500,
                                        chunk_overlap=200
                                    )
                                    splits = splitter.split_documents(documents)
                                    
                                    # Create vector database
                                    st.session_state.vectordb = FAISS.from_documents(
                                        documents=splits,
                                        embedding=embeddings
                                    )
                                    
                                    # Create retriever
                                    retriever = st.session_state.vectordb.as_retriever(
                                        search_kwargs={"k": 3}
                                    )
                                    
                                    # Setup RAG chain
                                    st.session_state.conversation_rag_chain = setup_rag_chain(llm, retriever)
                                    
                                    st.success(f"Vector database created with {len(splits)} chunks!")
                                    
                                except Exception as e:
                                    st.error(f"Error creating vector database: {str(e)}")
                    
                    # Chat interface
                    if st.session_state.conversation_rag_chain:
                        st.subheader("Chat with your documents")
                        
                        # Display chat history
                        if session_id in st.session_state.store:
                            chat_history = st.session_state.store[session_id]
                            for message in chat_history.messages:
                                if hasattr(message, 'content'):
                                    if message.type == "human":
                                        st.chat_message("user").write(message.content)
                                    else:
                                        st.chat_message("assistant").write(message.content)
                        
                        # Chat input
                        user_input = st.chat_input("Ask a question about your documents...")
                        
                        if user_input:
                            # Display user message
                            st.chat_message("user").write(user_input)
                            
                            try:
                                # Get response
                                with st.spinner("Thinking..."):
                                    response = st.session_state.conversation_rag_chain.invoke(
                                        {"input": user_input},
                                        config={"configurable": {"session_id": session_id}}
                                    )
                                
                                # Display assistant response
                                st.chat_message("assistant").write(response['answer'])
                                
                            except Exception as e:
                                st.error(f"Error getting response: {str(e)}")
                    
                    else:
                        st.info("Please upload PDF files to start chatting!")
                
                else:
                    st.error("Failed to initialize embeddings. Please check your Cohere API key.")
        
        # PDF Summarization Mode
        elif st.session_state.pdf_mode == "summarize":
            st.markdown("### Summarize Large PDF Documents")
            st.info("Perfect for books, research papers, and large documents.")
            
            # Initialize LLM for PDF summarization
            llm = ChatGroq(
                model=pdf_chat_model,
                api_key=groq_api_key,
                temperature=temperature
            )
            
            # File upload for summarization
            uploaded_file = st.file_uploader(
                "Choose a PDF file to summarize", 
                type="pdf", 
                help="Upload a PDF file to get a comprehensive summary"
            )
            
            if uploaded_file:
                # Summarization options
                st.subheader("Summarization Settings")
                
                col1, col2 = st.columns(2)
                with col1:
                    summary_type = st.selectbox(
                        "Summary Type:",
                        ["Comprehensive", "Key Points", "Executive Summary", "Chapter-wise"]
                    )
                
                with col2:
                    chunk_size = st.selectbox(
                        "Chunk Size:",
                        [1000, 1500, 2000, 3000],
                        index=1,
                        help="Larger chunks for better context, smaller for more detailed processing"
                    )
                
                if st.button("Generate Summary", use_container_width=True, type="primary"):
                    try:
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Load PDF
                        with st.spinner(f"Loading PDF: {uploaded_file.name}..."):
                            loader = PyPDFLoader(tmp_path)
                            documents = loader.load()
                            
                        st.success(f"Loaded {len(documents)} pages")
                        
                        # Split documents for processing
                        with st.spinner("Processing document chunks..."):
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=200,
                                length_function=len
                            )
                            splits = splitter.split_documents(documents)
                            
                        st.info(f"Created {len(splits)} chunks for processing")
                        
                        # Create prompts based on summary type
                        if summary_type == "Comprehensive":
                            map_prompt = PromptTemplate(
                                input_variables=["text"],
                                template="""
                                Provide a detailed summary of the following text section. 
                                Include key concepts, important details, and main arguments:
                                
                                {text}
                                
                                DETAILED SUMMARY:
                                """
                            )
                            combine_prompt = PromptTemplate(
                                input_variables=["text"],
                                template="""
                                Create a comprehensive summary from the following section summaries.
                                Organize the content logically and ensure all important information is included:
                                
                                {text}
                                
                                COMPREHENSIVE SUMMARY:
                                """
                            )
                        
                        elif summary_type == "Key Points":
                            map_prompt = PromptTemplate(
                                input_variables=["text"],
                                template="""
                                Extract the key points and main ideas from the following text:
                                
                                {text}
                                
                                KEY POINTS (in bullet format):
                                """
                            )
                            combine_prompt = PromptTemplate(
                                input_variables=["text"],
                                template="""
                                Combine and organize the following key points into a structured summary:
                                
                                {text}
                                
                                ORGANIZED KEY POINTS:
                                """
                            )
                        
                        elif summary_type == "Executive Summary":
                            map_prompt = PromptTemplate(
                                input_variables=["text"],
                                template="""
                                Create a concise executive summary focusing on main conclusions and actionable insights:
                                
                                {text}
                                
                                EXECUTIVE SUMMARY SECTION:
                                """
                            )
                            combine_prompt = PromptTemplate(
                                input_variables=["text"],
                                template="""
                                Create a polished executive summary from the following sections:
                                
                                {text}
                                
                                FINAL EXECUTIVE SUMMARY:
                                """
                            )
                        
                        else:  # Chapter-wise
                            map_prompt = PromptTemplate(
                                input_variables=["text"],
                                template="""
                                Summarize this section as if it were a chapter, including main topics and conclusions:
                                
                                {text}
                                
                                CHAPTER SUMMARY:
                                """
                            )
                            combine_prompt = PromptTemplate(
                                input_variables=["text"],
                                template="""
                                Organize the following chapter summaries into a coherent document summary:
                                
                                {text}
                                
                                COMPLETE DOCUMENT SUMMARY:
                                """
                            )
                        
                        # Create and run summarization chain
                        with st.spinner(f"Generating {summary_type.lower()} summary..."):
                            chain = load_summarize_chain(
                                llm=llm,
                                chain_type="map_reduce",
                                map_prompt=map_prompt,
                                combine_prompt=combine_prompt,
                                verbose=False
                            )
                            
                            summary = chain.run(splits)
                        
                        # Display results
                        st.success("Summary Generated Successfully!")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pages Processed", len(documents))
                        with col2:
                            st.metric("Chunks Created", len(splits))
                        with col3:
                            st.metric("Summary Type", summary_type)
                        
                        st.markdown("---")
                        st.markdown(f"### {summary_type} Summary:")
                        st.markdown(summary)
                        
                        # Download option
                        st.download_button(
                            label="Download Summary",
                            data=summary,
                            file_name=f"{uploaded_file.name}_{summary_type}_summary.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        st.exception(e)
                    finally:
                        # Clean up temporary file
                        if 'tmp_path' in locals():
                            os.unlink(tmp_path)
        
        else:
            st.info("Please choose an operation: Chat with PDF or Summarize PDF")

# --- YouTube/Web Summarizer Mode ---
elif st.session_state.current_mode == "youtube_web":
    st.header("📺 YouTube & Website Summarizer")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to use the summarizer.")
    else:
        # URL input
        url = st.text_input("Enter YouTube or Website URL")
        
        # YouTube transcription method selection
        if url and ("youtube.com" in url or "youtu.be" in url):
            st.subheader("🎬 YouTube Transcription Method")
            transcription_method = st.selectbox(
                "Choose transcription method:",
                ["YouTube Captions (Free)", "OpenAI Whisper (Paid)"],
                help="YouTube Captions are free but may not be available for all videos. OpenAI Whisper is more reliable but requires an API key."
            )
            
            # OpenAI API key input if Whisper is selected
            openai_api_key = None
            if transcription_method == "OpenAI Whisper (Paid)":
                openai_api_key = st.text_input(
                    "OpenAI API Key", 
                    type="password",
                    help="Required for OpenAI Whisper transcription"
                )
                if not openai_api_key:
                    st.warning("Please enter your OpenAI API key to use Whisper transcription.")
        
        if st.button("Summarize", use_container_width=True):
            if not validators.url(url):
                st.error("Invalid URL")
                st.stop()

            try:
                # Initialize LLM for summarization
                model = ChatGroq(api_key=groq_api_key, model=summarizer_model)
                
                if "youtube.com" in url or "youtu.be" in url:
                    st.info("🎬 Processing YouTube URL...")
                    clean_url = clean_youtube_url(url)
                    video_id = extract_video_id(clean_url)
                    
                    if not video_id:
                        st.error("Could not extract video ID from URL")
                        st.stop()
                    
                    transcript = None
                    
                    if transcription_method == "YouTube Captions (Free)":
                        try:
                            with st.spinner("Getting YouTube captions..."):
                                transcript = get_youtube_transcript(video_id)
                            st.success("✅ Successfully retrieved YouTube captions")
                        except Exception as e:
                            st.error(f"❌ Could not get YouTube captions: {str(e)}")
                            st.info("💡 Try using OpenAI Whisper instead, or check if the video has captions available")
                            st.stop()
                    
                    elif transcription_method == "OpenAI Whisper (Paid)":
                        if not openai_api_key:
                            st.error("OpenAI API key is required for Whisper transcription")
                            st.stop()
                        
                        try:
                            with st.spinner("Downloading audio for transcription..."):
                                audio_file = download_audio_for_whisper(clean_url)
                            
                            with st.spinner("Transcribing with OpenAI Whisper..."):
                                transcript = openai_whisper_transcribe(audio_file, openai_api_key)
                            
                            # Clean up audio file
                            try:
                                os.unlink(audio_file)
                                # Also try to remove the directory if it's empty
                                audio_dir = os.path.dirname(audio_file)
                                if os.path.exists(audio_dir) and not os.listdir(audio_dir):
                                    os.rmdir(audio_dir)
                            except:
                                pass
                            
                            st.success("✅ Successfully transcribed with OpenAI Whisper")
                        except Exception as e:
                            st.error(f"❌ Whisper transcription failed: {str(e)}")
                            st.stop()
                    
                    if transcript:
                        docs = [Document(page_content=transcript)]
                    else:
                        st.error("No transcript could be obtained")
                        st.stop()
                
                else:
                    st.info("🌐 Scraping website content...")
                    loader = PlaywrightURLLoader(urls=[url], remove_selectors=["header", "footer", "nav"])
                    docs = loader.load()

                if not docs or not docs[0].page_content.strip():
                    st.warning("No content could be loaded from the URL.")
                    st.stop()

                # Show content preview
                content_preview = docs[0].page_content[:500] + "..." if len(docs[0].page_content) > 500 else docs[0].page_content
                with st.expander("📄 Content Preview"):
                    st.text(content_preview)

                # Create summarization chain
                map_prompt = PromptTemplate(
                    input_variables=["text"],
                    template="Summarize the following in detailed and bullet pointwise manner:\n\n{text}"
                )
                combine_prompt = PromptTemplate(
                    input_variables=["text"],
                    template="Combine the bullet points into a comprehensive summary with detailed pointwise format:\n\n{text}"
                )
                chain = load_summarize_chain(
                    llm=model,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt
                )

                with st.spinner("🤔 Generating summary..."):
                    summary = chain.run(docs)
                
                st.success("✅ Summary Generated!")
                
                # Display summary with better formatting
                st.markdown("### 📋 Summary:")
                st.markdown("---")
                st.markdown(summary)
                
                # Add download button for summary
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary,
                    file_name=f"summary_{url.split('/')[-1]}.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# --- Instructions ---
if st.session_state.current_mode is None:
    st.info("""
    ## Welcome to Assistant Summarizer!
    
    **Choose your preferred mode:**
    
    ### PDF Assistant
    - **Chat Mode**: Upload multiple PDFs and ask questions (requires Cohere API)
    - **Summarize Mode**: Get comprehensive summaries of large documents like books
    - Multiple summary types: Comprehensive, Key Points, Executive, Chapter-wise
    
    ### YouTube/Web Summarizer
    - Summarize YouTube videos (with captions or Whisper transcription)
    - Summarize website content
    - Get detailed bullet-point summaries
    - Requires only Groq API key
    
    **Getting Started:**
    1. Enter your API keys in the sidebar
    2. Click on your preferred mode button above
    3. Follow the instructions for your chosen mode
    """)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Assistant Summarizer - Powered by Prince Patel</p>
    </div>
    """, 
    unsafe_allow_html=True
)