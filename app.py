import streamlit as st
import time 
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory

from langchain_classic.chains import ConversationalRetrievalChain

# --- CONFIGURATION ---
st.set_page_config(page_title="FinAgent: AI Analyst", page_icon="📈")
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ Error: GROQ_API_KEY not found. Check your .env file!")
    st.stop()

# --- FUNCTIONS ---

def stream_parser(text: str):
    """Generator function that yields chunks of text for the typewriter effect"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02) 


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Downloads a small, free embedding model (all-MiniLM-L6-v2) to my machine
    # This runs LOCALLY on my CPU (no API cost)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"), 
        model_name="llama-3.1-8b-instant", 
        temperature=0
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # limited to k=2 to prevent token overflow 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), 
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

# --- MAIN UI ---
def main():
    st.sidebar.title("📈 FinAgent")
    st.sidebar.info("Upload your Annual Report to chat with it.")
    
    # Upload
    pdf_docs = st.sidebar.file_uploader("Upload PDF", accept_multiple_files=True)
    
    # Process
    if st.sidebar.button("Process Data"):
        if not pdf_docs:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Reading PDF & Building Vectors..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("System Ready! Ask your questions.")

    # Chat Area
    st.header("Financial Insight Dashboard")
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    # Chat Input
    if prompt := st.chat_input("Ask a question (e.g., 'What are the risks?')"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if 'conversation' in st.session_state:
            with st.chat_message("assistant"):
               # Show a temporary "Thinking..." spinner using strealit - Version 0 but I want it to show partial streaming
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({'question': prompt})
                    answer = response['answer']
                    sources = response['source_documents']
                
                # Streaming the answer (The "Typewriter" Effect) - like Gemini or ChatGPT
                st.write_stream(stream_parser(answer))
                
                # Show Sources - Verison 0 (Still needs to be updated) 
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.info(doc.page_content[:300] + "...")
                        
            #Save to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("Please upload and process a PDF first.")

if __name__ == '__main__':

    main()
