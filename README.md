# FinAgent: Financial RAG Chatbot 🤖📊

A Retrieval-Augmented Generation (RAG) chatbot designed to analyze financial documents. This application allows users to chat with **Annual Reports** (or any PDF) to extract insights, summarize key metrics, and answer specific questions about a company's financial health.

## 🚀 Overview

This project solves the problem of manually sifting through dense financial PDFs. By leveraging an Open Source LLM (Llama 3) with RAG, the bot retrieves the exact context needed from the uploaded documents to answer questions, ensuring responses are grounded in the actual data provided.

### Key Features
- **Document Ingestion:** Instantly processes PDF Annual Reports uploaded via the secure sidebar.
- **Semantic Search:** Uses local vector embeddings (FAISS) to find relevant sections of the report based on user queries.
- **Context-Aware Answers:** Generates responses referencing specific details (Revenue, EBITDA, Risks) with **page citations**.
- **Interactive UI:** A streamlined web interface built with Streamlit for real-time Q&A.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Interface:** Streamlit
* **Orchestration:** LangChain
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **LLM (Inference):** Meta Llama 3.1 8B (via Groq LPU)

## 📂 Project Structure

```bash
Fin_ChatBot_RAG/
│
├── venv/                # Virtual Environment
├── app.py               # Main application logic (UI + RAG pipeline)
├── .env                 # API keys (strictly git-ignored)
└── README.md            # Project documentation
```

## ⚙️ Setup & Installation

Follow these steps to get the app running locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/amehnd/Fin_ChatBot_RAG.git](https://github.com/amehnd/Fin_ChatBot_RAG.git)
cd Fin_ChatBot_RAG
```

### 2. Create a Virtual Environment
It's best practice to isolate your dependencies.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages:

```bash
pip install streamlit langchain langchain-community langchain-groq langchain-text-splitters faiss-cpu pypdf python-dotenv sentence-transformers
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory to store your API keys.

```bash
# Windows (PowerShell)
new-item .env
# Mac/Linux
touch .env
```

Add your Groq API key inside the file (Get one for free at [console.groq.com](https://console.groq.com)):

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

## 🏃‍♂️ Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

The web app should open automatically in your browser (usually at `http://localhost:8501`).

1.  **Upload:** Drag and drop your `annual_report.pdf` into the sidebar.
2.  **Process:** Click the "Process" button and wait for the "System Ready" signal.
3.  **Chat:** Type your question in the input box (e.g., *"What are the top 3 risks mentioned?"*).
4.  **Verify:** Expand the "View Source Documents" dropdown to see the exact text chunks the AI used.

## 🧠 How It Works

1.  **Ingestion:** The PDF is parsed using `PyPDF2` to extract raw text.
2.  **Chunking:** Text is split into smaller, meaningful chunks (e.g., 1000 characters) to fit the LLM's context window.
3.  **Embedding:** Chunks are converted into vector representations using the `HuggingFace` model.
4.  **Indexing:** Vectors are stored in a local `FAISS` index for millisecond-latency retrieval.
5.  **Retrieval:** When you ask a question, the system searches for the most similar text chunks.
6.  **Generation:** Llama 3 receives your question + the retrieved chunks to formulate a factual answer.

## 🔮 Future Roadmap

This project is currently in the MVP phase. Planned engineering improvements include:

* **Multi-Modal Parsing:** Integrating **LlamaParse** or **Unstructured.io** to extract data from financial tables and charts, allowing the bot to "see" and interpret visual data in Annual Reports.
* **Private Deployment:** Adding an option to switch the inference backend from Groq (Cloud) to **Ollama** (Local) for air-gapped security requirements in sensitive financial environments.
* **Containerization:** Dockerizing the application for consistent deployment across dev and prod environments.
  
## 🤝 Contributing

Feel free to open issues or submit pull requests if you have ideas for improvements!
