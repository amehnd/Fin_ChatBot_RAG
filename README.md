# Fin_ChatBot_RAG 🤖📊

A Retrieval-Augmented Generation (RAG) chatbot designed to analyze financial documents. This application allows users to chat with an **Annual Report** (or any PDF) to extract insights, summarize key metrics, and answer specific questions about the company's financial health.

## 🚀 Overview

This project solves the problem of manually sifting through dense financial PDFs. By leveraging an LLM with RAG, the bot retrieves the exact context needed from the `annual_report.pdf` to answer questions, ensuring responses are grounded in the actual data provided.

### Key Features
- **Document Ingestion:** Automatically loads and processes the `annual_report.pdf`.
- **Semantic Search:** Uses vector embeddings to find relevant sections of the report based on user queries.
- **Context-Aware Answers:** Generates responses referencing specific details from the text (Revenue, EBITDA, Risks, etc.).
- **Interactive UI:** Simple web interface (via Streamlit) for real-time Q&A.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Interface:** Streamlit
* **Orchestration:** LangChain
* **Vector Store:** FAISS / ChromaDB (Local vector storage)
* **Embeddings:** OpenAI Embeddings / HuggingFace
* **LLM:** OpenAI GPT-3.5/4 / Llama 2

## 📂 Project Structure

```bash
Fin_ChatBot_RAG/
│
├── annual_report.pdf    # The knowledge base (Financial Report)
├── app.py               # Main application logic (UI + RAG pipeline)
├── .env                 # API keys (not committed)
├── requirements.txt     # Python dependencies
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
If a `requirements.txt` is missing, you can install the standard stack:

```bash
pip install streamlit langchain openai faiss-cpu pypdf python-dotenv tiktoken
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory to store your API keys.

```bash
touch .env
```

Add your OpenAI API key (or other model provider keys) inside the file:

```text
OPENAI_API_KEY=sk-your-api-key-here
```

## 🏃‍♂️ Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

The web app should open automatically in your browser (usually at `http://localhost:8501`).

1.  The app will load `annual_report.pdf` on startup.
2.  Type your question in the chat input (e.g., *"What was the total revenue for 2023?"*).
3.  The bot will retrieve relevant chunks and generate an answer.

## 🧠 How It Works

1.  **Load:** The PDF is loaded using `PyPDFLoader`.
2.  **Split:** Text is split into smaller chunks (e.g., 1000 characters) to fit into the context window.
3.  **Embed:** Chunks are converted into vector embeddings.
4.  **Store:** Embeddings are stored in a local Vector Database.
5.  **Retrieve:** User queries are embedded, and the DB searches for the most similar text chunks.
6.  **Generate:** The LLM receives the question + relevant chunks to formulate an answer.

## 🤝 Contributing

Feel free to open issues or submit pull requests if you have ideas for improvements!
