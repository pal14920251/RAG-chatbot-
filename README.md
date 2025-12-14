# 🧠 LLM Chatbot RAG Assistant

A **Retrieval-Augmented Generation (RAG)** powered chatbot built with **Streamlit**, **LangChain**, and **Hugging Face models**.
It allows you to upload PDFs (like research papers, notes, or documents) and ask intelligent, context-aware questions.
The chatbot retrieves relevant sections and uses Hugging Face LLMs to generate precise answers.

---

## 🚀 Features

* 📂 Upload multiple **PDFs** for context
* 🔍 Retrieve top-*k* relevant chunks using **FAISS** + **Sentence Transformers**
* 💬 Interactive **Streamlit chat interface**
* 🧠 Response generation via **Hugging Face Inference API**
* 🔄 Support for both **text-generation** and **conversational** models

---

## 📁 Project Structure

```
RAG-chatbot/
│
├── app.py                # Streamlit web app
├── model.py              # Loads & queries Hugging Face LLM
├── rag_util.py           # Handles embeddings, FAISS, and PDF chunking
├── RAG.ipynb             # Optional notebook for step-by-step testing
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone the repository

```bash
git clone https://github.com/pal14920251/RAG-chatbot.git
cd RAG-chatbot
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the project root and add your **Hugging Face token**:

```bash
HF_API_TOKEN=your_huggingface_api_token
```

### 🧠 How to get a free Hugging Face token

You can use Hugging Face models **for free** (limited usage):

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **“New Token”** → choose **Read** access
3. Copy the token
4. Paste it inside your `.env` file as shown above

That’s it — no credit card or paid plan required.

---

## 🧠 Running the Chatbot

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal — typically:
👉 [http://localhost:8501](http://localhost:8501)

Upload one or more PDFs, then ask your question in the chat input!

---

## ⚙️ Model Configuration

You can easily switch models in **`app.py`** or **`model.py`**.
Example:

```python
# Example 1: Conversational model
model = ChatModel(model_id="deepseek-ai/DeepSeek-R1")

# Example 2: Text generation model
model = ChatModel(model_id="mistralai/Mistral-7B-Instruct-v0.2")
```

The code automatically detects whether to use `chat_completion` or `text_generation` mode based on the model type.

---

## 🧩 How It Works

1. **PDF Upload** → You upload one or more PDFs.
2. **Chunking & Embedding** → Each document is split into token-sized chunks and converted to embeddings using **Sentence Transformers**.
3. **Vector Search (FAISS)** → The most relevant chunks are retrieved based on your query.
4. **LLM Response** → Context + query are passed to a Hugging Face model for generation.


---

## 💬 Example Interaction

**User:**

> What is retrieval-augmented generation?

**Assistant:**

> Retrieval-Augmented Generation (RAG) combines information retrieval and large language models.
> It first searches for relevant context from uploaded documents and then uses that context to generate an informed response.

---

## 🧰 Dependencies
Dependencies are listed in requirements.txt
pip install -r requirements.txt

## ✨ Credits

* **Streamlit** → Chat UI
* **LangChain** → Document loading & text splitting
* **FAISS** → Fast vector search
* **Sentence Transformers** → Embeddings
* **Hugging Face Hub** → Model inference

---
