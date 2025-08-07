# 🤖 LLM-Powered Customer Support Advisor Tool

A LangChain-based **Agentic AI system** that leverages **Retrieval-Augmented Generation (RAG)** and open-source LLMs to automate customer support for SaaS companies — all running locally with GPU acceleration using HuggingFace models.

---

## 📌 Problem Statement

SaaS companies receive hundreds of customer support tickets every day. Most are repetitive issues: FAQs, feature requests, pricing questions, and how-to queries.

> Manual handling by human agents leads to long response times and increased customer frustration.

---

## 💡 AI-Powered Solution

Build a smart **AI Agent** using:
- 📚 Vector-based retrieval of past reviews
- 🧠 Open-source LLM for answer generation
- 🧰 Tools + Agents for reasoning
- ⚡ GPU acceleration using CUDA-enabled HuggingFace transformers

---

## ⚙️ Project Features

- LangChain + FAISS-based **RAG pipeline**
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- Falcon-7B Instruct model for **text generation**
- Tool-wrapped LLM agent to answer real user queries
- Entirely runs **locally** — no OpenAI key needed!
- Utilizes **GPU** (if available)

---

## 🐍 Python Version

Tested on: **Python 3.10.0**

---

## 🏗️ Setup Instructions

### 1️⃣ Create Conda Environment

```bash
conda create -n support_ai python=3.10.0
conda activate support_ai
```

### 2️⃣ Install Libraries
pip install pandas
pip install langchain
pip install transformers
pip install torch
pip install faiss-cpu
pip install -U langchain-community
pip install sentence-transformers
pip install tf-keras

⚠️ If you have a CUDA-enabled GPU, install PyTorch with CUDA: 

* pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### 📂 Dataset
The project uses a real-world dataset of SaaS product reviews (capterra_reviews.csv) with:
* Overall Review
* Pros
* Cons
* Feature ratings
* Ticket system name etc.

🚀 How to Run
* Place capterra_reviews.csv in the project directory

1. Run the Python script:
> python customer_support_agent.py

2. Ask questions like:
- Which ticket system has the best customer service?
- What are the most common pros mentioned in the reviews?

> The LLM agent will fetch relevant data and generate a response using Falcon-7B.


🧠 Architecture Overview

             Reviews   
               │
        📄 Format into Documents
               │
      ✂️ Chunk with TextSplitter
               │
     📌 Embed with HuggingFace Embeddings
               │
    🔍 Store in FAISS Vector Database
               │
    🧠 LLM (Falcon-7B Instruct - HF Pipeline)
               │
    🛠️ Tool + Agent (Langchain)
               │
     🤖 Answer Customer Queries

👨‍💻 Author

Made by Raguwing Gudla– feel free to fork and customize!

# NOTE - For some reason the IPYNB file is showing error, please download it and have a look at it.
