# 🧠 Agentic AI Systems
Tired of OpenAI API keys? Want to build real-world LLM applications without breaking the bank?

**This repo is for you.**

---

## 🔥 What Is This?

An end-to-end **Agentic AI system** that automates customer support for SaaS companies using:

- 🧩 **LangChain + Tools + Agents**
- 🧠 **HuggingFace LLMs** (No OpenAI Key Required!)
- 🔍 **RAG Pipeline** (Vector Search + Embeddings)
- ⚡ GPU Acceleration (CUDA support for blazing-fast inference)

It answers **real customer queries** by:
1. Retrieving relevant product reviews
2. Reasoning over the content using a local Falcon-7B model
3. Responding like a helpful, context-aware support agent

---

## 🧪 Why It Matters

💸 **Free to Run**  
→ Uses only open-source models and libraries. No credit card needed.

⚡ **Runs on Your GPU**  
→ Optimized for machines with CUDA. Use your local hardware to run Falcon-7B or similar.

🔎 **Solves a Real Business Problem**  
→ Automates tier-1 support with intelligence and context.

👨‍💻 **Built for Devs, by a Dev**  
→ Easy to clone, install, and customize. Comes with full explanations.

---

## 🧱 Tech Stack

| Tool         | Role                                    |
|--------------|-----------------------------------------|
| LangChain    | Agents, Tools, Chain Composition        |
| HuggingFace  | LLMs + Embedding Models                 |
| FAISS        | Fast Vector Search                      |
| PyTorch      | LLM Inference with GPU Support          |
| Pandas       | Dataset Handling                        |

---

## 👀 Example Queries

```text
❓ "Which ticketing tool has the best customer support?"
✅ Agent finds the answer from reviews and responds contextually

❓ "What are common complaints users have?"
✅ LLM finds patterns in the 'cons' section of reviews

❓ "Which tool scores highest in ease of use?"
✅ Accurate and fast retrieval + response
