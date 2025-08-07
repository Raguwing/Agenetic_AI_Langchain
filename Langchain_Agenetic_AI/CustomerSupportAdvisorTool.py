# import necessary libraries
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import os
import pandas as pd
import torch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

### Step-1: Data Preparation (Knowledge base)
customer_data = pd.read_csv(r"E:\Langchain_Agenetic_AI\Datasets\capterra_reviews.csv")
customer_data = customer_data.sample(n=1000, random_state=42)  # Reducing the data size for local testing
customer_data = customer_data.dropna(axis=0)

### Step-2: Convert each customer review into a text document

def format_customer_review(row):
    return f"""
### {row['ticket_system']} Review

**Title:** {row['title']}

**Overall Review:** {row['overall_text']}

**Pros:** {row['pros_text']}
**Cons:** {row['cons_text']}

**Overall Rating:** {row['overall_rating']} / 5
**Ease of Use:** {row['ease_of_use']} / 5
**Customer Service:** {row['customer_service']} / 5
**Features:** {row['features']} / 5
**Value for Money:** {row['value_for_money']} / 5
**Likelihood to Recommend:** {row['likelihood_to_recommend']} / 10
"""

docs = [Document(page_content=format_customer_review(row)) for _, row in customer_data.iterrows()]

print(f"Total number of documents created: {len(docs)}")
print(docs[0].page_content)  # Displaying the content of the first document to verify formatting

### Step-3: Splitting documents into small chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Total number of chunks created: {len(chunks)}")
print(chunks[0].page_content)  # Displaying the content of the first chunk

### Step 4: Embeddings & Vector data creation using- "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
print(f"Total number of vectors created: {len(vectorstore.index_to_docstore_id)}")

### Step-5: Loading LM from HuggingFace

llm_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct",
                        max_new_tokens=512,
                        device=device,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        temperature=0.2)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

### Step 6: Setting up Retrieval QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

### Step 7: Creating tool to be used by the Agent
tools = [
    Tool(
        name="Customer Ticket System QA",
        func=qa_chain.run,
        description="Use this tool to answer any customer question about the ticket system reviews"
    )
]

### Step-8: Creating the Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Step 9: Testing the Agent by Posing Questions
agent.run("What are the most common pros mentioned in the reviews?")