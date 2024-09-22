import streamlit as st
import pinecone
import cohere
import pandas as pd
import numpy as np
import pdf2text

# Set up Pinecone connection
pc = pinecone.Pinecone(api_key="e01e970f-2699-4366-8d9b-158e01fcbe38", environment="us-west1-gcp")

# Create a Pinecone index (optional, modify if needed)
index_name = "your_index_name"
if index_name not in pc.list_indexes().names():
    dimension = 768  # Adjust based on your embedding model
    metric = "cosine"
    pc.create_index(
        index_name,
        dimension=dimension,
        metric=metric,
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-East-1")  # Replace with your desired cloud and region
    )

# Load your dataset (if applicable)
data = pd.read_csv("Example_dataset.csv")

# Preprocess data (if applicable)
data['text'] = data['text'].apply(lambda x: preprocess_text(x))

# Generate embeddings using Cohere
cohere_api = cohere.Client("KagfbFbaBRA9hnOVEBvnH7yvoj0xyrTsoJJcvAJs")

embeddings = []
for text in data['text']:
    embedding = cohere_api.embed(text, model="command-large")
    embeddings.append(embedding.embeddings[0])

# Upsert embeddings to Pinecone
vectors = [
    pinecone.Vector(id=str(i), values=embedding)
    for i, embedding in enumerate(embeddings)
]
pc.upsert(index_name, vectors)

def query_rag_model(query, pc, cohere_api):
    try:
        query_embedding = cohere_api.embed(query, model="command-large").embeddings[0]
        results = pc.query(
            index_name="Temp_index",  # Replace with your desired index name
            query_embedding=query_embedding,
            top_k=5,
            include_metadata=True
        )

        # Retrieve relevant context from the retrieved documents
        context = " ".join([result["metadata"]["text"] for result in results["matches"]])

        # Generate a response using Cohere's generate API
        response = cohere_api.generate(
            prompt=f"Answer the question:\n{query}\nContext:\n{context}",
            model="command-large",
            max_tokens=200,
        )
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request."

def upload_and_process_pdf(pdf_file):
    with open(pdf_file, 'rb') as f:
        text = pdf2text.parse(f, raw=True)
    embedding = cohere_api.embed(text, model="command-large")
    vector = pinecone.Vector(id=str(len(pc.list_vectors(index_name="Temp_index"))), values=embedding)
    pc.upsert(index_name="Temp_index", vectors=[vector])

def main():
    st.title("Interactive QA Bot")

    uploaded_file = st.file_uploader("Upload a PDF file")

    if uploaded_file is not None:
        pdf_file = uploaded_file.name
        st.text(f"Processing PDF: {pdf_file}")
        upload_and_process_pdf(pdf_file)

    query = st.text_input("Ask your question:")

    if st.button("Submit"):
        if query:
            response = query_rag_model(query, pc, cohere_api)
            st.text(f"Response: {response}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
