import pinecone
import cohere
import pandas as pd
import numpy as np

def preprocess_text(text):
    # Implement your text cleaning logic here (e.g., remove stop words, lowercase)
    return text.lower()

def query_rag_model(query, pc, cohere_api):
    try:
        query_embedding = cohere_api.embed(query, model="command-large").embeddings[0]
        results = pc.query(
            index_name="my-chatbot-index",  # Replace with your desired index name
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

# Set up Pinecone connection
pc = pinecone.Pinecone(api_key="e01e970f-2699-4366-8d9b-158e01fcbe38", environment="us-west1-gcp")

# Create a Pinecone index (optional, modify if needed)
index_name = "my-chatbot-index"  # Replace with your desired index name
if index_name not in pc.list_indexes().names():
    dimension = 768  # Adjust based on your embedding model
    metric = "cosine"
    pc.create_index(
        index_name,
        dimension=dimension,
        metric=metric,
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")  # Replace with your desired cloud and region
    )

# Load your dataset
data = pd.read_csv("Example_dataset.csv")

# Preprocess data (e.g., clean text, remove stop words)
data['text'] = data['text'].apply(lambda x: preprocess_text(x))

# Generate embeddings using Cohere
cohere_api = cohere.Client("KagfbFbaBRA9hnOVEBvnH7yvoj0xyrTsoJJcvAJs")

embeddings = []


for text in data['text']:
    embedding = cohere_api.embed(text, model="command-large")
    embeddings.append(embedding.embeddings[0])

# response = cohere_api.embed(
#     texts=data['text'], model="embed-english-v3.0", input_type="classification"
# )

# print(response)

# embeddings = response["embeddings"]

# Upsert embeddings to Pinecone
vectors = [
    pinecone.Vector(id=str(i), values=embedding)
    for i, embedding in enumerate(embeddings)
]
pc.upsert(index_name, vectors)

# Test the model with some example queries
queries = [
    "What is the capital of France?",
    "Who wrote the play Hamlet?",
    "When did World War II start?",
]

for query in queries:
    response = query_rag_model(query, pc, cohere_api)
    print(f"Query: {query}")
    print(f"Response: {response}")

