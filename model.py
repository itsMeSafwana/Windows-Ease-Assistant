import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

class ChatbotModel:
    def __init__(self, data_path):
        # Load dataset
        self.df = pd.read_csv(data_path)
        self.queries = self.df['Query'].tolist()
        self.responses = self.df['Response'].tolist()

        # Load pre-trained model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Compute embeddings
        self.embeddings = self.model.encode(self.queries, show_progress_bar=True)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def get_response(self, query, top_k=1):
        # Encode user query
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve response
        response = self.responses[indices[0][0]]
        return response
