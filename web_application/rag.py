import time
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from model import EmbeddingModel

class RAG:
    def __init__(self, api_key="", proxy_url=""):
        """Initialize the RAG class with the given API key and proxy URL."""
        self.embedding_model = EmbeddingModel()
        self.rag_db = self.initialize_rag_db(api_key, proxy_url)
    
    def initialize_rag_db(self, api_key, proxy_url):
        """Initialize the RAG database with Pinecone.

        Args:
            api_key (str): The API key for Pinecone.
            proxy_url (str): The proxy URL for Pinecone.

        Returns:
            Pinecone.Index: The initialized Pinecone index.
        """
        pc = Pinecone(api_key=api_key)
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        index_name = 'careconnect-knowledge-cosine'
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if index_name not in existing_indexes:
            raise ValueError(f"Index '{index_name}' does not exist.")

        index = pc.Index(index_name)
        time.sleep(1)
        return index
        
    def search(self, query, top_k=1, num_results=3, threshold=0.95):
        """Search the RAG database for the best match to the query.

        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to return.
            num_results (int): The number of results to return.
            threshold (float): The minimum score threshold for results.

        Returns:
            str: The best matching response or an empty string if no match meets the threshold.
        """
        emb = self.embedding_model.create_embeddings([query])[0].tolist()
        result = self.rag_db.query(vector=emb, top_k=top_k, include_metadata=True)
        bestmatch = result['matches'][0]
        score = bestmatch['score']
        metadata = bestmatch['metadata']
        question, answer = metadata['question'], metadata['answer']
        response = f'{question}\n{answer}'
        return response if score >= threshold else ""

    def parse_matches(self, matches, top_p=0.3):
        """Parse the matches from the RAG database query.

        Args:
            matches (list): The list of matches from the RAG database.
            top_p (float): The cumulative probability threshold for filtering matches.

        Returns:
            list: A list of tuples containing the context and score for each match.
        """
        contexts = []
        scores = []
        for eachMatch in matches:
            answer = eachMatch['metadata']['answer']
            score = eachMatch['score']
            contexts.append(answer)
            scores.append(score)
        top_p_indices = self.filter_by_top_p(scores, top_p)
        return [(contexts[index], scores[index]) for index in top_p_indices]

    def filter_by_top_p(self, scores, top_p):
        """Filter the scores by the top cumulative probability.

        Args:
            scores (list): The list of scores to filter.
            top_p (float): The cumulative probability threshold.

        Returns:
            list: A list of indices corresponding to the top p scores.
        """
        indexed_scores = list(enumerate(scores))
        sorted_indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        sorted_scores = [score for index, score in sorted_indexed_scores]
        sorted_indices = [index for index, score in sorted_indexed_scores]
        cumulative_sum = np.cumsum(sorted_scores)
        cumulative_probabilities = cumulative_sum / cumulative_sum[-1]
        top_p_index = np.searchsorted(cumulative_probabilities, top_p) + 1
        top_p_indices = sorted_indices[:top_p_index]
        return top_p_indices
