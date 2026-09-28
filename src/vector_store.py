"""
Vector store implementation using FAISS with sentence-transformers
"""
import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.data_ingestion import MedicalDocument


class VectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        self.documents: List[Dict[str, Any]] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _chunk_documents(self, documents: List[MedicalDocument]) -> List[Dict[str, Any]]:
        """Split documents into chunks with metadata"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.content)
            
            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    "doc_id": doc.doc_id,
                    "chunk_id": f"{doc.doc_id}_chunk_{i}",
                    "content": chunk,
                    "title": doc.title,
                    "source": doc.source,
                    "year": doc.year,
                    "url": doc.url,
                    "metadata": doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def add_documents(self, documents: List[MedicalDocument]):
        """Add documents to the vector store"""
        print(f"Chunking {len(documents)} documents...")
        chunked_docs = self._chunk_documents(documents)
        print(f"Created {len(chunked_docs)} chunks")
        
        print("Generating embeddings...")
        texts = [doc["content"] for doc in chunked_docs]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        print("Adding to FAISS index...")
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(chunked_docs)
        
        print(f"Vector store now contains {len(self.documents)} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Filter by threshold and format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, save_dir: str):
        """Save vector store to disk"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save documents and metadata
        docs_path = save_path / "documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save configuration
        config_path = save_path / "config.pkl"
        config = {
            "embedding_model_name": self.embedding_model.get_sentence_embedding_dimension(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "dimension": self.dimension,
            "num_documents": len(self.documents)
        }
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Vector store saved to {save_path}")
    
    def load(self, load_dir: str):
        """Load vector store from disk"""
        load_path = Path(load_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store directory not found: {load_dir}")
        
        # Load FAISS index
        index_path = load_path / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        
        # Load documents
        docs_path = load_path / "documents.pkl"
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"Vector store loaded from {load_path}")
        print(f"Loaded {len(self.documents)} document chunks")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        unique_docs = len(set(doc["doc_id"] for doc in self.documents))
        
        return {
            "total_chunks": len(self.documents),
            "unique_documents": unique_docs,
            "dimension": self.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }