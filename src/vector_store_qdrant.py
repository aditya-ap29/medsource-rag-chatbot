"""
Alternative vector store implementation using Qdrant
"""
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.data_ingestion import MedicalDocument


class QdrantVectorStore:
    """Qdrant-based vector store for document retrieval"""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "medical_documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        use_memory: bool = True
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        if use_memory:
            # Use in-memory storage (for local development)
            self.client = QdrantClient(":memory:")
        elif qdrant_url:
            # Use cloud Qdrant
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Use local persistent storage
            self.client = QdrantClient(path="./data/qdrant_db")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            # Collection already exists
            print(f"📁 Using existing collection: {self.collection_name}")
    
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
            normalize_embeddings=True
        )
        
        print("Adding to Qdrant...")
        points = []
        for idx, (doc, embedding) in enumerate(zip(chunked_docs, embeddings)):
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=doc
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"✅ Vector store now contains {len(chunked_docs)} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding[0].tolist(),
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append((result.payload, result.score))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        collection_info = self.client.get_collection(self.collection_name)
        
        # Get unique document count
        all_points = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000
        )[0]
        
        unique_docs = len(set(point.payload.get("doc_id") for point in all_points))
        
        return {
            "total_chunks": collection_info.points_count,
            "unique_documents": unique_docs,
            "dimension": self.dimension,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "distance_metric": "cosine"
        }
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection_name)
        print(f"🗑️  Deleted collection: {self.collection_name}")


# Example usage
if __name__ == "__main__":
    from src.data_ingestion import MedicalDocument
    
    # Create sample document
    doc = MedicalDocument(
        content="Vitamin D deficiency causes fatigue and bone pain.",
        title="Vitamin D Deficiency",
        source="Medical Textbook",
        year="2023"
    )
    
    # Initialize Qdrant vector store
    store = QdrantVectorStore(use_memory=True)
    
    # Add documents
    store.add_documents([doc])
    
    # Search
    results = store.search("symptoms of vitamin deficiency", top_k=3)
    
    for doc, score in results:
        print(f"Score: {score:.3f}")
        print(f"Title: {doc['title']}")
        print(f"Content: {doc['content']}\n")
