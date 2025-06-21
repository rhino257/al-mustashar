import os
import sys
import time # For timing
import logging
from typing import List, Optional, Any

# Robust import for text_processing_utils
try:
    from shared.text_processing_utils import normalize_arabic_text_for_embedding
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from src.shared.text_processing_utils import normalize_arabic_text_for_embedding
    except ImportError:
        def normalize_arabic_text_for_embedding(text: str) -> str:
            print("CRITICAL WARNING: normalize_arabic_text_for_embedding not found, using passthrough.")
            return text

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class BgeM3LawEmbeddings(BaseModel, Embeddings):
    """
    Custom LangChain compatible embedding class for mhaseeb1604/bge-m3-law model.
    Outputs 1024-dimensional embeddings.
    """
    model_name: str = Field(default="mhaseeb1604/bge-m3-law")
    normalize_text: bool = Field(default=True, description="Whether to apply Arabic text normalization.")
    
    _model_instance_private: Optional[SentenceTransformer] = PrivateAttr(default=None)
    _device_private: str = PrivateAttr(default=DEVICE)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        print(f"BgeM3LawEmbeddings initialized. Device: {self._device_private}")
        self._load_model()

    def _load_model(self) -> None:
        if self._model_instance_private is None:
            try:
                print(f"Loading SentenceTransformer model: {self.model_name} on device {self._device_private}")
                self._model_instance_private = SentenceTransformer(
                    self.model_name,
                    device=self._device_private
                )
                print(f"Model {self.model_name} loaded successfully.")
            except Exception as e:
                print(f"Error loading SentenceTransformer model {self.model_name}: {e}")
                raise
        else:
            print(f"Model {self.model_name} is already loaded.")

    def _get_model(self) -> SentenceTransformer:
        if self._model_instance_private is None:
            self._load_model() 
        if not isinstance(self._model_instance_private, SentenceTransformer):
            raise TypeError("BgeM3LawEmbeddings: Model instance is not correctly loaded.")
        return self._model_instance_private

    def _normalize_if_enabled(self, texts: List[str]) -> List[str]:
        if self.normalize_text:
            # print(f"Normalizing {len(texts)} texts...") # Can be too verbose for many calls
            normalized_texts = [normalize_arabic_text_for_embedding(text) for text in texts]
            # print("Normalization complete.")
            return normalized_texts
        return texts

    def embed_documents(self, texts: List[str], show_progress_bar: bool = False) -> List[List[float]]:
        if not texts:
            return []
        
        normalized_texts = self._normalize_if_enabled(texts)
        model = self._get_model()
        
        logger.info(f"Embedding {len(normalized_texts)} documents with BGE-M3-Law...")
        embeddings_np = model.encode(normalized_texts, convert_to_tensor=False, show_progress_bar=show_progress_bar)
        logger.info("Finished embedding documents with BGE-M3-Law.")
        
        return [embedding.tolist() for embedding in embeddings_np]

    def embed_query(self, text: str, show_progress_bar: bool = False) -> List[float]:
        normalized_text = text
        if self.normalize_text:
            normalized_text = normalize_arabic_text_for_embedding(text)
        
        model = self._get_model()
        logger.info(f"Embedding query with BGE-M3-Law: '{normalized_text[:100]}...'")
        embedding_np = model.encode(normalized_text, convert_to_tensor=False, show_progress_bar=show_progress_bar)
        logger.info("Finished embedding query with BGE-M3-Law.")
        return embedding_np.tolist()

if __name__ == '__main__':
    print(f"Testing BgeM3LawEmbeddings on device: {DEVICE}")
    try:
        embedder = BgeM3LawEmbeddings()
        sample_texts_ar = [
            "هذه جملة قانونية باللغة العربية.",
            "ما هي أركان الجريمة؟"
        ]
        print(f"Sample texts: {sample_texts_ar}")

        print("\nTesting embed_documents:")
        doc_start_time = time.time()
        doc_embeddings = embedder.embed_documents(sample_texts_ar)
        doc_end_time = time.time()
        print(f"Document embeddings shape: ({len(doc_embeddings)}, {len(doc_embeddings[0]) if doc_embeddings else 0})")
        print(f"Time taken for embed_documents: {doc_end_time - doc_start_time:.4f} seconds")
        
        print("\nTesting embed_query:")
        query_start_time = time.time()
        query_embedding = embedder.embed_query(sample_texts_ar[0])
        query_end_time = time.time()
        print(f"Query embedding shape: ({len(query_embedding)})")
        print(f"Time taken for embed_query: {query_end_time - query_start_time:.4f} seconds")
        
        print("\nBgeM3LawEmbeddings test completed successfully.")
    except Exception as e:
        print(f"Error during BgeM3LawEmbeddings test: {e}")
        import traceback
        traceback.print_exc()
