import os
import sys
import logging
from typing import List, Optional, Any

# Robust import for text_processing_utils
try:
    # Standard import if 'src' is in PYTHONPATH or script is in 'src'
    from shared.text_processing_utils import normalize_arabic_text_for_embedding
except ImportError:
    # Fallback for running scripts from 'scripts/db_processing'
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from src.shared.text_processing_utils import normalize_arabic_text_for_embedding
    except ImportError:
        # Final fallback if all else fails
        def normalize_arabic_text_for_embedding(text: str) -> str:
            print("CRITICAL WARNING: normalize_arabic_text_for_embedding not found, using passthrough. Text will NOT be normalized.")
            return text

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr # Import PrivateAttr
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class MatryoshkaArabicEmbeddings(BaseModel, Embeddings):
    """
    Custom LangChain compatible embedding class for Arabic-Triplet-Matryoshka-V2 model.
    """
    model_name: str = Field(default="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2")
    truncate_dim: Optional[int] = Field(default=None, description="Dimension to truncate embeddings to. None means full dimension.")
    normalize_text: bool = Field(default=True, description="Whether to apply Arabic text normalization.")
    
    # Use PrivateAttr for internal state not part of the Pydantic model schema
    # This will store the actual SentenceTransformer instance.
    _model_instance_private: Optional[SentenceTransformer] = PrivateAttr(default=None)
    # Store device as a private attribute as well, initialized with the global DEVICE
    _device_private: str = PrivateAttr(default=DEVICE)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Initialize private attributes if they were passed in data (though not typical for PrivateAttr)
        # For _device_private, it's already defaulted. For _model_instance_private, it's loaded below.
        print(f"MatryoshkaArabicEmbeddings initialized. Device: {self._device_private}")
        self._load_model() # Load model on instantiation

    def _load_model(self) -> None:
        """Loads the SentenceTransformer model."""
        if self._model_instance_private is None: # Load only if not already loaded
            try:
                print(f"Loading SentenceTransformer model: {self.model_name} with truncate_dim={self.truncate_dim} on device {self._device_private}")
                self._model_instance_private = SentenceTransformer(
                    self.model_name,
                    device=self._device_private,
                    truncate_dim=self.truncate_dim 
                )
                print(f"Model {self.model_name} loaded successfully.")
            except Exception as e:
                print(f"Error loading SentenceTransformer model {self.model_name}: {e}")
                raise
        else:
            print(f"Model {self.model_name} is already loaded.")

    def _get_model(self) -> SentenceTransformer:
        """Ensures model is loaded and returns it."""
        if self._model_instance_private is None:
            self._load_model() 
        
        if not isinstance(self._model_instance_private, SentenceTransformer):
            print("CRITICAL ERROR: _model_instance_private is not a SentenceTransformer object after loading attempt.")
            raise TypeError("Model instance is not correctly loaded or is of an unexpected type.")
        return self._model_instance_private

    def _normalize_if_enabled(self, texts: List[str]) -> List[str]:
        if self.normalize_text:
            try:
                return [normalize_arabic_text_for_embedding(text) for text in texts]
            except NameError: 
                print("CRITICAL WARNING: normalize_arabic_text_for_embedding function is not defined. Using passthrough.")
                return texts
        return texts

    def embed_documents(self, texts: List[str], show_progress_bar: bool = False) -> List[List[float]]:
        """
        Embed a list of documents.
        Args:
            texts: The list of texts to embed.
            show_progress_bar: Whether to show a progress bar.
        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []
        
        model = self._get_model() 
        normalized_texts = self._normalize_if_enabled(texts)
        
        logger.info(f"Embedding {len(normalized_texts)} documents with Matryoshka...")
        embeddings_np = model.encode(normalized_texts, convert_to_tensor=False, show_progress_bar=show_progress_bar) 
        logger.info("Finished embedding documents with Matryoshka.")
        return [embedding.tolist() for embedding in embeddings_np]

    def embed_query(self, text: str, show_progress_bar: bool = False) -> List[float]:
        """
        Embed a single query text.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        model = self._get_model() 
        normalized_text = text
        if self.normalize_text:
            normalized_text = normalize_arabic_text_for_embedding(text)
            
        logger.info(f"Embedding query with Matryoshka: '{normalized_text[:100]}...'")
        embedding_np = model.encode(normalized_text, convert_to_tensor=False, show_progress_bar=show_progress_bar) 
        logger.info("Finished embedding query with Matryoshka.")
        return embedding_np.tolist()

if __name__ == '__main__':
    print(f"Testing MatryoshkaArabicEmbeddings on device: {DEVICE}") # DEVICE here refers to the global one

    # Test with default full dimension
    print("\n--- Testing with full dimension (768) ---")
    try:
        # When instantiating, Pydantic fields are passed. PrivateAttrs are handled by __init__ or default.
        embedder_full = MatryoshkaArabicEmbeddings() 
        sample_texts_ar = [
            "هذه جملة تجريبية باللغة العربية.",
            "ما هي الاستشارة القانونية؟",
            "القانون اليمني الجديد"
        ]
        print(f"Sample texts: {sample_texts_ar}")

        doc_embeddings_full = embedder_full.embed_documents(sample_texts_ar)
        print(f"Document embeddings shape: ({len(doc_embeddings_full)}, {len(doc_embeddings_full[0]) if doc_embeddings_full else 0})")
        
        query_embedding_full = embedder_full.embed_query(sample_texts_ar[0])
        print(f"Query embedding shape: ({len(query_embedding_full)})")
    except Exception as e:
        print(f"Error during full dimension test: {e}")
        import traceback
        traceback.print_exc()


    # Test with truncated dimension
    print("\n--- Testing with truncated dimension (256) ---")
    try:
        embedder_truncated = MatryoshkaArabicEmbeddings(truncate_dim=256)
        doc_embeddings_truncated = embedder_truncated.embed_documents(sample_texts_ar)
        print(f"Document embeddings shape (truncated): ({len(doc_embeddings_truncated)}, {len(doc_embeddings_truncated[0]) if doc_embeddings_truncated else 0})")

        query_embedding_truncated = embedder_truncated.embed_query(sample_texts_ar[0])
        print(f"Query embedding shape (truncated): ({len(query_embedding_truncated)})")
    except Exception as e:
        print(f"Error during truncated dimension test: {e}")
        import traceback
        traceback.print_exc()

    # Test normalization
    print("\n--- Testing normalization ---")
    try:
        embedder_norm = MatryoshkaArabicEmbeddings(normalize_text=True, truncate_dim=64) 
        embedder_no_norm = MatryoshkaArabicEmbeddings(normalize_text=False, truncate_dim=64)
        
        raw_text = "الإستشارة القَانُونية المٌقدمة كانت مٌفيدة."
        print(f"Raw text: '{raw_text}'")
        
        try:
            normalized_text_manual = normalize_arabic_text_for_embedding(raw_text)
            print(f"Manually normalized by imported function: '{normalized_text_manual}'")
        except NameError:
            print("normalize_arabic_text_for_embedding is not defined globally for manual test.")
            normalized_text_manual = raw_text 

        embedding_norm = embedder_norm.embed_query(raw_text)
        embedding_no_norm = embedder_no_norm.embed_query(raw_text)
        
        normalized_by_class = embedder_norm._normalize_if_enabled([raw_text])[0]
        if normalized_by_class != raw_text:
             print(f"Text normalized by class: '{normalized_by_class}' - different from raw, good.")
        else:
             print(f"Text normalized by class is same as raw: '{normalized_by_class}' - check normalization logic for this input if it was expected to change.")

        if embedding_norm != embedding_no_norm and normalized_by_class != raw_text :
            print("Normalization test: Embeddings differ when normalization changes text, as expected.")
        elif normalized_by_class == raw_text:
            print("Normalization test: Text was not changed by normalization, so embeddings might be the same.")
        else: 
            print("Normalization test: Text changed but embeddings are identical. Model might be robust to these specific normalizations.")

    except Exception as e:
        print(f"Error during normalization test: {e}")
        import traceback
        traceback.print_exc()
