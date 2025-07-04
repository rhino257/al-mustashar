"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.
"""

import logging # Added for logging
import os
from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared.configuration import BaseConfiguration 
from custom_providers.openai_custom_embeddings import OpenAICustomEmbeddings # Added import

logger = logging.getLogger(__name__) # Added logger instance

## Encoder constructors


def make_text_encoder(embedding_model_name_prefixed: str) -> Embeddings: # Renamed 'model' to 'embedding_model_name_prefixed' for clarity
    """Connect to the configured text encoder."""
    provider, model_name_part = embedding_model_name_prefixed.split("/", maxsplit=1)
    
    if provider == "openai_custom_embeddings":
        logger.info(f"Loading custom OpenAI embedding model: {model_name_part}")
        # Assuming OpenAICustomEmbeddings handles API key via environment variable OPENAI_API_KEY
        # and has its own defaults.
        # If specific kwargs (e.g. dimensions) were needed from BaseConfiguration
        # they would be passed here.
        return OpenAICustomEmbeddings(model_name=model_name_part)
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Loading OpenAI embedding model: {model_name_part}")
        return OpenAIEmbeddings(model=model_name_part)
    elif provider == "cohere":
        from langchain_cohere import CohereEmbeddings
        logger.info(f"Loading Cohere embedding model: {model_name_part}")
        return CohereEmbeddings(model=model_name_part)  # type: ignore
    elif provider == "matryoshka_arabic": # New case for Matryoshka
        from custom_providers.matryoshka_arabic_embeddings import MatryoshkaArabicEmbeddings
        logger.info(f"Loading MatryoshkaArabicEmbeddings model: {model_name_part}") # model_name_part could be 'default_768dim'
        # Assuming MatryoshkaArabicEmbeddings takes normalize_text and potentially other params
        # For now, using a common default. Adjust if specific params are needed from model_name_part
        return MatryoshkaArabicEmbeddings(normalize_text=True)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
