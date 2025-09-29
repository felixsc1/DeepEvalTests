#!/usr/bin/env python3
"""
Document filtering script to remove duplicates and low-quality documents
before running DeepEval synthesizer.

Uses LangChain's EmbeddingsRedundantFilter with local Ollama embeddings.
"""

import os
import json
import glob
from typing import List, Dict, Tuple
import re
from pathlib import Path

# LangChain imports
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def collect_document_paths_and_sources(main_folder: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Collect all .txt paths and map to URLs from source_map.json
    Same logic as in the notebook
    """
    document_paths = []
    path_to_url = {}

    # Walk through all subfolders
    for root, dirs, files in os.walk(main_folder):
        # Check for source_map.json in this subfolder
        source_map_path = os.path.join(root, 'source_map.json')
        source_map = {}
        if os.path.exists(source_map_path):
            with open(source_map_path, 'r') as f:
                source_map = json.load(f)

        # Find all .txt files in this subfolder
        txt_files = glob.glob(os.path.join(root, '*.txt'))
        for txt_path in txt_files:
            basename = os.path.basename(txt_path)
            url = source_map.get(basename, 'unknown')  # Get URL if mapped, else 'unknown'
            document_paths.append(txt_path)
            path_to_url[txt_path] = url

    return document_paths, path_to_url


def load_documents_from_paths(document_paths: List[str]) -> List[Document]:
    """
    Load documents from file paths into LangChain Document objects
    """
    documents = []
    for path in document_paths:
        try:
            # Load the text content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create LangChain Document
            doc = Document(
                page_content=content,
                metadata={"source": path}
            )
            documents.append(doc)

        except Exception as e:
            print(f"Warning: Failed to load document {path}: {e}")
            continue

    return documents


def is_document_useful(content: str, min_length: int = 100, min_words: int = 20) -> bool:
    """
    Basic heuristic filtering for nonsensical/useless documents

    Args:
        content: Document text content
        min_length: Minimum character length
        min_words: Minimum word count

    Returns:
        True if document appears useful, False otherwise
    """
    if not content or len(content.strip()) < min_length:
        return False

    # Count actual words (split by whitespace)
    words = content.strip().split()
    if len(words) < min_words:
        return False

    # Check for excessive repetition (simple heuristic)
    # If more than 50% of content is repeated words/phrases
    word_counts = {}
    for word in words:
        word = word.lower().strip('.,!?;:')
        if len(word) > 2:  # Only count meaningful words
            word_counts[word] = word_counts.get(word, 0) + 1

    total_meaningful_words = sum(count for word, count in word_counts.items() if count > 0)
    repeated_words = sum(count for word, count in word_counts.items() if count > 3)

    if total_meaningful_words > 0 and (repeated_words / total_meaningful_words) > 0.5:
        return False

    # Check for gibberish (high ratio of non-alphanumeric characters)
    alphanumeric_chars = sum(1 for c in content if c.isalnum() or c.isspace())
    total_chars = len(content)
    if total_chars > 0 and (alphanumeric_chars / total_chars) < 0.6:
        return False

    # Check for URLs or email patterns (might indicate navigation/error pages)
    url_pattern = r'https?://[^\s]+'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    if len(re.findall(url_pattern, content)) > len(words) * 0.1:  # Too many URLs
        return False

    if len(re.findall(email_pattern, content)) > 5:  # Too many emails
        return False

    return True


def filter_useful_documents(documents: List[Document]) -> List[Document]:
    """
    Filter out documents that appear to be nonsensical or useless
    """
    useful_docs = []
    filtered_count = 0

    for doc in documents:
        if is_document_useful(doc.page_content):
            useful_docs.append(doc)
        else:
            filtered_count += 1
            print(f"Filtered out low-quality document: {doc.metadata.get('source', 'unknown')}")

    print(f"Quality filtering: kept {len(useful_docs)} documents, filtered {filtered_count}")
    return useful_docs


def filter_duplicate_documents(documents: List[Document], ollama_base_url: str = "http://localhost:11434",
                              model_name: str = "nomic-embed-text", use_fallback: bool = True) -> List[Document]:
    """
    Use embeddings to remove duplicate/similar documents.
    Falls back to simple text-based deduplication if Ollama is unavailable.

    Args:
        documents: List of LangChain Document objects
        ollama_base_url: Ollama server URL
        model_name: Ollama embedding model name
        use_fallback: Whether to use fallback method if Ollama fails

    Returns:
        Filtered list of documents with duplicates removed
    """
    if not documents:
        return []

    print(f"Starting duplicate filtering on {len(documents)} documents...")

    # Try to use Ollama embeddings first
    try:
        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=model_name
        )
        print(f"Using Ollama embeddings with model: {model_name}")

        # Test embedding functionality
        test_embedding = embeddings.embed_query("test")
        print(f"Embeddings working, dimension: {len(test_embedding)}")

        # Use embedding-based filtering
        return _filter_duplicates_with_embeddings(documents, embeddings)

    except Exception as e:
        print(f"Ollama embeddings failed: {e}")
        if use_fallback:
            print("Falling back to text-based duplicate detection...")
            return _filter_duplicates_text_based(documents)
        else:
            print("Skipping duplicate filtering due to Ollama unavailability.")
            return documents


def _filter_duplicates_with_embeddings(documents: List[Document], embeddings) -> List[Document]:
    """
    Filter duplicates using embeddings
    """
    filtered_docs = []
    seen_embeddings = []
    similarity_threshold = 0.9

    print("Computing embeddings and filtering duplicates...")
    for i, doc in enumerate(documents):
        if i % 50 == 0:
            print(f"Processed {i}/{len(documents)} documents...")

        try:
            # Get embedding for current document
            embedding = embeddings.embed_query(doc.page_content)

            # Check similarity with previously seen documents
            is_duplicate = False
            for seen_embedding in seen_embeddings:
                # Simple cosine similarity
                import numpy as np
                dot_product = np.dot(embedding, seen_embedding)
                norm_a = np.linalg.norm(embedding)
                norm_b = np.linalg.norm(seen_embedding)

                if norm_a == 0 or norm_b == 0:
                    continue

                similarity = dot_product / (norm_a * norm_b)

                if similarity > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_docs.append(doc)
                seen_embeddings.append(embedding)

        except Exception as e:
            print(f"Error processing document {doc.metadata.get('source', 'unknown')}: {e}")
            # Keep the document if we can't process it
            filtered_docs.append(doc)

    duplicate_count = len(documents) - len(filtered_docs)
    print(f"Embedding-based duplicate filtering: kept {len(filtered_docs)} documents, removed {duplicate_count} duplicates")

    return filtered_docs


def _filter_duplicates_text_based(documents: List[Document]) -> List[Document]:
    """
    Simple text-based duplicate filtering as fallback
    """
    filtered_docs = []
    seen_hashes = set()

    for doc in documents:
        # Create a simple hash of the document content (first 1000 chars)
        content_hash = hash(doc.page_content[:1000].strip().lower())

        if content_hash not in seen_hashes:
            filtered_docs.append(doc)
            seen_hashes.add(content_hash)

    duplicate_count = len(documents) - len(filtered_docs)
    print(f"Text-based duplicate filtering: kept {len(filtered_docs)} documents, removed {duplicate_count} duplicates")

    return filtered_docs


def save_filtered_paths(filtered_documents: List[Document], output_file: str = "filtered_document_paths.json"):
    """
    Save the filtered document paths to a JSON file
    """
    filtered_paths = [doc.metadata["source"] for doc in filtered_documents]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "filtered_document_paths": filtered_paths,
            "total_filtered": len(filtered_paths)
        }, f, indent=2)

    print(f"Saved {len(filtered_paths)} filtered document paths to {output_file}")


def main():
    """
    Main function to run the document filtering pipeline

    To use with Ollama embeddings:
    1. Install Ollama: https://ollama.ai/
    2. Pull an embedding model: ollama pull nomic-embed-text
    3. Start Ollama server: ollama serve
    4. Run this script

    If Ollama is not available, the script will fall back to text-based filtering.
    """
    # Configuration
    main_folder = r'C:\GitRepos\LangChainCourse\documentation_assistant\raw_documents'
    ollama_base_url = "http://localhost:11434"  # Default Ollama URL
    embedding_model = "mxbai-embed-large:latest"  # Good general-purpose embedding model
    output_file = "filtered_document_paths.json"

    print("=== Document Filtering Setup Instructions ===")
    print("For optimal duplicate detection, install and run Ollama:")
    print("1. Install Ollama from: https://ollama.ai/")
    print("2. Pull embedding model: ollama pull nomic-embed-text")
    print("3. Start server: ollama serve")
    print("4. Run this script")
    print("If Ollama is unavailable, text-based filtering will be used as fallback.")
    print()

    print("=== Document Filtering Pipeline ===")
    print(f"Main folder: {main_folder}")
    print(f"Ollama URL: {ollama_base_url}")
    print(f"Embedding model: {embedding_model}")
    print()

    # Step 1: Collect all documents
    print("Step 1: Collecting document paths...")
    document_paths, path_to_url = collect_document_paths_and_sources(main_folder)
    print(f"Found {len(document_paths)} total .txt files")
    print()

    # Step 2: Load documents into LangChain format
    print("Step 2: Loading documents...")
    documents = load_documents_from_paths(document_paths)
    print(f"Successfully loaded {len(documents)} documents")
    print()

    # Step 3: Filter out low-quality documents
    print("Step 3: Filtering low-quality documents...")
    quality_filtered_docs = filter_useful_documents(documents)
    print()

    # Step 4: Filter out duplicates using embeddings
    print("Step 4: Filtering duplicate documents...")
    final_filtered_docs = filter_duplicate_documents(
        quality_filtered_docs,
        ollama_base_url=ollama_base_url,
        model_name=embedding_model
    )
    print()

    # Step 5: Save results
    print("Step 5: Saving filtered document paths...")
    save_filtered_paths(final_filtered_docs, output_file)
    print()

    print("=== Filtering Complete ===")
    print(f"Original documents: {len(document_paths)}")
    print(f"After quality filtering: {len(quality_filtered_docs)}")
    print(f"After duplicate filtering: {len(final_filtered_docs)}")
    print(f"Filtered paths saved to: {output_file}")

    # Show sample of filtered paths
    if final_filtered_docs:
        print("\nSample filtered paths:")
        for i, doc in enumerate(final_filtered_docs[:5]):
            print(f"  {i+1}. {doc.metadata['source']}")


if __name__ == "__main__":
    main()
