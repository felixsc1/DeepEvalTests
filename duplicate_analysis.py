#!/usr/bin/env python3
"""
Script to analyze and filter golden QA pairs for duplicates using Ollama embeddings.
Based on functions from filter_documents.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import sys
import os

# Import Ollama embeddings
from langchain_ollama import OllamaEmbeddings


def setup_ollama_embeddings(
    ollama_base_url: str = "http://localhost:11434",
    model_name: str = "mxbai-embed-large:335m",
):
    """
    Set up Ollama embeddings with error handling
    Falls back to text-based similarity if Ollama unavailable
    """
    try:
        embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=model_name)

        # Test the embeddings
        test_embedding = embeddings.embed_query("test query")
        print(
            f"✓ Ollama embeddings working. Model: {model_name}, Dimension: {len(test_embedding)}"
        )
        return embeddings, "ollama"

    except Exception as e:
        print(f"✗ Failed to setup Ollama embeddings: {e}")
        print("Falling back to TF-IDF text similarity analysis...")
        return None, "text"


def compute_text_similarities(questions, similarity_threshold=0.8):
    """
    Compute text-based similarities using TF-IDF and cosine similarity
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(questions)

    print(f"✓ Computed TF-IDF matrix: {tfidf_matrix.shape}")

    # Calculate pairwise similarities
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Convert to DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=questions, columns=questions)

    return similarity_df


def load_and_filter_data(quality_cutoff=0.8):
    """Load data and filter by quality"""
    print("Loading synthetic dataset...")
    with open("synthetic_dataset_with_urls.json", "r") as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")

    # Filter by quality
    filtered_data = [
        entry
        for entry in data
        if entry["additional_metadata"]["synthetic_input_quality"] >= quality_cutoff
    ]
    print(
        f"After quality filtering (>= {quality_cutoff}): {len(filtered_data)} entries remain"
    )

    # Extract questions
    questions = [entry["input"] for entry in filtered_data]
    print(f"Extracted {len(questions)} questions for similarity analysis")

    return filtered_data, questions


def compute_embeddings(questions, embeddings):
    """Compute embeddings for all questions"""
    print(f"Computing embeddings for {len(questions)} questions...")

    question_embeddings = []
    for i, question in enumerate(questions):
        if i % 20 == 0:
            print(f"Processed {i}/{len(questions)} questions...")
        try:
            embedding = embeddings.embed_query(question)
            question_embeddings.append(embedding)
        except Exception as e:
            print(f"Error embedding question {i}: {e}")
            question_embeddings.append(None)  # Placeholder for failed embeddings

    print(f"✓ Computed embeddings for {len(question_embeddings)} questions")

    # Check for any failed embeddings
    failed_count = sum(1 for emb in question_embeddings if emb is None)
    if failed_count > 0:
        print(f"⚠ {failed_count} questions failed to embed")

    return question_embeddings


def calculate_similarities(question_embeddings, questions):
    """Calculate pairwise cosine similarities"""
    from scipy.spatial.distance import cosine

    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors"""
        return 1 - cosine(a, b)

    # Filter out None embeddings
    valid_data = [
        (q, emb) for q, emb in zip(questions, question_embeddings) if emb is not None
    ]
    valid_questions = [q for q, emb in valid_data]
    embedding_matrix = np.array([emb for q, emb in valid_data])

    print(f"Computing similarity matrix for {len(valid_questions)} questions...")

    # Calculate pairwise similarities
    n = len(embedding_matrix)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):  # Only compute upper triangle
            sim = cosine_similarity(embedding_matrix[i], embedding_matrix[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric matrix

    print(f"✓ Computed similarity matrix ({n}x{n})")

    # Convert to DataFrame for easier analysis
    similarity_df = pd.DataFrame(
        similarity_matrix, index=valid_questions, columns=valid_questions
    )

    return similarity_df, valid_questions, valid_data


def analyze_similarities(similarity_df):
    """Analyze similarity distribution and find duplicates"""
    # Extract upper triangle similarities (excluding diagonal)
    upper_triangle = similarity_df.where(
        np.triu(np.ones_like(similarity_df), k=1).astype(bool)
    )
    similarities = upper_triangle.stack().values

    print("Similarity Statistics:")
    print(pd.Series(similarities).describe())

    # Plot similarity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, edgecolor="black", alpha=0.7)
    plt.title("Distribution of Question Similarities")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.axvline(
        x=0.8, color="red", linestyle="--", label="High similarity threshold (0.8)"
    )
    plt.axvline(
        x=0.9,
        color="orange",
        linestyle="--",
        label="Very high similarity threshold (0.9)",
    )
    plt.legend()
    plt.show()

    # Find highly similar question pairs
    def find_similar_pairs(similarity_df, threshold=0.8):
        """Find pairs of questions above similarity threshold"""
        pairs = []
        for i in range(len(similarity_df)):
            for j in range(i + 1, len(similarity_df)):
                sim = similarity_df.iloc[i, j]
                if sim >= threshold:
                    pairs.append(
                        {
                            "question1": similarity_df.index[i],
                            "question2": similarity_df.columns[j],
                            "similarity": sim,
                            "index1": i,
                            "index2": j,
                        }
                    )
        return pd.DataFrame(pairs).sort_values("similarity", ascending=False)

    # Find duplicates at different thresholds
    high_sim_pairs = find_similar_pairs(similarity_df, threshold=0.8)
    very_high_sim_pairs = find_similar_pairs(similarity_df, threshold=0.9)

    print(f"\nHighly similar pairs (≥0.8): {len(high_sim_pairs)}")
    print(f"Very similar pairs (≥0.9): {len(very_high_sim_pairs)}")

    if len(high_sim_pairs) > 0:
        print("\nTop 10 most similar question pairs:")
        for _, pair in high_sim_pairs.head(10).iterrows():
            print(f"Similarity: {pair['similarity']:.3f}")
            print(
                f"Q1: {pair['question1'][:100]}{'...' if len(pair['question1']) > 100 else ''}"
            )
            print(
                f"Q2: {pair['question2'][:100]}{'...' if len(pair['question2']) > 100 else ''}"
            )
            print("---")

    return high_sim_pairs, very_high_sim_pairs


def analyze_duplicate_removal(similarity_df, threshold=0.8):
    """Analyze how many entries would be removed at a given similarity threshold"""
    n = len(similarity_df)

    # Simple greedy approach: keep first occurrence, remove later similar ones
    to_remove = set()
    kept_indices = set()

    for i in range(n):
        if i in to_remove:
            continue

        kept_indices.add(i)

        # Mark similar questions for removal
        for j in range(i + 1, n):
            if similarity_df.iloc[i, j] >= threshold and j not in to_remove:
                to_remove.add(j)

    kept_count = len(kept_indices)
    removed_count = len(to_remove)

    return {
        "threshold": threshold,
        "original_count": n,
        "kept_count": kept_count,
        "removed_count": removed_count,
        "reduction_percentage": (removed_count / n) * 100,
        "kept_indices": kept_indices,
        "removed_indices": to_remove,
    }


def main():
    """Main analysis pipeline"""
    print("=== Golden QA Pairs Duplicate Analysis ===\n")

    # Step 1: Load and filter data
    quality_cutoff = 0.8
    filtered_data, questions = load_and_filter_data(quality_cutoff)
    print()

    # Step 2: Setup embeddings or text similarity
    embeddings, similarity_method = setup_ollama_embeddings()
    print()

    # Step 3: Compute similarities
    if similarity_method == "ollama":
        # Use Ollama embeddings
        question_embeddings = compute_embeddings(questions, embeddings)
        similarity_df, valid_questions, valid_data = calculate_similarities(
            question_embeddings, questions
        )
        filtered_data_valid = [
            filtered_data[i]
            for i in range(len(filtered_data))
            if question_embeddings[i] is not None
        ]
    else:
        # Use text-based similarity
        similarity_df = compute_text_similarities(questions)
        valid_questions = questions
        valid_data = [(q, None) for q in questions]  # No embeddings for text method
        filtered_data_valid = filtered_data  # All data is valid for text method
    print()

    # Step 5: Analyze similarities
    high_sim_pairs, very_high_sim_pairs = analyze_similarities(similarity_df)
    print()

    # Step 6: Duplicate removal analysis
    print("=== Duplicate Removal Analysis ===")
    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    removal_analysis = []

    for thresh in thresholds:
        result = analyze_duplicate_removal(similarity_df, thresh)
        removal_analysis.append(result)

        print(
            f"Threshold {thresh}: Keep {result['kept_count']}, Remove {result['removed_count']} "
            f"({result['reduction_percentage']:.1f}% reduction)"
        )

    # Create analysis DataFrame
    analysis_df = pd.DataFrame(removal_analysis)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        analysis_df["threshold"],
        analysis_df["kept_count"],
        "bo-",
        linewidth=2,
        markersize=8,
    )
    plt.title("Questions Remaining vs Similarity Threshold")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Questions Remaining")
    plt.grid(True, alpha=0.3)
    plt.xticks(thresholds)

    # Add target line for ~100 questions
    plt.axhline(
        y=100, color="red", linestyle="--", alpha=0.7, label="Target: ~100 questions"
    )
    plt.legend()

    # Add value labels
    for i, row in analysis_df.iterrows():
        plt.annotate(
            f'{int(row["kept_count"])}\n({row["reduction_percentage"]:.1f}% removed)',
            (row["threshold"], row["kept_count"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.show()

    print("\nRecommended thresholds for ~100 questions:")
    target_range = analysis_df[
        (analysis_df["kept_count"] >= 90) & (analysis_df["kept_count"] <= 110)
    ]
    if len(target_range) > 0:
        for _, row in target_range.iterrows():
            print(
                f"Threshold {row['threshold']}: {row['kept_count']} questions remaining"
            )
    else:
        print("No threshold gives exactly ~100 questions. Closest options:")
        closest = analysis_df.iloc[
            (analysis_df["kept_count"] - 100).abs().argsort()[:2]
        ]
        for _, row in closest.iterrows():
            print(
                f"Threshold {row['threshold']}: {row['kept_count']} questions remaining"
            )

    # Step 7: Export filtered dataset
    print("\n=== Export Filtered Dataset ===")

    # Choose threshold that gives closest to 100 questions
    target_threshold = 0.85  # Adjust based on analysis - this should give ~100

    # Get the removal analysis for this threshold
    removal_result = analyze_duplicate_removal(similarity_df, target_threshold)
    kept_indices = list(removal_result["kept_indices"])

    # Get the kept data (need to map back to filtered_data_valid)
    kept_data = [filtered_data_valid[i] for i in kept_indices]

    print(f"Applying threshold {target_threshold}: keeping {len(kept_data)} questions")

    # Save the filtered dataset
    output_file = (
        f"filtered_goldens_quality_{quality_cutoff}_similarity_{target_threshold}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(kept_data, f, indent=2)

    print(f"✓ Saved {len(kept_data)} filtered golden entries to {output_file}")

    # Show statistics of the final dataset
    final_qualities = [
        entry["additional_metadata"]["synthetic_input_quality"] for entry in kept_data
    ]
    final_context_qualities = [
        entry["additional_metadata"]["context_quality"] for entry in kept_data
    ]

    print("\nFinal dataset statistics:")
    print(f"  Questions: {len(kept_data)}")
    print(f"  Avg input quality: {np.mean(final_qualities):.3f}")
    print(f"  Avg context quality: {np.mean(final_context_qualities):.3f}")
    print(f"  Quality range: {min(final_qualities):.3f} - {max(final_qualities):.3f}")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
