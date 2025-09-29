import json


def update_paths(json_file_path, old_prefix, new_prefix):
    """
    Updates all paths in the filtered_document_paths array of a JSON file.

    Args:
        json_file_path (str): Path to the JSON file to update
        old_prefix (str): The old path prefix to replace
        new_prefix (str): The new path prefix to use
    """
    # Load the JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Update all paths in the filtered_document_paths array
    updated_count = 0
    for i, path in enumerate(data["filtered_document_paths"]):
        if path.startswith(old_prefix):
            data["filtered_document_paths"][i] = path.replace(old_prefix, new_prefix, 1)
            updated_count += 1

    # Save the updated JSON back to the file
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated {updated_count} paths from '{old_prefix}' to '{new_prefix}'")


if __name__ == "__main__":
    # Define the paths
    json_file = "filtered_document_paths.json"
    # Use standard string literals so the trailing backslash is a single character
    old_prefix = (
        "C:\\GitRepos\\LangChainCourse\\documentation_assistant\\raw_documents\\"
    )
    new_prefix = "C:\\GitRepos\\RAG_learning\\DeepEvalTests\\raw_documents\\"

    # Update the paths
    update_paths(json_file, old_prefix, new_prefix)
