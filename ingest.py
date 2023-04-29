import argparse
import os
from pathlib import Path
import tiktoken

from spiralflow.memory import Memory
from spiralflow.loading import (
    DirectoryMultiLoader,
    PDFLoader,
    HTMLLoader,
    TextLoader,
)
from spiralflow.chunking import SmartChunker


def delete_files_without_extension(folder_path, extension=None):
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    deleted_files_count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if (extension is None and "." in file) or (
                extension is not None and not file.endswith(extension)
            ):
                file_path = Path(root) / file
                file_path.unlink()
                deleted_files_count += 1
                print(f"Deleted: {file_path}")


def ingest_data(
    directory,
    chunk_size,
    chunk_overlap_factor,
    vector_postfix,
    load=None,
    dry_run=False,
):
    # delete extra files
    # delete_files_without_extension("data/catalog", extension='.pdf')

    memory = Memory()
    if load is not None and len(load) > 0:
        memory.load(load)

    encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
    chunker = SmartChunker(
        encoder=encoder,
        chunk_size=chunk_size,
        overlap_factor=chunk_overlap_factor,
    )

    rchunker = SmartChunker(
        encoder=encoder,
        chunk_size=chunk_size * 4,
        overlap_factor=chunk_overlap_factor / 4,
        delimiters_tolerances_overlap=[
            ("\nclass ", 3 / 4 + 1 / 4 * 0.5, False),
            ("\n\n\n", 3 / 4 + 1 / 4 * 0.5, False),
            ("\n\n", 3 / 4 + 1 / 4 * 0.2, True),
            ("\n", 3 / 4 + 1 / 4 * 0.1, True),
            (" ", 3 / 4, True),
            ("", 3 / 4, True),
        ],
        prefer_large_chunks=False,
    )

    loaders = {
        ".*\.pdf": PDFLoader(chunker=chunker),
        ".*((\.txt)|(\.md))": TextLoader(chunker=chunker),
        ".*\.py": TextLoader(chunker=rchunker),
        ".*\.html": HTMLLoader(chunker=chunker),
    }
    loader = DirectoryMultiLoader(directory, loader=loaders, should_recurse=True)

    documents = loader.load()
    total_document_chunks = len(documents)

    for document in documents[:4]:
        print(f"'{document['content']}'", "\n", document["path"], end="\n\n\n")

    for document in documents:
        source = document["path"]
        if "page_number" in document:
            source += f"-pg-{document['page_number']}"
        if not dry_run:
            # print(len(document["content"]), source)
            memory.add(
                {
                    "text": document["content"],
                    "metadata": "source: " + source,
                }
            )

    if dry_run:
        print(
            f"Dry run completed. Total Document Chunks: {total_document_chunks} - Total Tokens Estimated: {total_document_chunks * chunk_size}"
        )
    else:
        memory.save(f"memory_{vector_postfix}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest data and create memory for spiral."
    )
    parser.add_argument(
        "-d", "--directory", help="Directory containing the data.", default="data/"
    )
    parser.add_argument(
        "-l", "--load", help="Load a previous memory and append to it.", default=""
    )
    parser.add_argument(
        "-pf", "--postfix", help="Postfix for the memory.", default="default"
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        help="Chunk size for text splitting.",
        type=int,
        default=240,
    )
    parser.add_argument(
        "-o",
        "--chunk_overlap_factor",
        help="Overlap factor between chunks for text splitting.",
        type=float,
        default=1 / 3,
    )
    parser.add_argument(
        "--dry_run",
        help="Dry run, stops before embeddings are declared.",
        action="store_true",
    )

    args = parser.parse_args()

    ingest_data(
        args.directory,
        args.chunk_size,
        args.chunk_overlap_factor,
        args.postfix,
        load=args.load,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
