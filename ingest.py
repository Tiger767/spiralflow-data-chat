import argparse
import os
from pathlib import Path

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter,
    TokenTextSplitter,
)
from langchain.document_loaders import (
    PyMuPDFLoader,
    BSHTMLLoader,
    DirectoryLoader,
    TextLoader,
)

from spiralflow.memory import Memory


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
    chunk_overlap,
    vector_postfix,
    load=None,
    dry_run=False,
):
    # delete extra files
    # delete_files_without_extension("data/catalog", extension='.pdf')

    memory = Memory()
    if load is not None and len(load) > 0:
        memory.load(load)

    loaders = {
        "pdf": DirectoryLoader(
            directory, loader_cls=PyMuPDFLoader, glob="*.pdf", recursive=True
        ),
        "text": DirectoryLoader(
            directory,
            loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
            glob="*.txt",
            recursive=True,
        ),
        "html": DirectoryLoader(
            directory, loader_cls=BSHTMLLoader, glob="*.html", recursive=True
        ),
        "python": DirectoryLoader(
            directory,
            loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
            glob="*.py",
            recursive=True,
        ),
        "markdown": DirectoryLoader(
            directory,
            loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
            glob="*.md",
            recursive=True,
        ),
    }

    total_documents = 0
    total_document_chunks = 0
    for input_format, loader in loaders.items():
        raw_documents = loader.load()
        total_documents += len(raw_documents)

        if input_format in ["text", "pdf", "html", "markdown"]:
            splitter = TokenTextSplitter(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Does not follow chunking size well
            # splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            #    encoding_name="cl100k_base",
            #    chunk_size=chunk_size,
            #    chunk_overlap=chunk_overlap,
            # )
        elif input_format == "python":
            splitter = PythonCodeTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            raise ValueError(f"Unknown input format: {input_format}")

        documents = splitter.split_documents(raw_documents)
        total_document_chunks += len(documents)

        for document in documents[::5][:3]:
            print(document.page_content, "\n", document.metadata, end="\n\n")

        for document in documents:
            source = document.metadata["source"]
            if "page_number" in document.metadata:
                source += f"-pg-{document.metadata['page_number']}"
            if not dry_run:
                memory.add(
                    {
                        "text": document.page_content,
                        "metadata": "source: " + source,
                    }
                )

    if dry_run:
        print(
            f"Dry run completed. Total Documents: {total_documents} - Total Document Chunks: {total_document_chunks} - Total Tokens Estimated: {total_document_chunks * chunk_size}"
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
        "--chunk_overlap",
        help="Overlap between chunks for text splitting.",
        type=int,
        default=80,
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
        args.chunk_overlap,
        args.postfix,
        load=args.load,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
