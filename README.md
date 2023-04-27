Spiralflow Data Chat
====================

Spiralflow Data Chat is a chatbot designed to ingest, index, and interact with data using langchain tools and the Spiralflow framework. It can utilize multiple data sources, such as memory, Google Search results, and general knowledge, to generate relevant context and provide accurate responses to user prompts. The chatbot takes into account the relevancy of each data source to ensure accurate and context-aware responses.

Features
--------

*   Ingest and index data using langchain tools and the Spiralflow framework.
*   Utilize multiple data sources for context generation, including memory, Google Search results, and general knowledge.
*   Determine the relevancy of each data source for responding to user prompts.
*   Generate accurate and context-aware responses based on the provided context and data sources.

Prerequisites
-------------

Before using Spiralflow Data Chat, make sure to have the following prerequisites:

*   Python 3.9 or later
*   An OpenAI API key with access to GPT-3.5-turbo (or another suitable language model)
*   A Google API key and Custom Search JSON API (CSE) ID for utilizing the Google Search Tool

Installation
------------

1.  Clone the repository:

bash

```bash
git clone https://github.com/Tiger767/spiralflow-data-chat.git
cd spiralflow-data-chat
```

2. Install the required Python packages:

bash

```bash
pip install -r requirements.txt
conda install -c conda-forge pytorch faiss-cpu -y
```

3.  Set up the required environment variables:

bash

```bash
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_CSE_ID="your_google_cse_id"
```

Usage
-----

To run the Spiralflow Data Chat chatbot, simply execute the main script:

bash

```bash
python main.py
```

You can also use command-line arguments to customize the chatbot's behavior, such as:

*   `--only_use_memory`: Use only memory for context generation.
*   `--max_num_databases`: Maximum number of databases to consider.
*   `--max_num_docs`: Maximum number of documents to include in the context.
*   `--max_num_memory_queries`: Maximum number of memory queries.
*   `--max_num_context_tokens`: Maximum number of tokens for the full context.
*   `--memory_score_threshold`: Threshold for memory queries. A value of .5 is strict and .7 is loose.
*   `--combine_threshold`: Threshold for combining memory queries.
*   `--memory_file`: Memory file to use.
*   `--verbose`: Increase output verbosity.
*   `--history`: Enables queryable chat history so prompts can refer to previous prompts and responses.
*   `--max_chat_history_tokens`: Number of tokens chat history can have. Any more will be truncated.
*   `--openai_chat_model`: OpenAI chat model to use (default: "gpt-3.5-turbo").
*   `--max_num_prompt_tokens`: Number of tokens a prompt can contain without having to break it up.
*   `--max_num_tokens_per_memory`: Number of tokens a memory can contain without having to break it up.
*   `--temperature`: Temperature for response generation.

For example:

bash

```bash
python main.py --only_use_memory --memory_file memory_default.pkl --temperature 0.3 --history
```


Ingesting Data with `ingest.py`
-------------------------------

The `ingest.py` script helps you to ingest data and create memory for the SpiralFlow Data Chat chatbot. You can customize the data ingestion process by specifying the directory containing the data, the chunk size, overlap between chunks, input format, and postfix for the memory.

### Usage

1.  Replace the contents of the `data` folder with your data files.
2.  Remove the placeholder `memory_default.pkl` file from the repository.
3.  Run the `ingest.py` script:

bash

```bash
python ingest.py
```

You can customize the data ingestion process using command-line arguments:

*   `-d`, `--directory`: Directory containing the data (default: "data/").
*   `-l`, `--load`: Load a previous memory and append to it.
*   `-pf`, `--postfix`: Postfix for the memory (default: "default").
*   `-c`, `--chunk_size`: Chunk size for text splitting (default: 240).
*   `-o`, `--chunk_overlap`: Overlap between chunks for text splitting (default: 80).
*   `--dry_run`: Perform a dry run, stopping before embeddings are declared.

For example, to ingest data from a directory named `my_data` with a chunk size of 500, an overlap of 200, and an input format of "text", you can run:

bash

```bash
python ingest.py -d my_data -c 500 -o 200
```

After running the `ingest.py` script, a new memory file will be generated based on your data. Make sure to update the `--memory_file` argument in the `main.py` script to use the new memory file.


License
-------

Spiralflow Data Chat is released under the [MIT License](LICENSE).