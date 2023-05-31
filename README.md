Spiralflow Data Chat
====================

Spiralflow Data Chat is a chatbot designed to ingest, index, and interact with data using the Spiralflow framework. It can utilize multiple data sources to generate relevant context and provide accurate responses to user prompts. The chatbot takes into account the relevancy of each data source to ensure accurate and context-aware responses.

Features
--------

*   Utilize multiple data sources for context generation.
*   Generate accurate and context-aware responses based on the provided context and data sources.

Prerequisites
-------------

Before using Spiralflow Data Chat, make sure to have the following prerequisites:

*   Python 3.9 or later
*   An OpenAI API key with access to GPT-3.5-turbo (or another suitable chat models)

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
pip install spiralflow fastapi
conda install -c conda-forge pytorch faiss-cpu -y
```

Usage
-----

To run the Spiralflow Data Chat chatbot website, run the below command and go to the respective website:

```bash
python main.py
```


To run the chatbot in the terminal, simply execute the chatbot script:

```bash
python chatbot.py
```

You can also use command-line arguments to customize the chatbot's behavior, such as:

*   `--memory`: Memory to use.
*   `--memory_file`: Memory file to use (default: "memory\_default.pkl").
*   `--max_num_query_results`: Maximum number of results for context (default: 14).
*   `--num_query_results`: Number of memory queries (default: 20).
*   `--max_memory_context_tokens`: Maximum number of tokens for the full context (default: 1500).
*   `--memory_score_threshold`: Threshold for memory queries. A value of .8 is strict and .7 is loose (default: 0.7).
*   `--combine_threshold`: Threshold for combining memory queries (default: 0.1).
*   `--summarize_context`: Each document in context will be summarized, attempting to extract the relevant parts.
*   `--openai_chat_model`: OpenAI chat model to use (default: "gpt-3.5-turbo").
*   `--max_num_prompt_tokens`: Number of tokens a prompt can contain without having to break it up (default: 2000).
*   `--max_num_tokens_per_memory`: Number of tokens a prompt can contain without having to break it up (default: 500).
*   `--temperature`: Temperature for response generation (default: 0.3).
*   `--persona`: Persona to use for response generation. The persona is one aspect in a response and so may not be stressed (default: "").
*   `--enable_chat_history`: Enables queryable chat history so prompts can refer to previous prompts and responses.
*   `--max_chat_history_tokens`: Number of tokens chat history can have. Any more will be truncated (default: 2000).
*   `--enable_prompt_response_memory`: Enables memory of prompts and responses.
*   `--verbose`: Increase output verbosity.

For example:

```bash
python chatbot.py --memory_file memory_default.pkl --temperature 0.1 --enable_chat_history
```


Ingesting Data with `ingest.py`
-------------------------------

The `ingest.py` script helps you to ingest data and create memory for the SpiralFlow Data Chat chatbot. You can customize the data ingestion process by specifying the directory containing the data, the chunk size, overlap between chunks, input format, and postfix for the memory.

### Usage

1.  Replace the contents of the `data` folder with your data files.
2.  Remove the placeholder `memory_default.pkl` file from the repository.
3.  Run the `ingest.py` script:

```bash
python ingest.py
```

You can customize the data ingestion process using command-line arguments:

*   `-d`, `--directory`: Directory containing the data (default: "data/").
*   `-l`, `--load`: Load a previous memory and append to it.
*   `-pf`, `--postfix`: Postfix for the memory (default: "default").
*   `-c`, `--chunk_size`: Chunk size for text splitting (default: 240).
*   `-o`, `--chunk_overlap_factor`: Overlap factor between chunks for text splitting (default: 1/3).
*   `--dry_run`: Perform a dry run, stopping before embeddings are declared.

For example, to ingest data from a directory named `my_data` with a chunk size of 500, an overlap of 200, and an input format of "text", you can run:

```bash
python ingest.py -d my_data -c 500 -o 200
```

After running the `ingest.py` script, a new memory file will be generated based on your data. Make sure to update the `--memory_file` argument in the `main.py` script to use the new memory file.


License
-------

Spiralflow Data Chat is released under the [MIT License](LICENSE).
