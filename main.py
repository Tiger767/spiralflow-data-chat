import os
import argparse
import concurrent.futures
import regex as re

import tiktoken

from spiralflow.message import Role
from spiralflow.chat_llm import ChatLLM
from spiralflow.flow import (
    ChatFlow,
    SequentialChatFlows,
    ConcurrentChatFlows,
    FuncChatFlow,
    NoHistory,
    MemoryChatFlow,
)
from spiralflow.tools import GoogleSearchTool
from spiralflow.memory import Memory
from spiralflow.chat_history import ChatHistory


def main(args):
    # Initialize memory, encoder, and conversation flows
    memory = Memory(args.memory_file)
    encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
    respond_flow = create_respond_flow(
        memory,
        encoder,
        max_num_memory_queries=args.max_num_memory_queries,
        combine_threshold=args.combine_threshold,
        max_num_databases=args.max_num_databases,
        max_num_docs=args.max_num_docs,
        memory_score_threshold=args.memory_score_threshold,
        max_num_context_tokens=args.max_num_context_tokens,
        only_use_memory=args.only_use_memory,
        verbose=args.verbose,
    )
    chat_llm = ChatLLM(gpt_model=args.openai_chat_model, temperature=args.temperature)

    # Initialize chat history
    chat_history = ChatHistory() if args.history else None

    # Main loop for receiving prompts and generating responses
    response = ""
    while True:
        response, chat_history = process_single_prompt(
            memory,
            encoder,
            respond_flow,
            chat_llm,
            chat_history,
            args,
            response,
        )


def process_single_prompt(
    memory,
    encoder,
    respond_flow,
    chat_llm,
    chat_history,
    args,
    response,
):
    """
    Process a single prompt, going through the respond_flow.
    Add the prompt to the memory and update chat history if using history.
    """
    prompt, num_tokens = handle_prompt(
        encoder,
        args.max_num_prompt_tokens,
        response,
    )
    if prompt is None:
        return
    os.system("clear")

    response, history = generate_response(
        respond_flow, chat_llm, chat_history, args.persona, prompt
    )

    if chat_history is not None:
        chat_history = update_memory_and_chat_history(
            memory,
            encoder,
            args.memory_file,
            args.max_chat_history_tokens,
            args.max_num_tokens_per_memory,
            history,
            prompt,
            response,
        )
    return response, chat_history


def handle_prompt(
    encoder,
    max_num_prompt_tokens,
    last_response,
):
    """
    Handles user input of prompts.

    Replaces [LAST_RESPONSE] in the prompt with the last single response.
    Checks if the prompt is too long and quits if so.
    Asks to clarify the prompt, and does so if requested.
    """
    prompt = get_input()
    if prompt is None:
        return None, 0
    prompt = prompt.replace("[LAST_RESPONSE]", last_response)

    num_tokens = len(encoder.encode(prompt))
    if num_tokens > max_num_prompt_tokens:
        print("\nPrompt too long.")
        return None, 0

    return prompt, num_tokens


def generate_response(
    respond_flow, chat_llm, chat_history, persona, prompt, print_prompt=False
):
    """
    Generates response to a prompt, running it through the respond_flow.
    """
    input_variables = {
        "prompt": prompt,
        "identity": "You are an intelligent assistant who thinks thoroughly through all the context to best respond to a prompt. "
        + persona,
        "database_names": "Top Google Search Result, General Knowledge",
        "databases": "Top Google Search Result - The top google search result is the best source for live information and can have any other relevant information.\n"
        "General Knowledge - Well-known facts and information that is generally known to the public.",
    }

    variables, history = respond_flow(
        input_variables, chat_llm=chat_llm, input_chat_history=chat_history
    )

    prompt = variables["prompt"]
    response = variables["response"]

    if print_prompt:
        print("\nPrompt:", prompt)

    print("\nTop Possible Sources:")
    for ndx, source in enumerate(variables["sources"]):
        print(f"{ndx + 1}. {source[0]}")

    print("\nResponse:", response)

    return response, history


def update_memory_and_chat_history(
    memory,
    encoder,
    memory_file,
    max_chat_history_tokens,
    max_num_tokens_per_memory,
    chat_history,
    query,
    response,
):
    """
    Asks the user if they want to skip adding prompt/response pair to memory
    and if they want to clear all prompt/response history. Then does so.
    """
    if input("\nSkip adding prompt/response pair to memory? (y/n): ") == "y":
        if input("\nClear all prompt/response history? (y/n): ") == "y":
            chat_history = ChatHistory()
            memory.load(memory_file)
    else:
        # Update history and add prompt response pair to memory for future possible queries
        update_memory(memory, encoder, max_num_tokens_per_memory, query, response)

        chat_history = handle_long_chat_history(
            encoder, chat_history, max_chat_history_tokens
        )

    return chat_history


def update_memory(memory, encoder, max_num_tokens_per_memory, query, response):
    """
    Adds prompt/response pairs to memory.
    """
    num_query_tokens = len(encoder.encode(query))
    num_response_tokens = len(encoder.encode(response))
    cushion = 10

    if num_query_tokens + num_response_tokens + cushion > max_num_tokens_per_memory:
        handle_large_memory_entries(
            memory,
            encoder,
            query,
            response,
            num_query_tokens,
            num_response_tokens,
            max_num_tokens_per_memory,
            cushion,
        )
    else:
        memory.add(
            {
                "text": f"Prompt: {query}\nResponse: {response}",
                "metadata": f"source: Past Prompt & Response",
            }
        )


def handle_large_memory_entries(
    memory,
    encoder,
    query,
    response,
    num_query_tokens,
    num_response_tokens,
    max_num_tokens_per_memory,
    cushion,
):
    """
    Adds long past prompt/response pairs to the memory.
    This done by attempting to chunk the respective long prompt or
    response while still including the entire smaller portion.
    However, if both the prompt and response are too long, then
    both are chunked and added separately.
    """
    if num_query_tokens < max_num_tokens_per_memory / 3:
        for response_part in smart_chunk(
            response,
            encoder,
            max_num_tokens_per_memory - num_query_tokens - cushion,
            1 / 3,
        ):
            memory.add(
                {
                    "text": f"Prompt: {query}\nResponse: {response_part}",
                    "metadata": f"source: Past Prompt & Partial Response",
                }
            )
    elif num_response_tokens < max_num_tokens_per_memory / 3:
        for query_part in smart_chunk(
            query,
            encoder,
            max_num_tokens_per_memory - num_response_tokens - cushion,
            1 / 3,
        ):
            memory.add(
                {
                    "text": f"Prompt: {query_part}\nResponse: {response}",
                    "metadata": f"source: Past Partial Prompt & Response",
                }
            )
    else:
        for query_part in smart_chunk(
            query, encoder, max_num_tokens_per_memory - cushion, 1 / 3
        ):
            memory.add(
                {
                    "text": f"Prompt: {query_part}",
                    "metadata": f"source: Past Prompt",
                }
            )

        for response_part in smart_chunk(
            response, encoder, max_num_tokens_per_memory - cushion, 1 / 3
        ):
            memory.add(
                {
                    "text": f"Response: {response_part}",
                    "metadata": f"source: Past Response",
                }
            )


def handle_long_chat_history(encoder, chat_history, max_chat_history_tokens):
    """
    Truncates chat history to be less than max_chat_history_tokens.
    """
    # Improve by also adding summarization
    num_tokens = 0
    shortened_messages = []
    for message in chat_history.messages:
        num_tokens += len(encoder.encode(message.content))
        if num_tokens > max_chat_history_tokens:
            break
        shortened_messages.append(message)
    return ChatHistory(shortened_messages)


def chunk(text, encoder, chunk_size, overlap_factor):
    """
    Chunks text into chunks of size chunk_size (token units) with overlap.
    """
    overlap = int(chunk_size * overlap_factor)
    tokens = encoder.encode(text)

    chunked_tokens = [
        tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size - overlap)
    ]

    return [encoder.decode(chunk) for chunk in chunked_tokens]


def smart_chunk(text, encoder, chunk_size, overlap_factor):
    """
    Chunks text into chunks of size chunk_size with overlap, but
    will first try to split on '\n\n' or more '\n's without overlap.
    """
    split_text = re.split(r"\n{2,}", text)  # Split text on '\n\n' or more '\n's
    result = []

    for substring in split_text:
        if len(substring) > 0:
            result.extend(chunk(substring, encoder, chunk_size, overlap_factor))

    return result


def get_input(prompt_text="\nPrompt: "):
    """
    Get output from user. If no input, return None.
    Output can be multi-line with [LONG] and [END].

    :return: prompt
    """
    prompt = input(
        "\nUse [LONG] to add multiple lines and [END] to end inputting. " + prompt_text
    )

    if prompt.startswith("[LONG]"):
        prompt = prompt[6:]
        while True:
            prompt += "\n" + input()

            if prompt.endswith("[END]"):
                prompt = prompt[:-5]
                break

    if len(prompt) == 0:
        return None
    return prompt


def create_respond_flow(
    memory,
    encoder,
    max_num_memory_queries,
    combine_threshold,
    max_num_databases,
    max_num_docs,
    memory_score_threshold,
    max_num_context_tokens,
    only_use_memory,
    verbose,
):
    """
    Creates a full respond flow.


    :returns: a chat flow object that takes ("prompt", "identity", "database_names", "databaes")
              and outputs ("response", "query", "document", "relevancy", "context", "sources", "memory")
    """
    # Define the individual flows
    contextualize_flow = create_contextualize_flow()
    memory_flow = create_memory_flow(
        contextualize_flow,
        max_num_memory_queries,
        combine_threshold,
        memory,
    )
    document_generation_flow = create_document_generation_flow()
    relevancy_flow = create_relevancy_flow()
    prepare_context_flow = create_prepare_context_flow(
        only_use_memory,
        max_num_databases,
        max_num_docs,
        memory_score_threshold,
        max_num_context_tokens,
        encoder,
    )
    answer_flow = create_answer_flow()

    # Only memory_flow gets past histories, which only come from the full history of each answer_flow
    # This means answer_flow itself does not get such histories.
    respond_flow = SequentialChatFlows(
        [
            NoHistory(memory_flow, allow_input_history=True)
            if only_use_memory
            else NoHistory(
                ConcurrentChatFlows(
                    [memory_flow, document_generation_flow],
                    max_workers=2,
                ), allow_input_history=True
            ),
            NoHistory(prepare_context_flow)
            if only_use_memory
            else NoHistory(SequentialChatFlows([relevancy_flow, prepare_context_flow])),
            answer_flow,
        ],
        verbose=verbose,
    )

    return respond_flow


def create_contextualize_flow():
    """
    Creates a prompt standalonizer chat flow.

    :return: a chat flow object that takes ("prompt") and outputs ("query")
    """
    return ChatFlow.from_dicts(
        [
            {
                Role.SYSTEM: "You take context from all the conversations to make a standalone prompt."
            },
            {
                Role.USER: "Prompt: {prompt}\n\n"
                "Create a standalone prompt of the prompt above using all of the context from the previous conversations above. Do this by trying to fill-in all the known what, where, who, when, and why's. Do not respond to the prompt, but only rewrite it using any relevant information found in the previous conversations. The standalone prompt at the core should ultimately be the same as the original prompt.\n\n"
                "Your response should have no explanations at all and should always look like this:\n"
                "Standalone Prompt: the standalone prompt\n\n"
                "Standalone Prompt: "
            },
            {Role.ASSISTANT: "{query}"},
        ],
        verbose=True,
    )


def create_memory_flow(
    contextualize_flow, max_num_memory_queries, combine_threshold, memory
):
    """
    Create the memory chat flow.

    :return: a chat flow object that takes ("query") and outputs ("memory")
    """
    return MemoryChatFlow(
        contextualize_flow,
        memory,
        memory_query_kwargs={
            "k": max_num_memory_queries,
            "combine_threshold": combine_threshold,
        },
        verbose=True,
    )


def create_document_generation_flow():
    """
    Create the document generation flow (general knowledge).

    :return: a chat flow object that takes ("prompt") and outputs ("document")
    """
    return ChatFlow.from_dicts(
        [
            {
                Role.SYSTEM: "You are acting as a response/document/code generator named HyDE."
            },
            {
                Role.USER: "Prompt: {prompt}  (write a full detailed response to respond; avoid AI Language model fluff)"
            },
            {Role.ASSISTANT: "{document}"},
        ],
        verbose=True,
    )


def create_relevancy_flow():
    """
    Create the flow for rating relevancy of databases.

    :return: a chat flow object that takes ("database_names", "prompt", "databases") and outputs ("relevancy")
    """
    return ChatFlow.from_dicts(
        [
            {
                Role.SYSTEM: "You are a live data warehouse with access to many different databases. You rate the relevancy of each database for responding to a prompt."
            },
            {
                Role.USER: "Among your databases ({database_names}), which ones contain information relevant to the prompt below? Use the previous conversation for context.\n\n"
                "Prompt: {prompt}\n\n"
                "Available Databases with their descriptions:\n"
                "{databases}\n\n"
                "Only respond using the following format:\n"
                "Database Name - rating Not Relevant / Slightly Relevant / Relevant / Highly Relevant"
            },
            {Role.ASSISTANT: "{relevancy}"},
        ],
        verbose=True,
    )


def create_prepare_context_flow(
    only_use_memory,
    max_num_databases,
    max_num_docs,
    memory_score_threshold,
    max_num_context_tokens,
    encoder,
):
    """
    Create the flow for gathering context for a prompt.

    Steps:
    1. Gather context from memory using a standalone query based off of the prompt
    2. Gather context from relevant databases (Google, General Knowledge etc.)
    3. Sort and Remove excess documents
    4. Create a single context string containing all the documents upto a token limit

    :return: a chat flow object that takes ("memory", "relevancy", "document", "database_names", "query")
             and outputs ("context", "sources")
    """
    google_search = GoogleSearchTool(
        os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID")
    )

    def prepare_context_func(variables, chat_llm, input_chat_history):
        nonlocal google_search, max_num_databases, max_num_docs, only_use_memory, memory_score_threshold

        all_docs = []

        # Add memories to all_docs
        memories = variables["memory"]
        query = variables["query"]
        for memory in memories:
            if memory["score"] < memory_score_threshold:
                all_docs.append(
                    (
                        "Memory".upper(),
                        memory["text"],
                        memory["metadata"].replace("source: ", ""),
                        memory["score"],
                    )
                )

        # Rate the relevancy of databases and, if relevant, add query results to all_docs
        relevancy = variables.get("relevancy")
        if relevancy is not None and not only_use_memory:
            document = variables["document"]
            names = variables["database_names"].lower().split(", ")

            ratings = []
            possible_ratings = [
                "not relevant",
                "slightly relevant",
                "relevant",
                "highly relevant",
            ]
            for database in relevancy.split("\n"):
                name, rating = database.lower().split(" - ", 1)
                if rating in possible_ratings and name in names:
                    rating = possible_ratings.index(rating.strip())
                    ratings.append((name, rating))

            ratings = dict(
                sorted(ratings, key=lambda x: x[1], reverse=True)[:max_num_databases]
            )

            if (
                "top google search result" in ratings
                and ratings["top google search result"] > 1
            ):
                search_result = google_search.use({"query": query})
                # Scores could be actually calcualted if embedded the results and compared
                all_docs.append(
                    ("top google search result".upper(), search_result, "Google", 0.45)
                )

            if "general knowledge" in ratings and ratings["general knowledge"] > 1:
                all_docs.append(
                    (
                        "general knowledge".upper(),
                        document,
                        "GPT General Knowledge",
                        0.46,
                    )
                )

            # add code here for other databases

        all_docs = sorted(all_docs, key=lambda x: x[3], reverse=False)[:max_num_docs]

        current_name = None
        text = ""
        num_tokens = 0
        sources = {}
        for name, doc, source, score in all_docs:
            text_seg = f"\n{name}:\n{doc}\n" if name != current_name else doc + "\n"
            current_name = name
            num_tokens += len(encoder.encode(text_seg))

            if num_tokens > max_num_context_tokens:
                break

            text += text_seg
            sources[source] = min(sources.get(source, score), score)

        sources = sorted(sources.items(), key=lambda x: x[1], reverse=False)

        return {"context": text, "sources": sources}, ([input_chat_history], [])

    return FuncChatFlow(
        prepare_context_func,
        input_varnames=set(["memory"])
        if only_use_memory
        else set(["memory", "relevancy", "document", "database_names", "query"]),
        output_varnames=set(["context", "sources"]),
    )


def create_answer_flow():
    """
    Create the flow for responding to a prompt using previous gathered context from a query.

    :return: The flow object
    """
    return ChatFlow.from_dicts(
        [
            {Role.SYSTEM: "{identity}"},
            {Role.USER: "Give me context to the prompt below:\n{query}"},
            {
                Role.ASSISTANT: "Here is some context related to the prompt:\n{context}",
                "type": "input",
            },
            {
                Role.USER: "Using the context you provided and your general knowledge, respond directly and accurately to the prompt below: (prefer specific knowledge over general in the context)\n"
                "Prompt: {prompt}"
            },
            {Role.ASSISTANT: "{response}"},
        ],
        verbose=True,
    )


def get_args():
    """
    Generate argparse object and parse command-line arguments.

    :return: An argparse object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Chatbot with global variables as parameters."
    )
    parser.add_argument(
        "--only_use_memory", action="store_true", help="Only use memory for context."
    )
    parser.add_argument(
        "--max_num_databases",
        type=int,
        default=3,
        help="Maximum number of databases for context.",
    )
    parser.add_argument(
        "--max_num_docs",
        type=int,
        default=8,
        help="Maximum number of documents for context.",
    )
    parser.add_argument(
        "--max_num_memory_queries",
        type=int,
        default=10,
        help="Maximum number of memory queries.",
    )
    parser.add_argument(
        "--max_num_context_tokens",
        type=int,
        default=1500,
        help="Maximum number of tokens for the full context.",
    )
    parser.add_argument(
        "--memory_score_threshold",
        type=float,
        default=0.6,
        help="Threshold for memory queries. A value of .5 is strict and .7 is loose.",
    )
    parser.add_argument(
        "--combine_threshold",
        type=float,
        default=0.1,
        help="Threshold for combining memory queries.",
    )
    parser.add_argument(
        "--memory_file",
        type=str,
        default="memory_default.pkl",
        help="Memory file to use.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity."
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Enables queryable chat history so prompts can refer to previous prompts and responses.",
    )
    parser.add_argument(
        "--max_chat_history_tokens",
        type=int,
        default=2000,
        help="Number of tokens chat history can have. Any more will be truncated.",
    )
    parser.add_argument(
        "--openai_chat_model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI chat model to use.",
    )
    parser.add_argument(
        "--max_num_prompt_tokens",
        type=int,
        default=2000,
        help="Number of tokens a prompt can contain without having to break it up.",
    )
    parser.add_argument(
        "--max_num_tokens_per_memory",
        type=int,
        default=500,
        help="Number of tokens a prompt can contain without having to break it up.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for response generation.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="",
        help="Persona to use for response generation. The persona is one aspect in a response and so may not be stressed.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
