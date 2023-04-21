import os
import argparse

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


def main(args):
    # Define the individual flows
    contextualize_flow = create_contextualize_flow()
    memory_flow = create_memory_flow(
        contextualize_flow,
        args.max_num_memory_queries,
        args.combine_threshold,
        args.memory_file,
    )
    document_generation_flow = create_document_generation_flow()
    relevancy_flow = create_relevancy_flow()
    prepare_context_flow = create_prepare_context_flow(
        args.only_use_memory, args.max_num_databases, args.max_num_docs
    )
    answer_flow = create_answer_flow()

    # Combine the flows into the main flow
    main_flow = SequentialChatFlows(
        [
            NoHistory(memory_flow)
            if args.only_use_memory
            else NoHistory(
                ConcurrentChatFlows(
                    [memory_flow, document_generation_flow], max_workers=2
                )
            ),
            NoHistory(prepare_context_flow)
            if args.only_use_memory
            else NoHistory(SequentialChatFlows([relevancy_flow, prepare_context_flow])),
            answer_flow,
        ],
        verbose=args.verbose,
    )

    chat_llm = ChatLLM(gpt_model=args.openai_chat_model, temperature=args.temperature)

    # Main loop for receiving prompts and generating responses
    # Chat History for each question is not shared in this loop. Each prompt must be standalone.
    # However, everything exists above to easily change the below loop to make prompts dependent on previous prompts.
    while True:
        prompt = input("\nPrompt: ")

        input_variables = {
            "prompt": prompt,
            "identity": "You are an intelligent assistant who thinks thoroughly through all the context to best respond to a prompt.",
            "database_names": "Top Google Search Result, General Knowledge",
            "databases": "Top Google Search Result - The top google search result is the best source for live information and can have any other relevant information.\n"
            "General Knowledge - Well-known facts and information that is generally known to the public.",
        }

        variables, history = main_flow(input_variables, chat_llm=chat_llm)

        if args.verbose:
            print("\n\nExtracted Variables:")
            for key, value in variables.items():
                print(f"{key}: {value}\n")
            print("\n\nChat History:")
            for message in history.messages:
                print(f"{message.role.title()}: {message.content}")

        print("\nTop Possible Sources:")
        for ndx, source in enumerate(variables["sources"]):
            print(f"{ndx + 1}. {source[0]}")

        print("\nResponse:", variables["response"])


def create_contextualize_flow():
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
    contextualize_flow, max_num_memory_queries, combine_threshold, memory_file
):
    return MemoryChatFlow(
        contextualize_flow,
        Memory(memory_file),
        memory_query_kwargs={
            "k": max_num_memory_queries,
            "combine_threshold": combine_threshold,
        },
    )


def create_document_generation_flow():
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


def create_prepare_context_flow(only_use_memory, max_num_databases, max_num_docs):
    google_search = GoogleSearchTool(
        os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID")
    )

    def prepare_context_func(variables, chat_llm, input_chat_history):
        nonlocal google_search, max_num_databases, max_num_docs, only_use_memory

        all_docs = []

        memories = variables["memory"]
        for memory in memories:
            all_docs.append(
                (
                    "Memory",
                    memory["text"],
                    memory["metadata"].replace("source: ", ""),
                    memory["score"],
                )
            )

        relevancy = variables.get("relevancy")
        if relevancy is not None and not only_use_memory:
            document = variables["document"]
            query = variables["query"]
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
        sources = {}
        for name, doc, source, score in all_docs:
            if name == current_name:
                text += doc + "\n"
            else:
                current_name = name
                text += f"\n{name}:\n{doc}\n"

            if source in sources:
                sources[source] = min(sources[source], score)
            else:
                sources[source] = score

        sources = sorted(sources.items(), key=lambda x: x[1], reverse=False)

        return {"context": text, "sources": sources}, ([], [])

    return FuncChatFlow(
        prepare_context_func,
        input_varnames=set(["memory"])
        if only_use_memory
        else set(["memory", "relevancy", "document", "database_names", "query"]),
        output_varnames=set(["context", "sources"]),
    )


def create_answer_flow():
    return ChatFlow.from_dicts(
        [
            {Role.SYSTEM: "{identity}"},
            {Role.USER: "Give me context to the prompt below:\n{prompt}"},
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chatbot with global variables as parameters."
    )
    parser.add_argument(
        "--only_use_memory", action="store_true", help="Only use memory for context."
    )
    parser.add_argument(
        "--max_num_databases", type=int, default=3, help="Maximum number of databases."
    )
    parser.add_argument(
        "--max_num_docs", type=int, default=5, help="Maximum number of documents."
    )
    parser.add_argument(
        "--max_num_memory_queries",
        type=int,
        default=8,
        help="Maximum number of memory queries.",
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
        "--openai_chat_model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI chat model to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for response generation.",
    )

    args = parser.parse_args()
    main(args)
