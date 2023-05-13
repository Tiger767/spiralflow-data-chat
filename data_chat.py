"""
Author: Travis Hammond
Version: 5_4_2023
"""

import os
import concurrent.futures
import argparse
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
from spiralflow.chunking import SmartChunker


class DataChat:
    def __init__(
        self,
        only_use_memory=False,
        max_num_databases=3,
        max_num_docs=10,
        max_num_memory_queries=14,
        max_num_context_tokens=1500,
        memory_score_threshold=0.7,
        combine_threshold=0.1,
        summarize_context=False,
        memory_file="memory_default.pkl",
        history=False,
        max_chat_history_tokens=2000,
        openai_chat_model="gpt-3.5-turbo",
        max_num_prompt_tokens=2000,
        max_num_tokens_per_memory=500,
        temperature=0.3,
        persona="",
        memory=None,
        verbose=False,
    ):
        self.only_use_memory = only_use_memory
        self.max_num_databases = max_num_databases
        self.max_num_docs = max_num_docs
        self.max_num_memory_queries = max_num_memory_queries
        self.max_num_context_tokens = max_num_context_tokens
        self.memory_score_threshold = memory_score_threshold
        self.combine_threshold = combine_threshold
        self.summarize_context = summarize_context
        self.memory_file = memory_file
        self.history = history
        self.max_chat_history_tokens = max_chat_history_tokens
        self.openai_chat_model = openai_chat_model
        self.max_num_prompt_tokens = max_num_prompt_tokens
        self.max_num_tokens_per_memory = max_num_tokens_per_memory
        self.temperature = temperature
        self.persona = persona
        self.verbose = verbose

        if memory is None:
            self.memory = Memory(memory_file)
        else:
            self.memory = memory
        self.encoder = tiktoken.encoding_for_model(
            "text-embedding-ada-002"
        )  # assuming same as openai_chat_model

        self.create_respond_flow()
        self.create_assumption_clarification_flow()

        self.chat_llm = ChatLLM(
            gpt_model=self.openai_chat_model, temperature=self.temperature
        )

        self.chat_history = ChatHistory() if history else None

    def process_multiple_prompts(self, prompts):
        """
        Process multiple prompts, going through the respond_flow.
        Add the prompts and responses to the memory if using history.
        """
        num_tokens = max([len(self.encoder.encode(prompt)) for prompt in prompts])
        if num_tokens > self.max_num_prompt_tokens:
            raise ValueError(
                f"A prompt is too long: {num_tokens} > {self.max_num_prompt_tokens}"
            )
        response_list, sources_list, history_list = self.generate_responses(prompts)

        total_tokens_list = []
        for history in history_list:
            total_tokens_list.append(0)
            for message in history.messages:
                total_tokens_list[-1] += len(self.encoder.encode(message.content))

        if self.chat_history is not None:
            # Since concurrent prompts, wont actually add to history

            for prompt, response in zip(prompts, response_list):
                self.update_memory(prompt, response)

        return response_list, sources_list, total_tokens_list

    def process_single_prompt(self, prompt, instruction=""):
        num_tokens = len(self.encoder.encode(prompt))
        if num_tokens > self.max_num_prompt_tokens:
            raise ValueError(
                f"Prompt too long: {num_tokens} > {self.max_num_prompt_tokens}"
            )

        response, sources, history = self.generate_responses(
            [prompt], instruction=instruction
        )
        response, sources, history = response[0], sources[0], history[0]

        total_tokens = 0
        for message in history.messages:
            print(f'\n"{message.content}"\n')
            total_tokens += len(self.encoder.encode(message.content))

        if self.chat_history is not None:
            self.chat_history = history
            self.handle_long_chat_history()

            self.update_memory(prompt, response)
        return response, sources, total_tokens

    def process_long_text_chunk(self, prompt, long_text):
        """
        Process a single prompt through a large amount of text, going through the respond_flow.
        Adds the full prompts and responses to the memory if using history.
        """
        prompt = "\n\n----\n\nFOR THE ABOVE DO THE FOLLOWING:\n" + prompt
        chunked_text = SmartChunker(
            self.encoder,
            self.max_num_prompt_tokens - len(self.encoder.encode(prompt)),
            1 / 4,
        ).chunk(long_text)
        prompts = [chunked_text + prompt for chunked_text in chunked_text]

        num_tokens = max([len(self.encode(prompt)) for prompt in prompts])
        if num_tokens > self.max_num_prompt_tokens:
            raise ValueError(
                f"A prompt+chunk is too long: {num_tokens} > {self.max_num_prompt_tokens}"
            )

        response_list, sources_list, history_list = self.generate_responses(prompts)

        total_tokens_list = []
        for history in history_list:
            total_tokens_list.append(0)
            for message in history.messages:
                total_tokens_list[-1] += len(self.encoder.encode(message.content))

        if self.chat_history is not None:
            # Since concurrent prompts and long text, wont actually add to history

            for prompt, response in zip(prompts, response_list):
                self.update_memory(prompt, response)

        return response_list, sources_list, total_tokens_list

    def clarify_prompt(self, prompt):
        variables, _ = self.ac_flow(
            {"prompt": prompt},
            chat_llm=self.chat_llm,
            input_chat_history=self.chat_history,
        )
        return {
            "clarifications": variables["clarifications"],
            "assumptions": variables["assumptions"],
        }

    def generate_responses(self, prompts, instruction=""):
        """
        Generates responses to multiple prompts by concurrently running it through the respond_flow.
        """
        input_variables_list = []
        for prompt in prompts:
            input_variables = {
                "prompt": prompt,
                "identity": "You are an intelligent assistant who thinks thoroughly through all the context to best respond to a prompt. "
                + self.persona,
                "database_names": "Top Google Search Result, General Knowledge",
                "databases": "Top Google Search Result - The top google search result is the best source for live information and can have any other relevant information.\n"
                "General Knowledge - Well-known facts and information that is generally known to the public.",
                "instruction": instruction,
            }
            input_variables_list.append(input_variables)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            responses_histories = list(
                executor.map(
                    lambda input_variables: self.respond_flow(
                        input_variables,
                        chat_llm=self.chat_llm,
                        input_chat_history=self.chat_history,
                    ),
                    input_variables_list,
                )
            )

        response_list, sources_list, history_list = [], [], []
        for variables, history in responses_histories:
            prompt = variables["prompt"]
            response = variables["response"]
            sources = variables["sources"]
            response_list.append(response)
            sources_list.append(sources)
            history_list.append(history)

        return response_list, sources_list, history_list

    def clear_history(self):
        if self.chat_history is not None:
            self.chat_history = ChatHistory()

    def clear_memory(self):
        self.memory.load(self.memory_file)

    def handle_long_chat_history(self):
        """
        Truncates chat history to be less than max_chat_history_tokens.
        """
        # Improve by also adding summarization
        num_tokens = 0
        shortened_messages = []
        for message in self.chat_history.messages:
            num_tokens += len(self.encoder.encode(message.content))
            if num_tokens > self.max_chat_history_tokens:
                break
            shortened_messages.append(message)
        return ChatHistory(shortened_messages)

    def update_memory(self, query, response):
        """
        Adds prompt/response pairs to memory.
        """
        num_query_tokens = len(self.encoder.encode(query))
        num_response_tokens = len(self.encoder.encode(response))
        cushion = 10

        if (
            num_query_tokens + num_response_tokens + cushion
            > self.max_num_tokens_per_memory
        ):
            self.handle_large_memory_entries(
                query,
                response,
                cushion,
            )
        else:
            self.memory.add(
                {
                    "text": f"Prompt: {query}\nResponse: {response}",
                    "metadata": f"source: Past Prompt & Response",
                }
            )

    def handle_large_memory_entries(
        self,
        query,
        response,
        cushion,
    ):
        """
        Adds long past prompt/response pairs to the memory.
        This done by attempting to chunk the respective long prompt or
        response while still including the entire smaller portion.
        However, if both the prompt and response are too long, then
        both are chunked and added separately.
        """
        if self.num_query_tokens < self.max_num_tokens_per_memory / 3:
            for response_part in SmartChunker(
                self.encoder,
                self.max_num_tokens_per_memory - self.num_query_tokens - cushion,
                1 / 3,
            ).chunk(response):
                self.memory.add(
                    {
                        "text": f"Prompt: {query}\nResponse: {response_part}",
                        "metadata": f"source: Past Prompt & Partial Response",
                    }
                )
        elif self.num_response_tokens < self.max_num_tokens_per_memory / 3:
            for query_part in SmartChunker(
                self.encoder,
                self.max_num_tokens_per_memory - self.num_response_tokens - cushion,
                1 / 3,
            ).chunk(query):
                self.memory.add(
                    {
                        "text": f"Prompt: {query_part}\nResponse: {response}",
                        "metadata": f"source: Past Partial Prompt & Response",
                    }
                )
        else:
            for query_part in SmartChunker(
                self.encoder, self.max_num_tokens_per_memory - cushion, 1 / 3
            ).chunk(query):
                self.memory.add(
                    {
                        "text": f"Prompt: {query_part}",
                        "metadata": f"source: Past Prompt",
                    }
                )

            for response_part in SmartChunker(
                self.encoder, self.max_num_tokens_per_memory - cushion, 1 / 3
            ).chunk(response):
                self.memory.add(
                    {
                        "text": f"Response: {response_part}",
                        "metadata": f"source: Past Response",
                    }
                )

    def create_respond_flow(self):
        """
        Creates a full respond flow.


        :returns: a chat flow object that takes ("prompt", "identity", "database_names", "databaes")
                and outputs ("response", "query", "document", "relevancy", "context", "sources", "memory")
        """
        # Define the individual flows
        self.memory_flow = self.create_memory_flow()
        self.document_generation_flow = self.create_document_generation_flow()
        self.relevancy_flow = self.create_relevancy_flow()
        self.prepare_context_flow = self.create_prepare_context_flow()
        self.answer_flow = self.create_answer_flow()

        # Only memory_flow gets past histories, which only come from the full history of each answer_flow
        # This means answer_flow itself does not get such histories.
        self.respond_flow = SequentialChatFlows(
            [
                NoHistory(self.memory_flow, allow_input_history=True)
                if self.only_use_memory
                else NoHistory(
                    ConcurrentChatFlows(
                        [self.memory_flow, self.document_generation_flow],
                        max_workers=2,
                    ),
                    allow_input_history=True,
                ),
                NoHistory(self.prepare_context_flow)
                if self.only_use_memory
                else NoHistory(
                    SequentialChatFlows(
                        [self.relevancy_flow, self.prepare_context_flow]
                    )
                ),
                self.answer_flow,
            ],
            verbose=self.verbose,
        )

        return self.respond_flow

    def create_contextualize_flow(self):
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
            verbose=self.verbose,
        )

    def create_memory_flow(self):
        """
        Create the memory chat flow.

        :return: a chat flow object that takes ("query") and outputs ("memory")
        """
        return MemoryChatFlow(
            self.create_contextualize_flow(),
            self.memory,
            memory_query_kwargs={
                "k": self.max_num_memory_queries,
                "combine_threshold": self.combine_threshold,
            },
            verbose=self.verbose,
        )

    def create_document_generation_flow(self):
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
            verbose=self.verbose,
        )

    def create_relevancy_flow(self):
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
            verbose=self.verbose,
        )

    def create_prepare_context_flow(self):
        """
        Create the flow for gathering context for a prompt.

        Steps:
        1. Gather context from memory using a standalone query based off of the prompt
        2. Gather context from relevant databases (Google, General Knowledge etc.)
        3. Sort and Remove excess documents
        4. Summarize each document (optional)
        5. Create a single context string containing all the documents upto a token limit

        :return: a chat flow object that takes ("memory", "relevancy", "document", "database_names", "query")
                and outputs ("context", "sources")
        """
        google_search = GoogleSearchTool(
            os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID")
        )

        prompted_summarization_flow = ChatFlow.from_dicts(
            [
                {
                    Role.SYSTEM: "You are a summarizer and prompt responder. If the prompt cannot be responded to directly from the text, you provide a detailed summary of the text as your response."
                },
                {
                    Role.USER: "{text}\n\n----\n\n"
                    "Using the above text, respond to the following prompt:\n\n"
                    "Prompt: {prompt}\n\n"
                    "(if the prompt cannot be answered using the text, summarize the text; do not refer to the prompt or text at all, just directly respond)"
                },
                {Role.ASSISTANT: "{prompted_summarization}"},
            ]
        )

        def prepare_context_func(variables, chat_llm, input_chat_history):
            nonlocal google_search

            all_docs = []

            # Add memories to all_docs
            memories = variables["memory"]
            query = variables["query"]
            for memory in memories:
                if memory["score"] > self.memory_score_threshold:
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
            if relevancy is not None and not self.only_use_memory:
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
                    sorted(ratings, key=lambda x: x[1], reverse=True)[
                        : self.max_num_databases
                    ]
                )

                if (
                    "top google search result" in ratings
                    and ratings["top google search result"] > 1
                    and all_docs[-1][3] < 0.8
                ):
                    search_result = google_search.use({"query": query})
                    # Scores could be actually calcualted if embedded the results and compared
                    all_docs.append(
                        (
                            "top google search result".upper(),
                            search_result,
                            "Google",
                            0.75,
                        )
                    )

                if (
                    "general knowledge" in ratings
                    and ratings["general knowledge"] > 1
                    and all_docs[-1][3] < 0.8
                ):
                    all_docs.append(
                        (
                            "general knowledge".upper(),
                            document,
                            "GPT General Knowledge",
                            0.75,
                        )
                    )

                # add code here for other databases

            all_docs = sorted(all_docs, key=lambda x: x[3], reverse=True)[
                : self.max_num_docs
            ]

            if self.summarize_context:

                def summarize_doc(args):
                    name, doc, source, score = args
                    variables, _ = prompted_summarization_flow(
                        {"text": doc, "prompt": query}, chat_llm=chat_llm
                    )
                    short_doc = variables["prompted_summarization"]

                    # Add logit to not run if already small, if summarizon large than input, dont use??
                    # Improve summarize prompt, sometimes wont summarize if not relevant, has boiler plate,

                    return (name, short_doc, source, score)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    concise_all_docs = list(executor.map(summarize_doc, all_docs))
                all_docs = concise_all_docs

            current_name = None
            text = ""
            num_tokens = 0
            sources = {}
            for name, doc, source, score in all_docs:
                text_seg = f"\n{name}:\n{doc}\n" if name != current_name else doc + "\n"
                # text_seg += f"Source: {source}\n\n"
                current_name = name
                num_tokens += len(self.encoder.encode(text_seg))

                if num_tokens > self.max_num_context_tokens:
                    break

                text += text_seg
                sources[source] = max(sources.get(source, score), score)

            sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)

            return {"context": text, "sources": sources}, ([input_chat_history], [])

        return FuncChatFlow(
            prepare_context_func,
            input_varnames=set(["memory"])
            if self.only_use_memory
            else set(["memory", "relevancy", "document", "database_names", "query"]),
            output_varnames=set(["context", "sources"]),
        )

    def create_answer_flow(self):
        f"""
        Create the flow for responding to a prompt using previous gathered context from a query.

        :return: a chat flow that takes ("identity", "query", "context", "prompt", "instruction") and outputs ("response")
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
                    Role.USER: "Using the context you provided and your general knowledge, respond directly and accurately to the prompt below: (prefer specific knowledge over general in the context;{instruction})\n"
                    "Prompt: {prompt}"
                },
                {Role.ASSISTANT: "{response}"},
            ],
            verbose=self.verbose,
        )

    def create_assumption_clarification_flow(self):
        """
        Creates a flow to state possibly needed clarifications and assumptions.

        :return: a chat flow that takes ("prompt") and outputs ("clarifications", "assumptions")
        """
        clarity_request_flow = ChatFlow.from_dicts(
            [
                {
                    Role.SYSTEM: "You are acting as a diligent critic and clarity requester. You want the prompt to have enough information to reliably respond."
                },
                {
                    Role.USER: "Before you actually respond to the last prompt below, is there anything that would need to be clarified? Use the previous conversation for context. The prompt is addressed to you, do not respond to the prompt or explain, but only ask for clarification if any is needed.\n\n"
                    "Prompt: {prompt}\n\n"
                    "Only respond using the following format:\n"
                    "Clarifications:\n"
                    " - bullet list all clarifications that you would need to more concisely respond to the prompt without making any assumptions. List at least 1."
                },
                {Role.ASSISTANT: "{clarifications}"},
            ],
            verbose=self.verbose,
        )

        assumption_request_flow = ChatFlow.from_dicts(
            [
                {
                    Role.SYSTEM: "You are acting as a diligent critic and assumption identifier. You want to outline all your assumptions and unknowns before responding."
                },
                {
                    Role.USER: "Before you actually respond to the last prompt below, is there anything that is unknown or being assumed? Use the previous conversation for context. The prompt is addressed to you, but do not respond to the prompt or explain, but only list assumptions.\n\n"
                    "Prompt: {prompt}\n\n"
                    "Only respond using the following format:\n"
                    "Assumptions:\n"
                    " - bullet list all assumptions and resolves to them that you are making in order to respond to the prompt. List at least 1."
                },
                {Role.ASSISTANT: "{assumptions}"},
            ],
            verbose=self.verbose,
        )

        self.ac_flow = ConcurrentChatFlows(
            [clarity_request_flow, assumption_request_flow], verbose=self.verbose
        )

        return self.ac_flow

    def run(self):
        def get_input(prompt_text="\nPrompt: "):
            """
            Get output from user. If no input, return None.
            Output can be multi-line with [LONG] and [END].

            :return: prompt
            """
            prompt = input(
                "\nUse [LONG] to add multiple lines and [END] to end inputting. "
                + prompt_text
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

        while True:
            # Check for concurrent processing of prompts
            if (
                input("\nSubmit multiple prompts for concurrent processing? (y/n): ")
                == "y"
            ):
                print(
                    "Input exactly `[END]` with nothing else in a prompt to finish entering prompts."
                )
                prompts = []
                while True:
                    prompt = get_input()
                    if prompt == "[END]":
                        break
                    prompts.append(prompt)

                print(f"\nResponding now.")

                (
                    response_list,
                    sources_list,
                    num_tokens_list,
                ) = self.process_multiple_prompts(prompts)

                for prompt, response, sources, num_tokens in zip(
                    prompts, response_list, sources_list, num_tokens_list
                ):
                    print("\nPrompt:", prompt)

                    print("\nTop Possible Sources:")
                    for ndx, source in enumerate(sources):
                        print(f"{ndx + 1}. {source[0]}")

                    print("\nResponse:", response)
                    print(f"\nNumber of tokens: {num_tokens / 1000:.1f}K")
                    print()

            # Check for long text chunking and processing
            elif (
                input(
                    "\nSubmit a prompt with a long text to chunk and respond? (y/n): "
                )
                == "y"
            ):
                prompt = get_input()
                if prompt is None:
                    continue
                long_text = get_input("\nLong Text: ")

                print(f"\nResponding now.")

                response_list, sources_list = self.process_long_text_chunk(
                    prompt, long_text
                )

                for response, sources, num_tokens in zip(
                    response_list, sources_list, num_tokens_list
                ):
                    print("\nTop Possible Sources:")
                    for ndx, source in enumerate(sources):
                        print(f"{ndx + 1}. {source[0]}")

                    print("\nResponse:", response)
                    print(f"\nNumber of tokens: {num_tokens / 1000:.1f}K")
                    print()

            # Process single prompt
            else:
                prompt = get_input()
                if prompt is None:
                    continue
                instruction = input("Extra Instructions: ").strip()
                if instruction != "":
                    instruction = f" {instruction}"

                print(f"\nResponding now.")

                response, sources, num_tokens = self.process_single_prompt(
                    prompt, instruction=instruction
                )

                print("\nTop Possible Sources:")
                for ndx, source in enumerate(sources):
                    print(f"{ndx + 1}. {source[0]}")

                print("\nResponse:", response)
                print(f"\nNumber of tokens: {num_tokens / 1000:.1f}K")


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
        default=0.7,
        help="Threshold for memory queries. A value of .8 is strict and .7 is loose.",
    )
    parser.add_argument(
        "--combine_threshold",
        type=float,
        default=0.1,
        help="Threshold for combining memory queries.",
    )
    parser.add_argument(
        "--summarize_context",
        action="store_true",
        help="Each document in context will be summarized, attempting to extract the relevant parts.",
    )
    parser.add_argument(
        "--memory_file",
        type=str,
        default="data/memory_default.pkl",
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
    dc = DataChat(**vars(get_args()))
    dc.run()
