"""
Author: Travis Hammond
Version: 5_13_2023
"""

import os
import concurrent.futures
import argparse
import tiktoken
from typing import List, Optional, Tuple, Dict, Any, Union
from copy import deepcopy

from spiralflow.message import Role
from spiralflow.chat_llm import ChatLLM
from spiralflow.flow import (
    ChatFlow,
    SequentialChatFlows,
    FuncChatFlow,
    NoHistory,
    MemoryChatFlow,
)
from spiralflow.memory import Memory
from spiralflow.chat_history import ChatHistory
from spiralflow.chunking import SmartChunker


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


class Chatbot:
    def __init__(
        self,
        openai_chat_model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        persona: str = "You are a helpful assistant.",
        max_num_prompt_tokens: int = 2000,
        enable_chat_history: bool = False,
        max_chat_history_tokens: int = 2000,
        verbose: bool = False,
    ):
        """
        :param openai_chat_model: the model to use from OpenAI's chat models
        :param temperature: the randomness of the model's responses
        :param persona: the persona that the chatbot should adopt
        :param max_num_prompt_tokens: the maximum number of tokens that a single prompt can have
        :param enable_chat_history: whether or not to keep a chat history
        :param max_chat_history_tokens: the maximum number of tokens that the chat history can have
        :param verbose: whether to print verbose output or not
        """
        self.encoder = tiktoken.encoding_for_model("text-embedding-ada-002")

        self.settings = {
            "openai_chat_model": openai_chat_model,
            "temperature": temperature,
            "persona": persona,
            "max_num_prompt_tokens": max_num_prompt_tokens,
            "enable_chat_history": enable_chat_history,
            "max_chat_history_tokens": max_chat_history_tokens,
            "verbose": verbose,
        }

    def chat(
        self,
        prompts: Union[str, List[str]],
        chat_history: Optional[ChatHistory] = None,
        split_long_prompt: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[List[str], List[ChatHistory], List[int], Optional[ChatHistory]]:
        """
        :param prompts: a list of prompts to be processed by the chatbot
        :param chat_history: existing chat history, if any
        :param split_long_prompt: flag to split long prompts
        :param kwargs: additional parameters to override default settings
        :return: a tuple of lists with responses, history of each chat, total tokens in each chat, and updated chat history object
        """
        if not isinstance(prompts, list):
            prompts = [prompts]

        if split_long_prompt and (len(prompts) != 1):
            raise ValueError(
                f"If split_long_prompt is provided, prompt must be a list of length 1."
            )

        settings = self._update_settings(kwargs)

        if split_long_prompt:
            prompts = self._handle_long_text(settings, split_long_prompt, prompts[0])

        self._check_prompts_tokens(settings, prompts)

        chat_llm = ChatLLM(
            gpt_model=settings["openai_chat_model"], temperature=settings["temperature"]
        )

        chat_history = self._get_history(settings, chat_history)

        variables_list = [{"prompt": prompt} for prompt in prompts]

        response_list, history_list = self.generate_responses(
            settings, chat_llm, chat_history, variables_list
        )

        total_tokens_list = self._get_history_num_tokens(history_list)

        if settings["enable_chat_history"]:
            # Cannot handle when there are multiple prompts / histories
            if len(history_list) == 1:
                chat_history = self._handle_long_chat_history(settings, history_list[0])

        return response_list, history_list, total_tokens_list, chat_history

    def _get_history(
        self, settings: Dict[str, Any], chat_history: Optional[ChatHistory]
    ) -> Optional[ChatHistory]:
        """
        :param settings: chatbot settings
        :param chat_history: existing chat history, if any
        :return: updated chat history
        """
        if chat_history is None:
            chat_history = ChatHistory() if settings["enable_chat_history"] else None
        else:
            chat_history = deepcopy(chat_history)
            settings["enable_chat_history"] = True
        return chat_history

    def _check_prompts_tokens(
        self, settings: Dict[str, Any], prompts: List[str]
    ) -> None:
        """
        :param settings: chatbot settings
        :param prompts: a list of prompts
        """
        num_tokens = max([len(self.encoder.encode(prompt)) for prompt in prompts])
        if num_tokens > settings["max_num_prompt_tokens"]:
            raise ValueError(
                f"A prompt is too long: {num_tokens} > {settings['max_num_prompt_tokens']}"
            )

    def _update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param new_settings: a dictionary of new settings to update
        :return: updated settings
        """
        settings = dict(self.settings)
        if len(new_settings) > 0:
            for key, value in new_settings.items():
                if key in settings:
                    settings[key] = value
                else:
                    raise ValueError(f"Unknown setting: {key}")
        return settings

    def _get_history_num_tokens(self, history_list: List[ChatHistory]) -> List[int]:
        """
        :param history_list: a list of chat histories
        :return: a list of total tokens in each chat history
        """
        return [
            sum(
                len(self.encoder.encode(message.content))
                for message in history.messages
            )
            for history in history_list
        ]

    def _handle_long_chat_history(
        self, settings: Dict[str, Any], chat_history: ChatHistory
    ) -> ChatHistory:
        """
        Truncates chat history to be less than max_chat_history_tokens.

        :param settings: chatbot settings
        :param chat_history: existing chat history
        :return: truncated chat history
        """
        # Improve by also adding summarization
        num_tokens = 0
        shortened_messages = []
        for message in chat_history.messages:
            num_tokens += len(self.encoder.encode(message.content))
            if num_tokens > settings["max_chat_history_tokens"]:
                break
            shortened_messages.append(message)
        return ChatHistory(shortened_messages)

    def _handle_long_text(
        self, settings: Dict[str, Any], prompt: str, long_text: str
    ) -> List[str]:
        """
        :param settings: chatbot settings
        :param prompt: a prompt for the chat
        :param long_text: long text to be chunked
        :return: list of chunked prompts
        """
        prompt = "\n\n----\n\nFOR THE ABOVE DO THE FOLLOWING:\n" + prompt
        chunked_text = SmartChunker(
            self.encoder,
            settings["max_num_prompt_tokens"] - len(self.encoder.encode(prompt)),
            1 / 4,
        ).chunk(long_text)
        prompts = [chunked_text + prompt for chunked_text in chunked_text]
        return prompts

    def generate_responses(
        self,
        settings: Dict[str, Any],
        chat_llm: ChatLLM,
        chat_history: Optional[ChatHistory],
        variables_list: List[Dict[str, str]],
    ) -> Tuple[List[str], List[ChatHistory]]:
        """
        Generates responses to multiple prompts by concurrently running it through a respond flow.

        :param settings: chatbot settings
        :param chat_llm: the language model for generating responses
        :param chat_history: existing chat history, if any
        :param variables_list: a list of variables for each prompt
        :return: a tuple of lists with responses and history for each chat
        """
        input_variables_list = []
        for varriables in variables_list:
            input_variables = {
                **varriables,
                "identity": settings["persona"],
            }
            input_variables_list.append(input_variables)

        respond_flow = self.create_respond_flow(settings)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            responses_histories = list(
                executor.map(
                    lambda input_variables: respond_flow(
                        input_variables,
                        chat_llm=chat_llm,
                        input_chat_history=chat_history,
                        return_all=False,
                    ),
                    input_variables_list,
                )
            )

        responses = []
        histories = []
        for response, history in responses_histories:
            responses.append(response)
            histories.append(history)

        return responses, histories

    def create_respond_flow(self, settings: Dict[str, Any]) -> ChatFlow:
        """
        Create the flow for responding to a prompt.

        :param settings: chatbot settings
        :return: a chat flow that takes ("identity", "prompt") and outputs ("response")
        """
        return ChatFlow.from_dicts(
            [
                {Role.SYSTEM: "{identity}"},
                {Role.USER: "{prompt}"},
                {Role.ASSISTANT: "{response}"},
            ],
            verbose=settings["verbose"],
        )

    def run(self) -> None:
        chat_history = None
        while True:
            option = input(
                "\nSelect an option:\n"
                "1. Submit multiple prompts for concurrent processing\n"
                "2. Submit a prompt with a long text to chunk and respond\n"
                "3. Process a single prompt\n"
                "Enter option (1, 2, or 3): "
            )
            if option == "1":
                print(
                    "\nInput exactly `[END]` with nothing else in a prompt to finish entering prompts."
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
                    history_list,
                    total_tokens_list,
                    chat_history,
                ) = self.chat(prompts, chat_history=chat_history)

                for prompt, response, num_tokens in zip(
                    prompts, response_list, total_tokens_list
                ):
                    print("\nPrompt:", prompt)
                    print("\nResponse:", response)
                    print(f"\nNumber of tokens: {num_tokens / 1000:.1f}K")
                    print()

            elif option == "2":
                long_text = get_input("\nLong Text: ")
                if long_text is None:
                    continue
                prompt = get_input()
                if prompt is None:
                    continue

                print(f"\nResponding now.")

                (
                    response_list,
                    history_list,
                    total_tokens_list,
                    chat_history,
                ) = self.chat(
                    long_text, split_long_prompt=prompt, chat_history=chat_history
                )

                for response, num_tokens in zip(response_list, total_tokens_list):
                    print("\nResponse:", response)
                    print(f"\nNumber of tokens: {num_tokens / 1000:.1f}K")
                    print()

            elif option == "3" or option == "":
                prompt = get_input()
                if prompt is None:
                    continue

                print(f"\nResponding now.")

                (
                    response_list,
                    history_list,
                    total_tokens_list,
                    chat_history,
                ) = self.chat(prompt, chat_history=chat_history)

                print("\nResponse:", response_list[0])
                print(f"\nNumber of tokens: {total_tokens_list[0] / 1000:.1f}K")

            else:
                print("Invalid option. Please try again.")


class ContextChatbot(Chatbot):
    def __init__(
        self,
        openai_chat_model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        persona: str = "You are a helpful assistant.",
        max_num_prompt_tokens: int = 2000,
        enable_chat_history: bool = False,
        max_chat_history_tokens: int = 2000,
        verbose: bool = False,
    ):
        """
        :param openai_chat_model: the model to use from OpenAI's chat models
        :param temperature: the randomness of the model's responses
        :param persona: the persona that the chatbot should adopt
        :param max_num_prompt_tokens: the maximum number of tokens that a single prompt can have
        :param enable_chat_history: whether or not to keep a chat history
        :param max_chat_history_tokens: the maximum number of tokens that the chat history can have
        :param verbose: whether to print verbose output or not
        """
        super().__init__(
            openai_chat_model=openai_chat_model,
            temperature=temperature,
            persona=persona,
            max_num_prompt_tokens=max_num_prompt_tokens,
            enable_chat_history=enable_chat_history,
            max_chat_history_tokens=max_chat_history_tokens,
            verbose=verbose,
        )
        self.settings.update(
            {
                "state_prompt_for_context": False,
            }
        )

    # Need to implement chat for passing context

    def create_respond_flow(self, settings: Dict[str, Any]) -> ChatFlow:
        """
        Creates a full respond flow.

        :returns: a chat flow object that takes ("prompt", "instruction", "identity", "context")
                and outputs ("response", "memory")
        """

        # Define the individual flows
        def already_contextualized(variables, chat_llm, input_chat_history):
            return {"query": variables["prompt"]}, ([input_chat_history], [])

        add_query = FuncChatFlow(
            already_contextualized,
            input_varnames=set(["prompt"]),
            output_varnames=set(["query"]),
        )

        answer_flow = self.create_answer_flow(settings)

        respond_flow = SequentialChatFlows(
            [
                add_query,
                answer_flow,
            ],
            verbose=settings["verbose"],
        )

        return respond_flow

    def create_answer_flow(self, settings: Dict[str, Any]) -> ChatFlow:
        """
        Create the flow for responding to a prompt using previous gathered context from a query.

        :return: a chat flow that takes ("identity", "query", "context", "prompt", "instruction") and outputs ("response")
        """
        return ChatFlow.from_dicts(
            [
                {Role.SYSTEM: "{identity}"},
                {Role.USER: "Give me context to the prompt below:\n{query}"}
                if settings["state_prompt_for_context"]
                else {Role.USER: "Give me context to the prompt."},
                {
                    Role.ASSISTANT: "Here is some context related to the prompt:\n{context}",
                    "type": "input",
                },
                {
                    Role.USER: "Use the context you provided and your general knowledge to respond directly and accurately to the prompt below (prefer using the context over your general knowledge;{instruction}):\n"
                    "Prompt: {prompt}"
                },
                {Role.ASSISTANT: "{response}"},
            ],
            verbose=settings["verbose"],
        )


class MemoryChatbot(ContextChatbot):
    def __init__(
        self,
        memory: Optional[Memory] = None,
        memory_file: str = "memory_default.pkl",
        max_num_query_results: int = 14,
        num_query_results: int = 20,
        max_memory_context_tokens: int = 1500,
        memory_score_threshold: float = 0.7,
        combine_threshold: float = 0.1,
        summarize_context: bool = False,
        openai_chat_model: str = "gpt-3.5-turbo",
        max_num_prompt_tokens: int = 2000,
        max_num_tokens_per_memory: int = 500,
        temperature: float = 0.3,
        persona: str = "",
        enable_chat_history: bool = False,
        max_chat_history_tokens: int = 2000,
        enable_prompt_response_memory: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a MemoryChatbot.

        :param memory: Memory object to use, default is None.
        :param memory_file: File path to save memory data.
        :param max_num_query_results: Maximum number of memory query results.
        :param num_query_results: Number of query results.
        :param max_memory_context_tokens: Maximum number of memory context tokens.
        :param memory_score_threshold: Memory score threshold.
        :param combine_threshold: Combine threshold.
        :param summarize_context: Whether to summarize context or not.
        :param openai_chat_model: OpenAI chat model to use.
        :param max_num_prompt_tokens: Maximum number of prompt tokens.
        :param max_num_tokens_per_memory: Maximum number of tokens per memory.
        :param temperature: Randomness of the model's responses.
        :param persona: Persona that the chatbot should adopt.
        :param enable_chat_history: Whether to enable chat history or not.
        :param max_chat_history_tokens: Maximum number of chat history tokens.
        :param enable_prompt_response_memory: Whether to enable prompt response memory or not.
        :param verbose: Whether to print verbose output or not.
        """
        super().__init__(
            openai_chat_model=openai_chat_model,
            temperature=temperature,
            persona=persona,
            max_num_prompt_tokens=max_num_prompt_tokens,
            enable_chat_history=enable_chat_history,
            max_chat_history_tokens=max_chat_history_tokens,
            verbose=verbose,
        )

        if memory is None:
            self.memory = Memory(memory_file)
        else:
            self.memory = memory

        self.settings.update(
            {
                "max_num_query_results": max_num_query_results,
                "num_query_results": num_query_results,
                "max_memory_context_tokens": max_memory_context_tokens,
                "memory_score_threshold": memory_score_threshold,
                "combine_threshold": combine_threshold,
                "summarize_context": summarize_context,
                "max_num_tokens_per_memory": max_num_tokens_per_memory,
                "enable_prompt_response_memory": enable_prompt_response_memory,
                "token_cushion": 10,
                "max_instruction_tokens": 100,
            }
        )

    def chat(
        self,
        prompts: Union[str, List[str]],
        instructions: Optional[Union[str, List[str]]] = None,
        chat_history: Optional[ChatHistory] = None,
        split_long_prompt: Optional[bool] = None,
        **kwargs: Dict,
    ) -> Tuple[
        List[str], List[str], List[ChatHistory], List[int], Optional[ChatHistory]
    ]:
        """
        :param prompts: Single prompt or list of prompts.
        :param instructions: Single instruction or list of instructions, default is None.
        :param chat_history: Chat history, default is None.
        :param split_long_prompt: Whether to split long prompts or not, default is None.
        :param kwargs: Additional arguments.

        :returns: A tuple containing lists of responses, sources, histories, and total tokens,
                  and an optional chat history.
        """
        if not isinstance(prompts, list):
            prompts = [prompts]
        if instructions is not None and not isinstance(instructions, list):
            instructions = [instructions]

        if (
            not split_long_prompt
            and instructions is not None
            and len(instructions) != len(prompts)
        ):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) and instructions ({len(instructions)}) do not match."
            )
        if split_long_prompt and instructions is not None and len(instructions) != 1:
            raise ValueError(
                f"If split_long_prompt is provided, instructions must be a list of length 1."
            )
        if split_long_prompt and (len(prompts) != 1):
            raise ValueError(
                f"If split_long_prompt is provided, prompt must be a list of length 1."
            )

        settings = self._update_settings(kwargs)

        if instructions is not None:
            num_tokens = max(
                [len(self.encoder.encode(instruction)) for instruction in instructions]
            )
            if num_tokens > settings["max_instruction_tokens"]:
                raise ValueError(
                    f"An instruction is too long: {num_tokens} > {settings['max_instruction_tokens']}"
                )

        settings["persona"] = (
            "You are an intelligent assistant who thinks thoroughly through all the context to best respond to a prompt. "
            + settings["persona"]
        )

        if split_long_prompt:
            prompts = self._handle_long_text(settings, split_long_prompt, prompts[0])
            if instructions is not None:
                instructions = [instructions[0]] * len(prompts)

        self._check_prompts_tokens(settings, prompts)

        chat_llm = ChatLLM(
            gpt_model=settings["openai_chat_model"], temperature=settings["temperature"]
        )

        chat_history = self._get_history(settings, chat_history)

        if chat_history is not None:
            for message in chat_history.messages:
                print(message)

        if instructions is None:
            instructions = [""] * len(prompts)

        variables_list = [
            {"prompt": prompt, "instruction": instruction}
            for prompt, instruction in zip(prompts, instructions)
        ]

        response_list, history_list = self.generate_responses(
            settings, chat_llm, chat_history, variables_list
        )

        sources_list = [response["sources"] for response in response_list]
        response_list = [response["response"] for response in response_list]

        total_tokens_list = self._get_history_num_tokens(history_list)

        if settings["enable_chat_history"]:
            # Cannot handle when there are multiple prompts / histories
            if len(history_list) == 1:
                chat_history = self._handle_long_chat_history(settings, history_list[0])

        if settings["enable_prompt_response_memory"]:
            for prompt, response in zip(prompts, response_list):
                self._update_memory(settings, prompt, response)

        return (
            response_list,
            sources_list,
            history_list,
            total_tokens_list,
            chat_history,
        )

    def _update_memory(
        self, settings: Dict[str, Any], prompt: str, response: str
    ) -> None:
        """
        Adds prompt/response pairs to memory.

        :param settings: A dictionary of settings.
        :param prompt: The prompt string.
        :param response: The response string.
        """
        num_prompt_tokens = len(self.encoder.encode(prompt))
        num_response_tokens = len(self.encoder.encode(response))

        if (
            num_prompt_tokens + num_response_tokens + settings["token_cushion"]
            > settings["max_num_tokens_per_memory"]
        ):
            self._handle_large_memory_entries(
                settings,
                prompt,
                response,
                num_prompt_tokens,
                num_response_tokens,
            )
        else:
            self.memory.add(
                {
                    "text": f"Prompt: {prompt}\nResponse: {response}",
                    "metadata": f"source: Past Prompt & Response",
                }
            )

    def _handle_large_memory_entries(
        self,
        settings: Dict,
        prompt: str,
        response: str,
        num_prompt_tokens: int,
        num_response_tokens: int,
    ) -> None:
        """
        Adds long past prompt/response pairs to the memory.
        This done by attempting to chunk the respective long prompt or
        response while still including the entire smaller portion.
        However, if both the prompt and response are too long, then
        both are chunked and added separately.

        :param settings: A dictionary of settings.
        :param prompt: The prompt string.
        :param response: The response string.
        :param num_prompt_tokens: Number of prompt tokens.
        :param num_response_tokens: Number of response tokens.
        """
        if num_prompt_tokens < settings["max_num_tokens_per_memory"] / 3:
            for response_part in SmartChunker(
                self.encoder,
                settings["max_num_tokens_per_memory"]
                - num_prompt_tokens
                - settings["token_cushion"],
                1 / 3,
            ).chunk(response):
                self.memory.add(
                    {
                        "text": f"Prompt: {prompt}\nResponse: {response_part}",
                        "metadata": f"source: Past Prompt & Partial Response",
                    }
                )
        elif num_response_tokens < settings["max_num_tokens_per_memory"] / 3:
            for prompt_part in SmartChunker(
                self.encoder,
                settings["max_num_tokens_per_memory"]
                - num_response_tokens
                - settings["token_cushion"],
                1 / 3,
            ).chunk(prompt):
                self.memory.add(
                    {
                        "text": f"Prompt: {prompt_part}\nResponse: {response}",
                        "metadata": f"source: Past Partial Prompt & Response",
                    }
                )
        else:
            for prompt_part in SmartChunker(
                self.encoder,
                settings["max_num_tokens_per_memory"] - settings["token_cushion"],
                1 / 3,
            ).chunk(prompt):
                self.memory.add(
                    {
                        "text": f"Prompt: {prompt_part}",
                        "metadata": f"source: Past Prompt",
                    }
                )

            for response_part in SmartChunker(
                self.encoder,
                settings["max_num_tokens_per_memory"] - settings["token_cushion"],
                1 / 3,
            ).chunk(response):
                self.memory.add(
                    {
                        "text": f"Response: {response_part}",
                        "metadata": f"source: Past Response",
                    }
                )

    def create_respond_flow(self, settings: Dict[str, Any]) -> ChatFlow:
        """
        Creates a full respond flow.


        :returns: a chat flow object that takes ("prompt", "identity")
                and outputs ("response", "query", "context", "sources", "memory")
        """
        # Define the individual flows
        memory_flow = self.create_memory_flow(settings)
        prepare_context_flow = self.create_prepare_context_flow(settings)
        answer_flow = self.create_answer_flow(settings)

        # Only memory_flow gets past histories, which only come from the full history of each answer_flow
        # This means answer_flow itself does not get such histories.
        respond_flow = SequentialChatFlows(
            [
                NoHistory(memory_flow, allow_input_history=True),
                NoHistory(prepare_context_flow),
                answer_flow,
            ],
            verbose=settings["verbose"],
        )

        return respond_flow

    def create_contextualize_flow(self, settings: Dict[str, Any]) -> ChatFlow:
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
            verbose=settings["verbose"],
        )

    def create_memory_flow(self, settings: Dict[str, Any]) -> ChatFlow:
        """
        Create the memory chat flow.

        :return: a chat flow object that takes ("query") and outputs ("memory")
        """

        def already_contextualized(variables, chat_llm, input_chat_history):
            return {"query": variables["prompt"]}, ([input_chat_history], [])

        return MemoryChatFlow(
            self.create_contextualize_flow(settings)
            if settings["enable_chat_history"]
            else FuncChatFlow(
                already_contextualized,
                input_varnames=set(["prompt"]),
                output_varnames=set(["query"]),
            ),
            self.memory,
            memory_query_kwargs={
                "k": settings["max_num_query_results"],
                "combine_threshold": settings["combine_threshold"],
            },
            verbose=settings["verbose"],
        )

    def create_prepare_context_flow(self, settings: Dict[str, Any]) -> ChatFlow:
        """
        Create the flow for gathering context for a prompt.

        Steps:
        1. Gather context from memory using a standalone query based off of the prompt
        2. Sort and Remove excess documents
        3. Summarize each document (optional)
        4. Create a single context string containing all the documents upto a token limit

        :return: a chat flow object that takes ("memory", "query")
                and outputs ("context", "sources")
        """
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
            all_docs = []

            # Add memories to all_docs
            memories = variables["memory"]
            query = variables["query"]
            for memory in memories:
                if memory["score"] > settings["memory_score_threshold"]:
                    all_docs.append(
                        (
                            "Memory".upper(),
                            memory["text"],
                            memory["metadata"].replace("source: ", ""),
                            memory["score"],
                        )
                    )

            all_docs = sorted(all_docs, key=lambda x: x[3], reverse=True)[
                : settings["num_query_results"]
            ]

            if settings["summarize_context"]:

                def summarize_doc(args):
                    name, doc, source, score = args
                    variables, _ = prompted_summarization_flow(
                        {"text": doc, "prompt": query}, chat_llm=chat_llm
                    )
                    short_doc = variables["prompted_summarization"]

                    # Add logic to not run if already small, if summarizon large than input, dont use??
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

                if num_tokens > settings["max_memory_context_tokens"]:
                    break

                text += text_seg
                sources[source] = max(sources.get(source, score), score)

            sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)

            return {"context": text, "sources": sources}, ([input_chat_history], [])

        return FuncChatFlow(
            prepare_context_func,
            input_varnames=set(["memory", "query"]),
            output_varnames=set(["context", "sources"]),
        )

    def run(self) -> None:
        chat_history = None
        while True:
            option = input(
                "\nSelect an option:\n"
                "1. Submit multiple prompts for concurrent processing\n"
                "2. Submit a prompt with a long text to chunk and respond\n"
                "3. Process a single prompt\n"
                "Enter option (1, 2, or 3): "
            )
            if option == "1":
                print(
                    "\nInput exactly `[END]` with nothing else in a prompt to finish entering prompts."
                )
                prompts = []
                instructions = []
                while True:
                    prompt = get_input()
                    if prompt == "[END]":
                        break
                    prompts.append(prompt)

                    instruction = input("Extra Instructions: ").strip()
                    if instruction == "":
                        instruction = None
                    else:
                        instruction = f" {instruction}"
                    instructions.append(instruction)

                print(f"\nResponding now.")

                (
                    response_list,
                    sources_list,
                    history_list,
                    total_tokens_list,
                    chat_history,
                ) = self.chat(
                    prompts, instructions=instructions, chat_history=chat_history
                )

                for prompt, response, sources, num_tokens in zip(
                    prompts, response_list, sources_list, total_tokens_list
                ):
                    print("\nPrompt:", prompt)
                    print("\nResponse:", response)

                    print("\nSources:")
                    for ndx, source in enumerate(sources):
                        print(f"{ndx + 1}. {source}")

                    print(f"\nNumber of tokens: {num_tokens / 1000:.1f}K")
                    print()

            elif option == "2":
                long_text = get_input("\nLong Text: ")
                if long_text is None:
                    continue
                prompt = get_input()
                if prompt is None:
                    continue

                print(f"\nResponding now.")

                (
                    response_list,
                    sources_list,
                    history_list,
                    total_tokens_list,
                    chat_history,
                ) = self.chat(
                    long_text, split_long_prompt=prompt, chat_history=chat_history
                )

                for response, sources, num_tokens in zip(
                    response_list, sources_list, total_tokens_list
                ):
                    print("\nResponse:", response)

                    print("\nSources:")
                    for ndx, source in enumerate(sources):
                        print(f"{ndx + 1}. {source}")

                    print(f"\nNumber of tokens: {num_tokens / 1000:.1f}K")
                    print()

            elif option == "3" or option == "":
                prompt = get_input()
                if prompt is None:
                    continue
                instruction = input("Extra Instructions: ").strip()
                if instruction == "":
                    instruction = None
                else:
                    instruction = f" {instruction}"

                print(f"\nResponding now.")

                (
                    response_list,
                    sources_list,
                    history_list,
                    total_tokens_list,
                    chat_history,
                ) = self.chat(
                    prompt, instructions=instruction, chat_history=chat_history
                )

                print("\nResponse:", response_list[0])

                print("\nSources:")
                for ndx, source in enumerate(sources_list[0]):
                    print(f"{ndx + 1}. {source}")

                print(f"\nNumber of tokens: {total_tokens_list[0] / 1000:.1f}K")

            else:
                print("Invalid option. Please try again.")


def get_memory_chat_args():
    """
    Generate argparse object and parse command-line arguments for MemoryChat

    :return: An argparse object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Chat.")

    parser.add_argument(
        "--memory",
        type=str,
        default=None,
        help="Memory to use.",
    )
    parser.add_argument(
        "--memory_file",
        type=str,
        default="memory_default.pkl",
        help="Memory file to use.",
    )
    parser.add_argument(
        "--max_num_query_results",
        type=int,
        default=14,
        help="Maximum number of results for context.",
    )
    parser.add_argument(
        "--num_query_results",
        type=int,
        default=20,
        help="Number of memory queries.",
    )
    parser.add_argument(
        "--max_memory_context_tokens",
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
    parser.add_argument(
        "--enable_chat_history",
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
        "--enable_prompt_response_memory",
        action="store_true",
        help="Enables memory of prompts and responses.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity."
    )

    return parser.parse_args()


if __name__ == "__main__":
    chat = MemoryChatbot(**vars(get_memory_chat_args()))
    # chat = Chatbot(persona=vars(get_memory_chat_args())["persona"])
    chat.run()
