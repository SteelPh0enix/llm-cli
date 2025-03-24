from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Iterator, cast

import ollama
from colorama import just_fix_windows_console
from termcolor import colored

from tools_loader import LLMFunction, load_tools

DEFAULT_MODEL_NAME = "MistralSmall:latest"


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_json(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatHistory:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def _add_message(self, role: str, message: str) -> None:
        self.messages.append(ChatMessage(role, message))

    def add_user_message(self, message: str) -> None:
        self._add_message("user", message)

    def add_system_message(self, message: str) -> None:
        self._add_message("system", message)

    def add_assistant_message(self, message: str) -> None:
        self._add_message("assistant", message)

    def add_tool_message(self, message: str) -> None:
        self._add_message("tool", message)

    def to_json(self) -> list[dict[str, str]]:
        return [msg.to_json() for msg in self.messages]


def colored_system_message(text: str) -> str:
    return colored(text, "magenta")


def colored_user_message(text: str) -> str:
    return colored(text, "green")


def colored_assistant_message(text: str) -> str:
    return colored(text, "red")


@dataclass
class CLIArgs:
    model: str = DEFAULT_MODEL_NAME
    list_tools: bool = False
    verbose_logs: bool = False
    debug_logs: bool = False


def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(
        description="Chat with ollama-managed LLMs",
        epilog="Colors:\n"
        f"{colored_system_message('System message')}\n"
        f"{colored_user_message('User message')}\n"
        f"{colored_assistant_message('Assistant message')}",
    )

    parser.add_argument(
        "-l", "--list-tools", action="store_true", help="List available tools"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Specify the model to use",
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logs")

    args = parser.parse_args()
    return CLIArgs(
        args.model,
        args.list_tools,
        args.verbose,
        args.debug,
    )


def stream_chat_response(model: str, chat: ChatHistory) -> str:
    stream: Iterator[ollama.ChatResponse] = ollama.chat(
        model=model,
        messages=chat.to_json(),
        stream=True,
    )

    complete_message = ""
    for chunk in stream:
        message_chunk = chunk.message.content
        complete_message += cast(str, message_chunk)
        print(colored_assistant_message(message_chunk), end="", flush=True)  # type: ignore

    print(flush=True)
    return complete_message


def use_tool(
    model: str, chat: ChatHistory, tools: list[LLMFunction]
) -> tuple[str, str | None]:
    response: ollama.ChatResponse = ollama.chat(
        model=model,
        messages=chat.to_json(),
        tools=[tool.to_json() for tool in tools],
    )

    logging.info(f"LLM tool call response: {response}")
    if response.message.tool_calls is None:
        return cast(str, response.message.content), None

    tool_call = response.message.tool_calls[0]
    tool_function = tool_call.function
    if tool_function.name not in [tool.name for tool in tools]:
        raise ValueError(f"Unknown tool: {tool_function.name}")

    tool = [tool for tool in tools if tool.name == tool_function.name][0]
    tool_result = tool.function(**tool_function.arguments)

    return str(tool_result), tool_function.name


def tools_list_to_str(tools: list[LLMFunction]) -> str:
    return "\n".join(
        [
            f"{tool.name}({', '.join([p.name for p in tool.parameters.properties])})"
            for tool in tools
        ]
    )


def handle_user_command(
    command: str,
    prompt: str,
    model: str,
    chat: ChatHistory,
    tools: list[LLMFunction],
) -> ChatHistory:
    logging.info(f"Handling user command '{command}' with prompt '{prompt}'")

    match command:
        case "tool":
            response, called_tool_name = use_tool(model, chat, tools)
        case _:
            print(colored_system_message(f"Invalid command: {command}"))
            return chat

    return chat


def process_conversation(
    prompt: str,
    model: str,
    chat: ChatHistory,
    tools: list[LLMFunction],
) -> ChatHistory:
    if prompt.startswith("/"):
        command_end_index = prompt.find(" ")
        if command_end_index == -1:
            return handle_user_command(
                prompt[1:],
                "",
                model,
                chat,
                tools,
            )
        else:
            return handle_user_command(
                prompt[1:command_end_index],
                prompt[command_end_index + 1 :],
                model,
                chat,
                tools,
            )

    logging.info(f"Processing conversation with prompt '{prompt}'")
    chat.add_user_message(prompt)
    response = stream_chat_response(model, chat)
    chat.add_assistant_message(response)

    return chat


def main() -> int:
    args = parse_args()

    if args.debug_logs:
        log_level = logging.DEBUG
    elif args.verbose_logs:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s]<%(levelname)s> %(message)s",
    )
    logging.debug("Debug logs enabled!")
    logging.info("Verbose logs enabled!")

    tools = load_tools()

    if args.list_tools:
        print(tools_list_to_str(tools))
        return 0

    chat = ChatHistory()

    logging.info("Detected tools:")
    logging.info(f"\n{tools_list_to_str(tools)}")

    while True:
        try:
            prompt = input(colored_user_message(">")).strip()
            if not prompt:
                continue
            chat = process_conversation(prompt, args.model, chat, tools)
            # conversation.append({"role": "user", "content": prompt})
            # logging.debug(f"Querying LLM with following conversation: {conversation}")
            # if prompt.startswith("/tool"):
            #     response, called_tool_name = use_tool(conversation, tools)
            #     if called_tool_name:
            #         conversation.append({"role": "tool", "content": response})
            #         print(f"[{called_tool_name} called, return value: {response}]")
            #         post_tool_response = stream_chat_response(conversation)
            #         conversation.append(
            #             {"role": "assistant", "content": post_tool_response}
            #         )
            #     else:
            #         conversation.append({"role": "assistant", "content": response})
            #         print(response)
            # else:
            #     response = stream_chat_response(conversation)
            #     conversation.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            print(flush=True)
            return 0


if __name__ == "__main__":
    just_fix_windows_console()
    sys.exit(main())
