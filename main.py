from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
import logging
from typing import Iterator, cast

import ollama

from tools_loader import LLMFunction, load_tools

MODEL_NAME = "MistralSmall:latest"

ChatMessage = dict[str, str]


@dataclass
class CLIArgs:
    list_tools: bool = False


def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(
        description="Chat with ollama-managed LLMs"
    )
    parser.add_argument(
        "-l", "--list-tools", action="store_true", help="List available tools"
    )
    args = parser.parse_args()
    return CLIArgs(args.list_tools)


def stream_chat_response(messages: list[ChatMessage]) -> str:
    stream: Iterator[ollama.ChatResponse] = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )

    complete_message = ""
    for chunk in stream:
        message_chunk = chunk.message.content
        complete_message += cast(str, message_chunk)
        print(message_chunk, end="", flush=True)

    print(flush=True)
    return complete_message


def use_tool(
    messages: list[ChatMessage], tools: list[LLMFunction]
) -> tuple[str, str | None]:
    response: ollama.ChatResponse = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=[tool.to_json() for tool in tools],
    )

    logging.debug(f"Tool response: {response}")
    if response.message.tool_calls is None:
        return cast(str, response.message.content), None

    tool_call = response.message.tool_calls[0]
    tool_function = tool_call.function
    if tool_function.name not in [tool.name for tool in tools]:
        raise ValueError(f"Unknown tool: {tool_function.name}")

    tool = [tool for tool in tools if tool.name == tool_function.name][0]
    tool_result = tool.function(**tool_function.arguments)

    return str(tool_result), tool_function.name


def main() -> int:
    args = parse_args()
    tools = load_tools()

    if args.list_tools:
        print("Detected tools:")
        for tool in tools:
            print(
                f"{tool.name}({', '.join([p.name for p in tool.parameters.properties])})"
            )
        return 0

    conversation: list[ChatMessage] = []

    logging.debug("Detected tools:")
    for tool in tools:
        logging.debug(
            f"{tool.name}({', '.join([p.name for p in tool.parameters.properties])})"
        )

    while True:
        try:
            prompt = input(">").strip()
            if not prompt:
                continue
            conversation.append({"role": "user", "content": prompt})
            logging.debug(f"Querying LLM with following conversation: {conversation}")
            if prompt.startswith("/tool"):
                response, called_tool_name = use_tool(conversation, tools)
                if called_tool_name:
                    conversation.append({"role": "tool", "content": response})
                    print(f"[{called_tool_name} called, return value: {response}]")
                    post_tool_response = stream_chat_response(conversation)
                    conversation.append({"role": "assistant", "content": post_tool_response})
                else:
                    conversation.append({"role": "assistant", "content": response})
                    print(response)
            else:
                response = stream_chat_response(conversation)
                conversation.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            print(flush=True)
            return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="[%(asctime)s]<%(levelname)s> %(message)s"
    )
    logging.debug("Debug enabled!")
    sys.exit(main())
