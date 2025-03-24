from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable


class LLMParameterType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"

    @staticmethod
    def from_python_type(python_type: type) -> LLMParameterType:
        if python_type is str:
            return LLMParameterType.STRING
        elif python_type is int:
            return LLMParameterType.INTEGER
        elif python_type is float:
            return LLMParameterType.NUMBER
        elif python_type is bool:
            return LLMParameterType.BOOLEAN
        else:
            raise ValueError(f"Unsupported Python type: {python_type}")


@dataclass
class LLMParameterProperty:
    """
    Data class representing a parameter property for an LLM function.

    Attributes:
        name (str): The name of the parameter.
        parameter_type (LLMParameterType): The type of the parameter.
        description (str): A detailed description of what the parameter represents and its expected usage.
        allowed_values (list[str] | None): Allowed values for the parameter, if any. Useful for enumerating possible options.
        is_required (bool): Indicates whether the parameter must be provided when calling the LLM function.
    """

    name: str
    parameter_type: LLMParameterType
    description: str
    allowed_values: list[str] | None
    is_required: bool = True

    def to_json(self) -> dict[str, Any]:
        """
        Convert the LLMParameterProperty object to a JSON-compatible dictionary for Ollama/OpenAI request.

        Returns:
            dict[str, Any]: The JSON-compatible dictionary.
        """
        json_obj: dict[str, str | list[str]] = {
            "type": str(self.parameter_type),
            "description": self.description,
        }

        if self.allowed_values is not None:
            json_obj["enum"] = self.allowed_values

        return json_obj

    # TODO: object parsing (implement in LLMParameterObject)
    @staticmethod
    def from_python_parameter(
        param: inspect.Parameter, description: str
    ) -> LLMParameterProperty:
        return LLMParameterProperty(
            name=param.name,
            parameter_type=LLMParameterType.from_python_type(param.annotation),
            description=description,
            # TODO: StrEnum parsing
            allowed_values=None,
            is_required=not param.annotation == inspect.Parameter.empty
            and not param.kind == inspect.Parameter.VAR_POSITIONAL,
        )


@dataclass
class LLMParameterObject:
    """
    Data class representing a parameter object for an LLM function.

    Attributes:
        name (str): The name of the parameter object.
        properties (list[LLMParameterProperty | LLMParameterObject]): The properties of the parameter object.
        is_required (bool): Whether the parameter object is required.
    """

    name: str
    properties: list[LLMParameterProperty | LLMParameterObject]
    is_required: bool = True

    def to_json(self) -> dict[str, Any]:
        """
        Convert the LLMParameterObject object to a JSON-compatible dictionary for Ollama/OpenAI request.

        Returns:
            dict[str, Any]: The JSON-compatible dictionary.
        """
        return {
            "type": "object",
            "properties": {prop.name: prop.to_json() for prop in self.properties},
            "required": [prop.name for prop in self.properties if prop.is_required],
            "additionalProperties": False,
        }


def _parse_doc_comment(comment: str) -> tuple[str, dict[str, str]]:
    """
    Parses the provided docstring of a Python function in the following format (--- not included, used only as a separator here):

    ---
    Function description and return format go here.

    Parameters:
        - First parameter (type, optional): A description of the first parameter.
        - Second parameter: A description of the second parameter.
    ---

    Returned parameters dict should be validated with function's signature.
    If the function's signature does not match the provided descriptions, an error should be raised.
    Type from doc comment is optional (only for humans), and is ignored during parsing.
    Parameters section should be omitted if function does not have any parameters.

    Returns:
        tuple[str, dict[str, str]]: A tuple containing the function's description
                                        and a dictionary of parameters descriptions,
                                        where the parameter name is the key.
    """
    parameters_header = "Parameters:"
    parameter_regex = r"-(.*?)(\(.*?\))?:(.*)"
    parameters_header_position = comment.find(parameters_header)
    if parameters_header_position == -1:
        return comment.strip().replace("\n", " ").replace("\t", " "), {}

    description = (
        comment[:parameters_header_position]
        .strip()
        .replace("\n", " ")
        .replace("\t", " ")
    )
    parameters_docs = comment[
        parameters_header_position + len(parameters_header) :
    ].strip()
    parameters: dict[str, str] = {}

    for line in [line.strip() for line in parameters_docs.splitlines()]:
        match = re.match(parameter_regex, line)
        if match:
            parameter_name = match.group(1).strip()
            parameter_description = match.group(3).strip()
            parameters[parameter_name] = parameter_description
    return description, parameters


@dataclass
class LLMFunction:
    """
    Data class representing a function for an LLM.

    Attributes:
        name (str): The name of the function.
        description (str): A description of the function.
        parameters (LLMParameterObject): The parameters of the function.
    """

    name: str
    description: str
    parameters: LLMParameterObject
    function: Callable[..., Any]

    def to_json(self) -> dict[str, Any]:
        """
        Convert the LLMFunction object to a JSON-compatible dictionary for Ollama/OpenAI request.

        Returns:
            dict[str, Any]: The JSON-compatible dictionary.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_json(),
                "strict": True,
            },
        }

    @staticmethod
    def from_python_function(function: Callable[..., Any]) -> LLMFunction:
        """
        Convert a Python function to an LLMFunction object.

        This method extracts the name, description (from docstring), and parameters of the provided Python function and returns an LLMFunction object representing it.

        Args:
            function (Callable): The Python function to convert.

        Returns:
            LLMFunction: An LLMFunction object representing the input Python function.

        Raises:
            ValueError: If the function lacks a docstring, docstring has invalid format, or if any parameter does not have type annotations.
        """

        name = function.__name__
        raw_doc = function.__doc__
        if raw_doc is None:
            raise ValueError(
                f"Function {name} is missing a description in its docstring."
            )
        function_description, params_description = _parse_doc_comment(raw_doc)

        signature = inspect.signature(function)
        parameters = LLMParameterObject(name="args", properties=[])
        if set(signature.parameters.keys()) != set(params_description.keys()):
            raise ValueError(
                f"Function {name} has parameters that are not described in its docstring ({set(signature.parameters.keys())} != {set(params_description.keys())})."
            )

        for param_name, param_type in signature.parameters.items():
            if param_name == "return":
                continue
            if param_type.annotation == inspect.Parameter.empty:
                raise ValueError(
                    f"Parameter {param_name} in function {name} has no type annotation."
                )
            parameters.properties.append(
                LLMParameterProperty.from_python_parameter(
                    param_type, params_description[param_name]
                )
            )

        return LLMFunction(name, function_description, parameters, function)


def import_module_from_path(module_name: str, file_path: Path) -> Any:
    """
    Import a Python module from a given file path at runtime.

    Args:
        module_name (str): The name of the module.
        file_path (Path): The file path to the module.

    Returns:
        module: The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create a module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    if spec.name in sys.modules:
        raise RuntimeError("Module was already imported")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_function_from_module(module: Any, function_name: str) -> Any:
    """
    Retrieve an object representing a function from the provided Python module.

    Args:
        module (module): The imported module.
        function_name (str): The name of the function to retrieve.

    Returns:
        function: The retrieved function object.
    """
    if not hasattr(module, function_name):
        raise AttributeError(
            f"Module '{module.__name__}' does not have an attribute named '{function_name}'"
        )

    attr = getattr(module, function_name)
    if not inspect.isfunction(attr):
        raise TypeError(
            f"The attribute '{function_name}' in module '{module.__name__}' is not a function."
        )

    return attr


def load_tools(tools_directory: Path = Path("./tools")) -> list[LLMFunction]:
    """
    Index all tools (functions) from the specified directory.

    Args:
        tools_directory (Path): The directory containing tool modules.

    Returns:
        list[LLMFunction]: A list of LLM function objects.
    """

    functions: list[LLMFunction] = []

    for python_file in tools_directory.glob("*.py"):
        function_name = python_file.stem
        module = import_module_from_path(function_name, python_file)
        tool_function = get_function_from_module(module, function_name)
        functions.append(LLMFunction.from_python_function(tool_function))

    return functions
