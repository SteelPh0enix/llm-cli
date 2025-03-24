from termcolor import colored
# some imports for the LLM
import os # type: ignore
import sys # type: ignore

def execute_python_code(code: str) -> str | None:
    """Executes arbitrary Python code. This function runs the code in current interpreter, therefore you should assume that only standard library is available and you should never try using anything else. Assume `os` and `sys` modules are already imported.

    Parameters:
       - code (str): The Python code to be executed.
    """
    print(colored(f"Executing following Python code: {code}", "light_red"))
    try:
        exec(code)
    except Exception as e:
        print(colored(f"Error executing code: {e}", "light_red"))
        return f"{e}"
    return None
