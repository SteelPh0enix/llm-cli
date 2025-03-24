from termcolor import colored


def execute_python_code(code: str) -> None:
    """Executes arbitrary Python code.

    Parameters:
       - code (str): The Python code to be executed.
    """
    print(colored(f"Executing following Python code: {code}", "light_red"))
    exec(code)
