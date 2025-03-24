from termcolor import colored

import subprocess


def run_process(command: str) -> dict[str, str]:
    """Runs a command in the shell and returns its stdout and stderr as dictionary.
    Example usage: run_process("ls -l"), run_process("echo hello world")
    
    Parameters:
        - command (str): The command to be executed.
    """
    print(colored(f"Running command: {command}", "light_red"))

    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        return {"stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        print(colored(f"Command failed with error: {e}", "light_red"))
        return {"stdout": "", "stderr": str(e)}
