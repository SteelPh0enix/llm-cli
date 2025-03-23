# The documentation comment provides the description of the function and it's arguments for the LLM.
# It must conform to the following format:
# First, provide the function's description
# Then create a paragraph with "Parameters:" string, followed by list of parameters, each on a new line.
# Each parameter must have the following format:
#    - {parameter_name} ({parameter_type}): A description of the parameter.
# Parameter type block is optional, the type will be inferred from annotation.
# Parameter's description must fit in a single line.
# Function's description may be multi-line (newlines will be replaced by spaces), but it should be short and concise.

def example_tool(example_number: float, example_string: str) -> int:
    """
    This is an example tool function.

    Parameters:
       - example_number (float): A floating-point number.
       - example_string (str | None): An optional string parameter.
    """
    print(
        f"Example tool function called with arguments '{example_number}' and '{example_string}'."
    )
    return round(example_number)
