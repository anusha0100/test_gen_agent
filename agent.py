import sys
import time

from google import genai
from google.genai import errors, types

from config import GEMINI_API_KEY, MODEL_NAME
from tools.list_functions import list_functions

client = genai.Client(api_key=GEMINI_API_KEY)

MAX_RETRIES = 5
BASE_DELAY_SECONDS = 2


def generate_with_retry(**kwargs):
    """Call generate_content, retrying with backoff on transient 503s."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_DELAY_SECONDS * (2**attempt)
            print(f"Model overloaded, retrying in {wait}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)

LIST_FUNCTIONS_DECLARATION = {
    "name": "list_functions",
    "description": "Parse a Python file (local path or GitHub blob URL) and return its top-level function definitions.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "filepath": {"type": "STRING", "description": "Path to the Python file"}
        },
        "required": ["filepath"],
    },
}

GENERATION_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[LIST_FUNCTIONS_DECLARATION])]
)


def run_tool(name: str, args: dict):
    if name == "list_functions":
        return list_functions(args["filepath"])
    raise ValueError(f"Unknown tool: {name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_python_file>")
        return

    filepath = sys.argv[1]
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=(
                f"Here's a Python file: {filepath}\n"
                "List its functions and tell me which ones look like they "
                "most need test coverage (e.g. branching logic, edge cases, "
                "non-trivial return values) and why."
            ))],
        )
    ]

    while True:
        response = generate_with_retry(
            model=MODEL_NAME,
            contents=contents,
            config=GENERATION_CONFIG,
        )

        part = response.candidates[0].content.parts[0]

        if not part.function_call:
            print(response.text)
            break

        contents.append(response.candidates[0].content)

        result = run_tool(part.function_call.name, dict(part.function_call.args))
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name=part.function_call.name,
                    response={"result": result},
                )],
            )
        )


if __name__ == "__main__":
    main()