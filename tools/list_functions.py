import ast


def list_functions(filepath: str) -> list[dict]:
    """Parse a Python file and return its top-level function definitions."""
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "docstring": ast.get_docstring(node) or "",
                "source": ast.get_source_segment(source, node),
            })

    return functions
