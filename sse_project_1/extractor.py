import re


def extract_joules(filepath: str):
    with open(filepath, "r") as file:
        content = file.read()

    pattern = r"Energy consumption in joules:\s*([\d\.]+)"
    values = [float(match) for match in re.findall(pattern, content)]
    return values
