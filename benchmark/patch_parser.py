import re


def extract_java_files(patch: str) -> list[str]:
    files = set()
    for match in re.finditer(
        r"^diff --git a/(.*?\.java) b/(.*?\.java)", patch, re.MULTILINE
    ):
        files.add(match.group(2))
    for match in re.finditer(
        r"^diff --git /dev/null b/(.*?\.java)", patch, re.MULTILINE
    ):
        files.add(match.group(1))
    return sorted(files)


def extract_methods_from_patch(patch: str) -> list[str]:
    methods = set()

    for match in re.finditer(r"^@@ .+? @@\s+.*?(\w+)\s*\(", patch, re.MULTILINE):
        method_name = match.group(1)
        if method_name not in (
            "if",
            "for",
            "while",
            "switch",
            "catch",
            "class",
            "interface",
            "enum",
            "new",
            "return",
        ):
            methods.add(method_name)
    sig_pattern = re.compile(
        r"^[+]\s*(?:public|private|protected)?\s*(?:static\s+)?"
        r"(?:final\s+)?(?:synchronized\s+)?(?:abstract\s+)?"
        r"(?:<[^>]+>\s+)?"
        r"\w+(?:<[^>]*>)?(?:\[\])?\s+"
        r"(\w+)\s*\(",
        re.MULTILINE,
    )
    for match in sig_pattern.finditer(patch):
        method_name = match.group(1)
        if method_name not in (
            "if",
            "for",
            "while",
            "switch",
            "catch",
            "class",
            "interface",
            "enum",
            "new",
            "return",
        ):
            methods.add(method_name)

    return sorted(methods)
