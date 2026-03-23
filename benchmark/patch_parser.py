"""Parse unified diff patches to extract changed Java files and methods."""

import re


def extract_java_files(patch: str) -> list[str]:
    """Extract unique Java file paths from a unified diff patch."""
    files = set()
    for match in re.finditer(r'^diff --git a/(.*?\.java) b/(.*?\.java)', patch, re.MULTILINE):
        # Use the 'b/' path (new file path)
        files.add(match.group(2))
    # Also handle new files (only in b/)
    for match in re.finditer(r'^diff --git /dev/null b/(.*?\.java)', patch, re.MULTILINE):
        files.add(match.group(1))
    return sorted(files)


def extract_methods_from_patch(patch: str) -> list[str]:
    """Extract method names from @@ hunk headers and modified method signatures."""
    methods = set()

    # From @@ hunk headers: @@ -12,7 +12,8 @@ public void handleRequest(...)
    for match in re.finditer(r'^@@ .+? @@\s+.*?(\w+)\s*\(', patch, re.MULTILINE):
        method_name = match.group(1)
        # Filter out common non-method keywords
        if method_name not in ('if', 'for', 'while', 'switch', 'catch', 'class', 'interface', 'enum', 'new', 'return'):
            methods.add(method_name)

    # From added/modified lines with method signatures
    sig_pattern = re.compile(
        r'^[+]\s*(?:public|private|protected)?\s*(?:static\s+)?'
        r'(?:final\s+)?(?:synchronized\s+)?(?:abstract\s+)?'
        r'(?:<[^>]+>\s+)?'  # generics
        r'\w+(?:<[^>]*>)?(?:\[\])?\s+'  # return type
        r'(\w+)\s*\(',
        re.MULTILINE
    )
    for match in sig_pattern.finditer(patch):
        method_name = match.group(1)
        if method_name not in ('if', 'for', 'while', 'switch', 'catch', 'class', 'interface', 'enum', 'new', 'return'):
            methods.add(method_name)

    return sorted(methods)
