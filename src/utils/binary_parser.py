#!/usr/bin/env python3
"""
Binary Answer Parser - Notebook-aligned Implementation
Implements the simple binary_answer() function from the notebook for consistency
"""

def binary_answer(text: str) -> int:
    """
    Simple binary answer parser from notebook.

    Args:
        text: Response text to parse

    Returns:
        int: 1 if 'YES' found in text, 0 otherwise
    """
    if not text:
        return 0

    # Exact implementation from notebook cell 69
    return 1 if 'YES' in text else 0


def parse_binary_response(response: str) -> str:
    """
    Extract binary answer from response and return standardized YES/NO.

    Args:
        response: LLM response text

    Returns:
        str: 'YES', 'NO', or 'UNKNOWN'
    """
    if not response:
        return 'UNKNOWN'

    response_upper = response.upper()

    # Simple check - if 'YES' appears anywhere, return YES
    if 'YES' in response_upper:
        return 'YES'
    elif 'NO' in response_upper:
        return 'NO'
    else:
        return 'UNKNOWN'


def notebook_compatible_binary_answer(text: str, debug: bool = False) -> int:
    """
    Notebook-compatible binary answer function with optional debug output.

    Args:
        text: Response text to parse
        debug: If True, print debug output like notebook

    Returns:
        int: 1 if 'YES' found in text, 0 otherwise
    """
    if debug:
        print("binary_answer text = " + str(text))

    return binary_answer(text)