# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import hashlib
import re


def sanitize_filename(text_prompt: str, max_length: int = 200) -> str:
    """
    Sanitize a text prompt to be used as a filename.
    
    Replaces invalid filesystem characters, truncates to max_length, and appends
    a hash suffix to ensure uniqueness.
    
    Args:
        text_prompt: The text prompt to sanitize
        max_length: Maximum length for the filename (default: 200)
                   Leaves room for extensions and path components
    
    Returns:
        A sanitized filename-safe string
    """
    # Generate hash suffix for uniqueness (8 chars + 1 underscore = 9 chars)
    prompt_hash = hashlib.md5(text_prompt.encode('utf-8')).hexdigest()[:8]
    hash_suffix = f"_{prompt_hash}"
    
    # Sanitize: replace invalid chars, collapse whitespace/underscores, trim
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', '_', text_prompt).strip('_.')
    
    # Use default if empty after sanitization
    sanitized = sanitized or "prompt"
    
    # Truncate to fit max_length with hash suffix
    available_length = max_length - len(hash_suffix)
    if len(sanitized) > available_length:
        sanitized = sanitized[:available_length]
    
    return f"{sanitized}{hash_suffix}"

