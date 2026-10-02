"""Strict digest syntax before comparing file identities."""
import re


def digest(value):
    if not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None:
        raise ValueError('SHA256 must be exactly 64 lowercase hexadecimal characters')
    return value
