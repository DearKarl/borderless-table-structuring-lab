"""Strict Docker inspect absence recognition for one exact requested target."""
import re

def inspect_absent(result,target):
    if list(result.args)!=['docker','inspect',target] or result.returncode!=1:
        return False
    if result.stdout.strip() not in ('','[]'):
        return False
    prefix=r'(?i:(?:error:\s*|error response from daemon:\s*)?no such (?:object|container):\s*)'
    return re.fullmatch(prefix+re.escape(target),result.stderr.strip()) is not None
