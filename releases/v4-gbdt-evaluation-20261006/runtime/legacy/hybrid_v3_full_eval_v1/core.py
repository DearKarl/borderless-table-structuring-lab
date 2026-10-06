"""Identity, closed paths, immutable records and a single absolute clock."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

class ContractError(RuntimeError):
    pass

class DeadlineError(ContractError):
    pass

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path, deadline=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            if deadline is not None:
                deadline.remaining()
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractError("Expected lowercase SHA256")
    return value

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

def progress(path, value):
    path = Path(path)
    tmp = path.with_suffix(".next")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, allow_nan=False)
    tmp.replace(path)

def new_dir(path):
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=False)
    return p

def relative(value):
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ContractError("Portable relative path required")
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(x in ("..", ".") for x in value.split("/")):
        raise ContractError("Path escapes its declared root")
    return p.as_posix()

def closed(root, value):
    p = (Path(root) / relative(value)).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ContractError("Symlink escapes declared root")
    return p

def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise ContractError("Unsafe page/asset identifier")
    if value.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *("COM"+str(i) for i in range(1,10)), *("LPT"+str(i) for i in range(1,10))}:
        raise ContractError("Reserved portable filename")
    return value

def positive(value, name, maximum=None, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ContractError("Invalid numeric budget: " + name)
    if maximum is not None and value > maximum:
        raise ContractError("Budget exceeds ceiling: " + name)
    return value

def verify_tree(root, files, deadline=None):
    root = Path(root).resolve()
    if not files or not isinstance(files, dict):
        raise ContractError("Nonempty exact file lock required")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ContractError("Asset file set differs from lock")
    for name, expected in files.items():
        if sha(closed(root, name), deadline) != digest(expected):
            raise ContractError("Asset/source tamper: " + name)

class Deadline:
    def __init__(self, start, total, reserve=15, clock=time.monotonic):
        self.start, self.total, self.reserve, self.clock = start, total, reserve, clock
        positive(total, "total")
        positive(reserve, "host_cleanup", 240, zero=True)
        if reserve >= total:
            raise ContractError("Deadline has no execution allowance")

    def remaining(self, cleanup=False):
        left = self.start + self.total - self.clock() - (0 if cleanup else self.reserve)
        if left <= 0:
            raise DeadlineError("Absolute host deadline reached")
        return left

    def bound(self, seconds, cleanup=False):
        return min(seconds, self.remaining(cleanup))
