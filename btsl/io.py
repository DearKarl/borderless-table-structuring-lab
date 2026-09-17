"""Local, hash-bound artifacts. A lifetime lock protects publication/resume."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       indent=2, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_once(path, data):
    """Under the enclosing run lock, publish a complete file without changing old bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Existing artifact differs: " + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish(path, value):
    write_once(path, canonical(value))


def inside(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(root):
        raise ValueError("Unsafe input path")
    return resolved


class Lease:
    """POSIX macOS/Linux only, inherited by model subprocess to prevent orphan overlap."""
    def __init__(self, path):
        import fcntl
        self.stream = Path(path).open("a+b")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise RuntimeError("Run is already owned; do not start another worker")

    def __enter__(self):
        return self.stream.fileno()

    def __exit__(self, *args):
        self.stream.close()
