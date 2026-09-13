#!/usr/bin/env python3
"""Download the exact Explicit-v2 Original weight, without loading a model."""

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

URL = ('https://github.com/DearKarl/borderless-table-structuring-lab/releases/'
       'download/training-explicit-v2-original-2026.09.13.1/model.safe-state')
SHA256 = 'c1615ce37de058e82b0ec33a84fb06afb25d4f6d4dfa401ff1e67f9d547019f0'
SIZE = 4754162


def install(stream, output, expected_sha=SHA256, expected_size=SIZE):
    """Verify bytes before atomic, no-overwrite publication on a local filesystem."""
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.training-download-', delete=False) as sink:
            temporary = Path(sink.name)
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                count += len(chunk)
                if count > expected_size:
                    raise ValueError('download exceeds expected size')
                digest.update(chunk)
                sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if count != expected_size or digest.hexdigest() != expected_sha:
            raise ValueError('checkpoint size or SHA256 mismatch')
        os.chmod(temporary, 0o644)
        # link is atomic and fails if any output already exists, including a symlink.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/training/model.safe-state')
    args = parser.parse_args()
    if os.path.lexists(Path(args.output).expanduser()):
        raise FileExistsError(args.output)
    request = urllib.request.Request(URL, headers={'User-Agent': 'borderless-table-structuring-lab/1'})
    with urllib.request.urlopen(request, timeout=120) as response:
        if urllib.parse.urlsplit(response.geturl()).scheme != 'https':
            raise ValueError('non-HTTPS redirect refused')
        path = install(response, args.output)
    print('Verified checkpoint: ' + str(path))
    print('SHA256: ' + SHA256)


if __name__ == '__main__':
    main()
