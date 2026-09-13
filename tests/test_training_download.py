import importlib.util
import hashlib
import io
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('training_download', Path(__file__).resolve().parents[1] / 'training/download_model.py')
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def test_verified_publication_and_no_overwrite(tmp_path):
    raw = b'project-authored-package-fixture'
    target = tmp_path / 'model'
    download.install(io.BytesIO(raw), target, hashlib.sha256(raw).hexdigest(), len(raw))
    assert target.read_bytes() == raw
    with pytest.raises(FileExistsError):
        download.install(io.BytesIO(raw), target, hashlib.sha256(raw).hexdigest(), len(raw))


@pytest.mark.parametrize('raw,size,sha', [(b'bad', 3, '0' * 64), (b'long', 2, '0' * 64), (b'a', 2, '0' * 64)])
def test_corrupt_download_not_published(tmp_path, raw, size, sha):
    target = tmp_path / 'model'
    with pytest.raises(ValueError):
        download.install(io.BytesIO(raw), target, sha, size)
    assert list(tmp_path.iterdir()) == []


def test_dangling_symlink_not_replaced(tmp_path):
    target = tmp_path / 'model'
    target.symlink_to(tmp_path / 'absent')
    with pytest.raises(FileExistsError):
        download.install(io.BytesIO(b''), target)
    assert target.is_symlink()
