"""Manifest creation is separate from inference; PDF import is lazy."""
from pathlib import Path
import importlib.metadata
from .core import ContractError, identifier, sha, write
from .contracts import inputs

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".jp2", ".gif"}

def make_manifest(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    paths = sorted(source.rglob("*")) if source.is_dir() else [source]
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | {".pdf"}]
    if not paths:
        raise ContractError("No supported inputs")
    pages = []
    for f in paths:
        if not f.is_relative_to(output.parent):
            raise ContractError("Place manifest at/above the input paths (closed relative paths)")
        rel = f.relative_to(output.parent).as_posix()
        h = sha(f)
        doc = "doc-" + h[:20]
        if f.suffix.lower() == ".pdf":
            if importlib.metadata.version("pypdfium2") != "4.30.0":
                raise ContractError("PDF inventory requires original pypdfium2==4.30.0")
            import pypdfium2
            pdf = pypdfium2.PdfDocument(str(f))
            try:
                count = len(pdf)
            finally:
                pdf.close()
            for ordinal in range(count):
                pages.append(dict(page_id=identifier(f.stem)+f"-p{ordinal+1:05d}", path=rel,
                    file_sha256=h, kind="pdf", source_document_id=doc, source_sha256=h,
                    page_count=count, page_ordinal=ordinal))
        else:
            pages.append(dict(page_id=identifier(f.stem), path=rel, file_sha256=h,
                              kind="image", source_document_id="unknown"))
    # Validate before the immutable output is committed.
    ids = [p["page_id"].casefold() for p in pages]
    if len(ids) != len(set(ids)):
        raise ContractError("Duplicate stems: supply an explicit uniquely named manifest")
    write(output, {"schema":1, "pages":pages})
    inputs(output)
    return {"manifest":str(output), "sha256":sha(output), "pages":len(pages)}
