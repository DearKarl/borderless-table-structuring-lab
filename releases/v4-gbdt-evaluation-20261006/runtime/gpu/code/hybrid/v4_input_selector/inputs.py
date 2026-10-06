"""Input transformations at the native rendered-page boundary."""
import hashlib
import json
import math

CAP = 3500

def pdf_geometry(page_size, dpi=200):
    if len(page_size) != 2 or not all(math.isfinite(x) and x > 0 for x in page_size):
        raise ValueError("Invalid display page size")
    scale = min(dpi / 72, CAP / max(page_size))
    # PDFium render uses ceil independently on the displayed width/height.
    return tuple(max(1, math.ceil(x * scale)) for x in page_size), scale

def image_fingerprint(image):
    return {"size": list(image.size), "mode": image.mode,
            "sha256": hashlib.sha256(image.tobytes()).hexdigest()}

def alias_key(image, pipeline_config_sha256):
    payload = {"image": image_fingerprint(image), "pipeline": pipeline_config_sha256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def transform_native_image(image_dict, action, source, page_size):
    """Never rewrap an image as PDF. A is an object-preserving pass-through."""
    if action not in source.available_actions():
        raise ValueError("Action unavailable for original provenance")
    image = image_dict["img_pil"]
    before = image_fingerprint(image)
    result = image_dict
    if action == "B" and source.source_type == "raster":
        from PIL import Image
        factor = min(1.5, CAP / max(image.size))
        if factor < 1:
            raise ValueError("Native baseline exceeds frozen cap")
        dims = tuple(max(1, math.floor(x * factor + .5)) for x in image.size)
        if dims != image.size:
            result = dict(image_dict)
            result["img_pil"] = image.resize(dims, Image.Resampling.BICUBIC)
            result["scale"] = image_dict["scale"] * factor
    elif action == "C":
        import cv2
        import numpy as np
        from PIL import Image
        dims, scale = pdf_geometry(page_size, 200)
        if image.mode != "RGB":
            raise ValueError("C requires verified native RGB mode; no silent conversion")
        result = dict(image_dict)
        result["img_pil"] = Image.fromarray(cv2.resize(
            np.asarray(image), dims, interpolation=cv2.INTER_AREA))
        result["scale"] = scale
    prepared = result["img_pil"]
    a_dims, a_scale = pdf_geometry(page_size, 200)
    audit = {"action": action, "original_source_type": source.source_type,
        "pdf_content_type": source.pdf_content_type, "before": before,
        "prepared": image_fingerprint(prepared), "scale": result["scale"],
        "a_geometry_size": list(a_dims), "a_geometry_scale": a_scale,
        "page_affine_xy": [prepared.width/page_size[0], prepared.height/page_size[1]],
        "cap_noop": action == "B" and prepared.size == a_dims,
        "effective_grid_increase": None}
    return result, audit

