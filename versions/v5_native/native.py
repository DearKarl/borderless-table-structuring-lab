"""Observe the default TeleOCR boundary without rerendering or altering inputs."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

from .io_utils import atomic_json, digest


def pixel_identity(image):
    return {"size": list(image.size), "mode": image.mode,
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()}


@contextmanager
def capture_render(folder):
    """Copy the very image consumed by doc_analyze; call its loader exactly once."""
    import TeleOCR.src.vlm_analyze as analyze
    from TeleOCR.tools.pdf_image_tools import get_page_size
    real = analyze.load_images_from_pdf
    captured = []

    def observed(*args, **kwargs):
        if captured:
            raise RuntimeError("Unexpected repeated native rendering")
        images, doc = real(*args, **kwargs)
        if len(images) != 1:
            raise ValueError("The experiment requires one page per sample")
        for index, item in enumerate(images):
            # No color conversion, resizing, or source-image reopening here.
            image = item["img_pil"].copy()
            points = list(get_page_size(doc[index]))
            middle_size = [int(x) for x in points]
            evidence = {**pixel_identity(image), "renderer_scale": item["scale"],
                        "pdf_page_size": points, "middle_page_size": middle_size,
                        "middle_to_pixels": [image.width / middle_size[0], image.height / middle_size[1]],
                        "configured_dpi": 200, "configured_long_edge_cap": 3500,
                        "source": "TeleOCR.src.vlm_analyze.load_images_from_pdf return value"}
            folder.mkdir(parents=True, exist_ok=True)
            image.save(folder / "rendered.png")
            evidence["png_sha256"] = digest(folder / "rendered.png")
            atomic_json(folder / "RENDER.json", evidence)
            captured.append((image, evidence))
        return images, doc

    analyze.load_images_from_pdf = observed
    try:
        yield captured
    finally:
        analyze.load_images_from_pdf = real


def native_page(client, path, folder, source_page_index=None):
    from TeleOCR.engine import do_parse
    from TeleOCR.tools.read_file import read_fn
    folder = Path(folder)
    with capture_render(folder / "capture") as captures:
        pages = None if source_page_index is None else [int(source_page_index)]
        middle = do_parse(str(folder / "native"), ["page"], [read_fn(path)], [pages], predictor=client)[0]
    if len(captures) != 1 or len(middle.get("pdf_info", [])) != 1:
        raise RuntimeError("Native capture cardinality differs")
    image, evidence = captures[0]
    if middle["pdf_info"][0]["page_size"] != evidence["middle_page_size"]:
        raise RuntimeError("Native coordinate frame changed")
    markdown = (folder / "native" / "page" / "page.md").read_text(encoding="utf-8")
    return middle, markdown, image


def tensor_identity(value):
    import torch
    if not isinstance(value, torch.Tensor):
        return None
    array = value.detach().cpu().contiguous()
    raw = array.view(torch.uint8).numpy().tobytes()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            **({"values": array.tolist()} if array.numel() <= 12 else {})}


def install_input_audit(client, emit):
    """Hash actual EngineCoreRequest tensors, including cached processed inputs."""
    engine = client.client.vllm_llm.llm_engine
    processor = engine.processor
    real_process = processor.process_inputs
    real_batch = client.client._predict_one_batch

    def process(*args, **kwargs):
        result = real_process(*args, **kwargs)
        request = result[1]
        features = []
        for spec in request.mm_features or []:
            fields = {}
            if spec.data is not None:
                for key, field in spec.data.items():
                    value = field.data
                    observed = tensor_identity(value)
                    if observed is not None:
                        fields[key] = observed
            features.append({"identifier": spec.identifier, "modality": spec.modality,
                             "fields": fields, "cache_reference_only": spec.data is None})
        emit({"event": "model_input", "request_id": request.request_id,
              "prompt_tokens": len(request.prompt_token_ids or []), "features": features})
        return result

    def batch(image_objs, chat_prompts, sampling_params):
        emit({"event": "request_batch", "requests": len(image_objs), "inputs": [
            {**pixel_identity(im), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
             "sampling": str(params)} for im, prompt, params in zip(image_objs, chat_prompts, sampling_params)]})
        result = real_batch(image_objs, chat_prompts, sampling_params)
        emit({"event": "request_batch_complete", "requests": len(result)})
        return result

    processor.process_inputs = process
    client.client._predict_one_batch = batch
    return {"process_inputs": real_process, "batch": real_batch}


def realized_view(client, image, kind):
    """Run only the unchanged CPU preprocessing path, never OCR or label access."""
    from TeleOCR.vlm_utils.structs import ContentBlock
    prepared, prompts, params, _ = client.helper.prepare_for_extract(
        image, [ContentBlock(kind, [0.0, 0.0, 1.0, 1.0])])
    if len(prepared) != 1:
        raise ValueError("Expected one feasible content input")
    internal = client.client
    prompt = internal.tokenizer.apply_chat_template(internal.build_messages(prompts[0]),
                                                    tokenize=False, add_generation_prompt=True)
    # Invoke the same HF processor directly, without populating vLLM's sender cache.
    # Populating that cache without submitting a request could create a false hit.
    mm_processor = internal.vllm_llm.llm_engine.processor.input_preprocessor._get_mm_processor()
    processed = mm_processor._call_hf_processor(prompt, {"images": [prepared[0]]}, {}, {})
    fields = {}
    for key in ("pixel_values", "image_grid_thw"):
        identity = tensor_identity(processed[key])
        if identity is not None:
            fields[key] = identity
    if "pixel_values" not in fields or "image_grid_thw" not in fields:
        raise RuntimeError("Cannot verify realized processor input")
    return {"crop": pixel_identity(image), "prepared": pixel_identity(prepared[0]), "tensors": fields,
            "identity": hashlib.sha256((fields["pixel_values"]["sha256"] + fields["image_grid_thw"]["sha256"]
                                         + hashlib.sha256(prompt.encode()).hexdigest()).encode()).hexdigest()}
