"""Explicit adapter callback interface; actual inference must supply fixed adapters."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from .core import assemble, sha


@dataclass(frozen=True)
class NativePrediction:
    model: str
    status: str
    raw: bytes
    native: dict | None
    provenance: dict


@dataclass(frozen=True)
class FixedAdapter:
    model: str
    source_sha256: str
    runtime_identity: dict
    infer: Callable[[Path,int,int],NativePrediction]


def run_image(image:Path,width:int,height:int,M:FixedAdapter,P:FixedAdapter):
    """No default/fake inference: caller must bind independently frozen adapters."""
    assert M.model=='M' and P.model=='P'
    assert all(len(a.source_sha256)==64 and a.runtime_identity for a in [M,P])
    m=M.infer(image,width,height);p=P.infer(image,width,height)
    assert m.model=='M' and p.model=='P'
    out,receipt=assemble(m.raw,m.native,p.native,width,height,m.status,p.status)
    receipt.update(mode='adapter_interface_actual_calls',input_image_sha256=sha(image.read_bytes()),
        adapter_bindings={a.model:{'source_sha256':a.source_sha256,'runtime_identity':a.runtime_identity} for a in [M,P]},
        native_provenance={'M':m.provenance,'P':p.provenance})
    return out,receipt


PLUGIN_INTERFACES={'formula':'implemented-v0','table':'reserved-not-implemented','text':'reserved-not-implemented','layout_order':'reserved-not-implemented'}
