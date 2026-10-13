"""Strict immutable image references and local-only launch identity evidence."""
import json
import re
import subprocess
from .core import ContractError

_ID = re.compile(r"sha256:[0-9a-f]{64}")
_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*"
_REPOSITORY = re.compile(r"(?:[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?/)?"
                         + _COMPONENT + r"(?:/" + _COMPONENT + r")*")

def parse_reference(reference, allow_local=True):
    if not isinstance(reference, str):
        raise ContractError("Immutable image reference must be text")
    if _ID.fullmatch(reference):
        if not allow_local:
            raise ContractError("Schema 1 requires repository digest")
        return "local_id"
    if reference.count("@") == 1:
        repository, value = reference.split("@")
        if _REPOSITORY.fullmatch(repository) and _ID.fullmatch(value):
            return "repo_digest"
    raise ContractError("Expected complete lowercase image ID or canonical repository@sha256 digest")

def validate_identity(reference, record):
    kind = parse_reference(reference)
    if not isinstance(record, dict) or record.get("reference") != reference or record.get("kind") != kind:
        raise ContractError("Image identity reference/kind mismatch")
    image_id = record.get("image_id")
    if not isinstance(image_id, str) or not _ID.fullmatch(image_id):
        raise ContractError("Missing complete image inspect Id")
    digests = record.get("RepoDigests")
    if not isinstance(digests, list) or any(not isinstance(v, str) for v in digests):
        raise ContractError("Invalid image RepoDigests")
    if kind == "local_id" and image_id != reference:
        raise ContractError("Local image ID differs from locked reference")
    if kind == "repo_digest" and reference not in digests:
        raise ContractError("Locked repository digest absent from local image")
    for field in ("Os", "Architecture", "Variant"):
        if field in record and (not isinstance(record[field], str) or not record[field]):
            raise ContractError("Invalid image platform field: " + field)
    return record

def resolve_image_identity(reference, deadline, runner=None):
    kind = parse_reference(reference)
    if runner is None:
        runner = subprocess.run
    result = runner(["docker", "image", "inspect", reference], text=True,
                    capture_output=True, check=True, timeout=deadline.bound(10))
    deadline.remaining()
    if result.returncode != 0:
        raise ContractError("Local image inspect failed")
    try:
        objects = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise ContractError("Invalid image inspect JSON") from exc
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
        raise ContractError("Image inspect must return exactly one object")
    obj = objects[0]
    record = {"reference": reference, "kind": kind, "image_id": obj.get("Id"),
              "RepoDigests": obj.get("RepoDigests", [])}
    for field in ("Os", "Architecture", "Variant"):
        if field in obj:
            record[field] = obj[field]
    return validate_identity(reference, record)

def effective_references(runtime, mode):
    if mode not in ("on", "off", "pass-through"):
        raise ContractError("Unknown image routing mode")
    references = {"native": runtime["image"]}
    if mode == "on" and runtime.get("v31", {}).get("components"):
        references["paddle"] = runtime["environments"]["paddle"]["image"]
    return references

def validate_identities(evidence, run_id, references):
    if (not isinstance(evidence, dict) or type(evidence.get("schema")) is not int
            or evidence.get("schema") != 1 or evidence.get("run_id") != run_id
            or not isinstance(evidence.get("roles"), dict)
            or set(evidence["roles"]) != set(references)):
        raise ContractError("Image identity run/role set differs")
    for role, reference in references.items():
        validate_identity(reference, evidence["roles"][role])
    if len({r["image_id"] for r in evidence["roles"].values()}) != len(references):
        raise ContractError("Independent environments resolve to the same image ID")
    return evidence

def resolve_identities(run_id, references, deadline, runner=None):
    evidence = {"schema": 1, "run_id": run_id,
                "roles": {role: resolve_image_identity(ref, deadline, runner)
                          for role, ref in references.items()}}
    return validate_identities(evidence, run_id, references)

def verify_container_image(state, reference, identity):
    validate_identity(reference, identity)
    if (state.get("Image") != identity["image_id"]
            or state.get("Config", {}).get("Image") != reference):
        raise ContractError("Container actual image ID/reference differs")

