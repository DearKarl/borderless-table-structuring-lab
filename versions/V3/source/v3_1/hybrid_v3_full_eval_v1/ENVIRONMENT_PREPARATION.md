# Offline image and asset preparation contract

This package performs no installation or download. Supply and validate the actual runtime environment separately before inference.

For schema 2, declaring Paddle, any frozen chardet image member, or any chardet member in auxiliary_models requires all four files from assets/chardet_capacity.json. Each image role holding them must include auxiliary_models in asset_roles; image hashes, auxiliary asset hashes and the classification identity must agree. Bind the independent chardet evidence to the canonical hash of the complete auxiliary_models file map. Its counts and classification_lock_sha256 must match the shipped classification lock, with relevant_members containing exactly those four files. Copy original resource bytes for offline evidence; keep actual image loading paths unchanged.

When the full auxiliary file map changes, fasttext/ONNX evidence may be derived only if every relevant member SHA remains identical. Retain original counts, measurement source, result SHA, member lock and unresolved gaps, and label the derivation source separately. An incomplete ledger is a draft, never an executable binding. Original measurement records remain immutable. Include 23,199,744 chardet parameters in the same 4-billion sum; static mappings and derived buffers are separately inventoried. A native session sharing the full chardet-containing auxiliary asset also requires the fourth row.

Tele model repository StarDoc-AI/TeleOCR is frozen at model revision 8706730e41382f4a8f2562616d47225fee8e2f7a. Its source repository https://github.com/caipeng328/TeleOCR is frozen at source revision 9921cffe380efe4e2fa010258b3d0c3cb70bab2d. These are distinct identities: never use the source commit as a model revision. The generated requirements expose models.tele and sources.tele separately; frozen model member hashes are unchanged.

Generate a machine-readable plan with all known exact versions/model/source/auxiliary hashes and explicit missing image/wheel/parameter evidence:

    python -B -m hybrid_v3_full_eval_v1.cli environment-requirements --output NEW_REQUIREMENTS.json

## Inputs to prepare-assets

The configuration supplies:
- profile and mode; native, v31-layout, v31-formula or v31-both.
- asset_roots: dedicated tele_model; optional layout_model/formula_model; environment (small evidence/config directory, not a relocated venv); parameter_evidence; any separate auxiliary model inventories referenced by the ledger. Supply `tele_source` explicitly as the pinned upstream checkout root. Only the locked inference subset is copied. Project-owned native support is included.
- environment_lock: JSON with native and, for an expert profile, paddle records.
- parameter_ledger: JSON array with one row for tele, fasttext and onnx, plus each declared expert.
- explicit lease_directory, idle_memory_mib, UUIDs, full validation_budget.

Paths in the preparation configuration may explicitly identify local directories. The bound runtime uses only relative asset paths and fixed image-local interpreter/source paths. Input symlinks resolving outside a declared root fail. Output directories must be fresh.

## Environment-lock JSON

Each role requires image (immutable name@sha256:digest), python (container entrypoint), resolved_python (actual image-local executable), site_packages, image_files (absolute container path → SHA256), packages (distribution → exact version), asset_roles (list of copied asset roles to reverify at load).

Use a separately provisioned immutable native image and immutable Paddle image. The lock must cover the resolved executable, installed source/native libraries and auxiliary model files; include its original wheel/resolver receipt in the environment evidence asset. An arbitrary mounted venv is not a valid environment. Resolve interpreter links inside the image and include both their source/target identity in the preparation evidence. Every image_files member is rehashed inside its owning worker before load, and the actual interpreter resolution is compared exactly.

Frozen native versions and auxiliary hashes are in assets/vendor/PARENT_ASSET_LOCK.json, including torch 2.12.0+cu130, transformers 4.57.6 and pypdfium2 4.30.0. Paddle versions include paddlepaddle-gpu 3.2.1, paddlex 3.6.1 and paddleocr 3.6.0; its full original declared subset is in that parent lock. PP-DocLayoutV3 revision and pinned source members are in layout_metadata.json. Do not substitute PP-DocLayoutV2 or change the native generation recipe.

If a matching image, wheel archive, Python executable or installed-file receipt is missing, preserve the failure and prepare those exact locked dependencies in a new image with independently validated dependency locks. Do not choose newer versions, upgrade a shared environment or declare an unknown image digest. Image availability must be validated on the deployment host.

## Parameter ledger

Each row requires component, model_asset, model_files_sha256 (canonical SHA256 of that asset's file lock), stored_elements, unique_trainable, buffers and evidence {asset, path, sha256}. The evidence JSON repeats those exact measured fields and includes method and measurement_source_sha256. Supported methods are loaded_unique_parameters, static_graph_parameters and exact_auxiliary_inventory. Evidence must originate from the actual frozen resource, not an estimate based on weight-file bytes or a passed boolean.

Tele and Paddle loaded unique parameters/buffers are checked against the evidence at load. Layout must have exact static-graph/loaded evidence for PP-DocLayoutV3. Fasttext and all ONNX auxiliaries must be included. Separate auxiliary inventories can be copied for evidence while their deployed image-local paths are independently hash-bound. Never put a ledger evidence file inside the model asset whose whole-file-lock hash it references, which would create a circular hash dependency.

Known context only: Tele unique parameter count was 1,259,490,304; Paddle stored elements were 958,588,736. These numbers do not fill missing buffer, layout or auxiliary measurements. An incomplete ledger fails before inference.

## License and source provenance

TeleOCR source is not redistributed. Supply upstream revision `9921cffe380efe4e2fa010258b3d0c3cb70bab2d` and retain its applicable terms. The binder checks every required member against the original hashes. Project-owned native support is included; the private parent freeze is not distributed.

Tele and Paddle model cards declare Apache-2.0; retain their actual model license/card files when preparing assets. PP-DocLayoutV3 and installed Paddle source license/notice records must accompany the separately supplied model/environment inventory. This package does not redistribute any weights or installed environment.

## Untested deployment boundaries

Image availability, interpreter/native-library compatibility, static Paddle loaded-graph counts, auxiliary load behavior, real preprocessing hook compatibility, GPU UUID checks and actual PDF/full-page outputs remain unverified for a new deployment. Synthetic CPU evidence must not be used as evidence of real runtime compatibility.

## V31 local immutable image identity (0.2.2-v31-imageid1)

Schema 2 accepts a complete lowercase sha256 image ID or canonical repository@sha256 digest; schema 1 remains repository-only. Never substitute a tag or fabricate RepoDigests from a config ID. Offline prepare/bind stays file-only. Before any create, inference resolves each effective role once with local docker image inspect under the existing deadline (at most 10 seconds each), without pulling. Off/pass-through resolves only native. Active native/Paddle IDs must differ. Container Config.Image must equal the locked reference and its top-level Image must equal the resolved ID; CID ownership and resource isolation remain mandatory. IMAGE_IDENTITIES.json is sealed into lifecycle evidence with image_identity_schema=1. Official evaluation retains its original repository pin and saves separate image evidence SHA. After save/load migration, recheck the actual image ID; tags are not identity locks. CPU checks do not establish OS-library, GPU, model or scoring compatibility.
# Explicit host driver binding

An optional schema-2 manual backend requires a separate schema-1 JSON descriptor:
`schema`, `driver_version`, `bundle_root`, `files`, `aliases`, and `provenance`.
Each file has `soname`, `target_basename`, `host_source`, and lowercase SHA256.
Aliases are a closed map of names to exact relative basenames, including soname
links. Provenance contains `driver_receipt_sha256` and `freeze_sha256`. Unknown
fields, extra directory entries, unlisted links, stubs/compat, and device files
are rejected. Supply the actual local descriptor on every deployment; no driver
binary or private deployment path belongs in the source package.

Run the host-only `bind-host-driver` CLI on the target host after explicitly
providing this specification. Then add `gpu_backend.kind=manual` and the produced
JSON's path and SHA to the prepare configuration. The binder copies only the JSON.
Real inference still validates the actual files, kernel, GPU XML and devices.
The NVIDIA backend stays available without this file and never falls back to manual.

`GPU_BINDINGS.json`, sealed `HOST_DRIVER_BINDING.json`, per-role
`GPU_BEFORE_START.json` and `GPU_BINDING_CHECK.json` are mandatory lifecycle
evidence for manual runs. Both CONTROL and RUN_MANIFEST declare
`gpu_binding_schema=1`. Removing both markers cannot bypass the runtime's explicit
manual backend requirement. Parameter evidence and the four-billion-parameter gate
remain separate prerequisites; import compatibility is insufficient.
