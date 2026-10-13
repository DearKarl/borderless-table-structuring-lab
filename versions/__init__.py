"""Compatibility imports for scientific sources consolidated under Version/."""
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
_LOCATIONS = {
    "v0": "V0", "v1": "V1", "v2": "V2", "v3_1": "V3", "v3_2": "V3",
    "v5": "V5", "v5_native": "V6", "v7": "V7", "generator_pilot": "V8",
}


class _VersionSources(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        if len(parts) != 2 or parts[0] != __name__ or parts[1] not in _LOCATIONS:
            return None
        name = parts[1]
        source = _ROOT / "Version" / _LOCATIONS[name] / "source" / name
        local = _ROOT / "versions" / name
        search = [str(p) for p in (source, local) if p.is_dir()]
        if not search:
            return None
        init = source / "__init__.py"
        if init.exists():
            return spec_from_file_location(fullname, init, submodule_search_locations=search)
        spec = ModuleSpec(fullname, loader=None, is_package=True)
        spec.submodule_search_locations = search
        return spec


if not any(isinstance(finder, _VersionSources) for finder in sys.meta_path):
    sys.meta_path.insert(0, _VersionSources())
