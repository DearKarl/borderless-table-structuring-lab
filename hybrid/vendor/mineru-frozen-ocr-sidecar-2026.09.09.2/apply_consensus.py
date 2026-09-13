"""Exact-span, structure-preserving commit for anchored OCR observations.

No model, label, metric, or confidence is consulted by this module. Geometric
proposals are recomputed from the same raw table and image-only detector output.
"""
import hashlib
import html
from pathlib import Path
import sys

LEGACY = Path(__file__).resolve().parent.parent / 'mineru-frozen-ocr-sidecar-2026.09.09.1'
EXPECTED_CORE = '07c5fc2e82ae4136d36dc52b722c1dda686eee7a83ac79a54e609ef3a39c5334'
if hashlib.sha256((LEGACY / 'core.py').read_bytes()).hexdigest() != EXPECTED_CORE:
    raise RuntimeError('Immutable parser binding changed')
sys.path.insert(0, str(LEGACY))
from core import _parse, _patch, _json_sha, _sha  # noqa: E402
from grounding import ground_table  # noqa: E402


def apply_table(raw, tokens, observations):
    """Commit only freshly re-recognized, unambiguously grounded proposals.

observations is keyed by DOM cell index, with exactly two distinct OCR engines.
Both strings must exactly match the grounded detector string after outer trim.
The detector is a locator, not an independent third vote.
"""
    grounded = ground_table(raw, tokens)
    parser = _parse(raw)
    skeleton = _json_sha(parser.markup)
    patches, commits, rejected = [], [], []
    for proposal in grounded['proposals']:
        idx = proposal['cell_index']
        obs = observations.get(idx)
        if (not isinstance(obs, list) or len(obs) != 2 or
                len({o.get('engine_id') for o in obs}) != 2 or
                any(not isinstance(o.get('engine_id'), str) or not o['engine_id'] or
                    not isinstance(o.get('text'), str) for o in obs)):
            rejected.append({'cell_index': idx, 'reason': 'MISSING_TWO_DISTINCT_ENGINES'})
            continue
        target = proposal['detector_text']
        if any(o['text'].strip() != target for o in obs):
            rejected.append({'cell_index': idx, 'reason': 'FRESH_ENGINE_DISAGREEMENT'})
            continue
        cell = parser.cells[idx]
        inner = raw[cell['start']:cell['end']]
        # Keep boundary whitespace byte-exact. Only the cell's text may change.
        leading = inner[:len(inner) - len(inner.lstrip())]
        trailing = inner[len(inner.rstrip()):]
        replacement = leading + html.escape(target, quote=True) + trailing
        if inner == replacement:
            continue
        patches.append((cell['start'], cell['end'], replacement))
        commits.append(dict(proposal, ocr=obs, original_inner_sha256=_sha(inner),
                            committed_inner_sha256=_sha(replacement)))
    final = _patch(raw, patches)
    after = _parse(final)
    if (len(after.cells) != len(parser.cells) or
            _json_sha(after.markup) != skeleton):
        raise RuntimeError('STRUCTURE_MISMATCH: no output is admitted')
    for commit in commits:
        commit.update(decision='COMMITTED_AFTER_EXACT_STRUCTURE_CHECK',
                      raw_table_sha256=_sha(raw), final_table_sha256=_sha(final),
                      structure_sha256_before=skeleton, structure_sha256_after=skeleton)
    return {'raw_sha256': _sha(raw), 'final_sha256': _sha(final), 'final': final,
            'structure_sha256': skeleton, 'post_commit_receipts': commits,
            'rejected': rejected, 'grounding': grounded}
