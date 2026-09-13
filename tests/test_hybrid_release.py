"""Model-free release checks; every text/image/geometry fixture is project-authored."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'hybrid'))
import common
import correct_table
import locate_tokens


RAW = '<table class="keep"><tr><td>AnchorA</td><td> 12O.5 </td><td>AnchorB</td></tr></table>'


def tokens():
    return [{'token_index': i, 'text': text, 'bbox': [i * 100, 0, i * 100 + 80, 20]}
            for i, text in enumerate(['AnchorA', '120.5', 'AnchorB'])]


def test_vendor_files_are_exact():
    assert len(common.verify_vendor()) == 4


def test_exact_consensus_commits_after_structure_check():
    _, _, apply = common.load_policy()
    r = apply.apply_table(RAW, tokens(), {1: [
        {'engine_id': 'p', 'text': '120.5'}, {'engine_id': 't', 'text': '120.5\n'}]})
    assert r['final'] == RAW.replace('12O.5', '120.5')
    assert len(r['post_commit_receipts']) == 1
    c = r['post_commit_receipts'][0]
    assert c['structure_sha256_before'] == c['structure_sha256_after']
    assert c['decision'] == 'COMMITTED_AFTER_EXACT_STRUCTURE_CHECK'


@pytest.mark.parametrize('observations', [
    {}, {1: [{'engine_id': 'p', 'text': '120.5'}, {'engine_id': 't', 'text': '120.6'}]},
    {1: [{'engine_id': 'p', 'text': '120.5'}, {'engine_id': 'p', 'text': '120.5'}]},
])
def test_missing_disagreeing_or_same_engine_never_commits(observations):
    _, _, apply = common.load_policy()
    result = apply.apply_table(RAW, tokens(), observations)
    assert result['final'] == RAW
    assert result['post_commit_receipts'] == []


@pytest.mark.parametrize('raw', [
    RAW.replace('AnchorA', ''), RAW.replace('<td>AnchorA', '<td colspan="2">AnchorA'),
    RAW.replace('AnchorA', '$x$'), RAW.replace('AnchorA', 'Anchor\nA'),
])
def test_historical_blank_merge_formula_multiline_veto(raw):
    _, grounding, _ = common.load_policy()
    assert grounding.ground_table(raw, tokens())['proposals'] == []


def envelope():
    return {'schema': 'hybrid-image-only-tokens/v1', 'status': 'ok',
            'input_image_sha256': 'a' * 64, 'image_width': 300, 'image_height': 30,
            'coordinate_space': 'absolute_xyxy_in_table_crop', 'ocr_tokens': tokens()}


def test_image_bound_tokens(tmp_path):
    p = tmp_path / 'tokens.json'
    common.write_json(p, envelope())
    assert correct_table.read_tokens(p, 'a' * 64, (300, 30))['ocr_tokens'] == tokens()
    with pytest.raises(ValueError, match='SHA'):
        correct_table.read_tokens(p, 'b' * 64, (300, 30))
    with pytest.raises(ValueError, match='dimensions'):
        correct_table.read_tokens(p, 'a' * 64, (301, 30))


def test_reference_fields_are_rejected(tmp_path):
    value = envelope()
    value['locator'] = {'nested': {'gold': 'must not be accepted'}}
    p = tmp_path / 'bad.json'
    common.write_json(p, value)
    with pytest.raises(ValueError, match='Reference'):
        correct_table.read_tokens(p, 'a' * 64, (300, 30))


def test_crop_exact_floor_ceil_and_bounds():
    assert correct_table.integer_crop_box([1.2, 2.8, 15.1, 20.2], 30, 30) == [1, 2, 16, 21]
    for box in ([-1, 0, 1, 1], [0, 0, 31, 10], [0, 0, float('nan'), 1]):
        with pytest.raises(ValueError):
            correct_table.integer_crop_box(box, 30, 30)


def test_no_overwrite(tmp_path):
    p = tmp_path / 'file.json'
    common.write_json(p, {'first': True})
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        common.write_json(p, {'second': True})
    assert p.read_bytes() == before


def test_locator_exact_fallback_and_original_coordinate_parser():
    common.verify_vendor()
    source = common.load_module('_test_locator', ROOT / 'hybrid/vendor/frozen_paddleocr_locator.py')
    config = locate_tokens.relocated_config(Path('/not/a/real/model/root'))
    at_limit = source.effective_inference_policy(config, {'image_width': 3000, 'image_height': 1000})
    large = source.effective_inference_policy(config, {'image_width': 3001, 'image_height': 1000})
    assert at_limit['predict_overrides'] == {}
    assert large['predict_overrides'] == {'text_det_limit_type': 'max', 'text_det_limit_side_len': 640}
    parsed, counts = source.parse_ocr_tokens({'rec_texts': ['A', ''], 'rec_scores': [0.1, 0.9],
        'rec_polys': [[[10, 20], [40, 20], [40, 35], [10, 35]], [[1, 1], [2, 1], [2, 2], [1, 2]]]}, 100, 50)
    assert parsed[0]['bbox'] == [10.0, 20.0, 40.0, 35.0]
    assert counts['retained'] == 1 and counts['dropped_empty_text'] == 1


@pytest.mark.parametrize('script', ['correct_table.py', 'locate_tokens.py'])
def test_help_does_not_load_models(script):
    result = subprocess.run([sys.executable, str(ROOT / 'hybrid' / script), '--help'],
                            check=True, capture_output=True, text=True, timeout=30)
    assert '--model-dir' in result.stdout and '--output' in result.stdout


def test_existing_output_fails_before_any_model_import(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'hybrid/correct_table.py'),
        '--html', 'absent', '--table-image', 'absent', '--ocr-tokens', 'absent',
        '--model-dir', 'absent', '--output', str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and 'FileExistsError' in result.stderr
    assert list(tmp_path.iterdir()) == []


def make_inputs(tmp_path, raw=RAW):
    from PIL import Image
    image = tmp_path / 'image.png'
    Image.new('RGB', (300, 30), 'white').save(image)
    html = tmp_path / 'raw.html'
    html.write_bytes(raw.encode('utf-8'))
    value = envelope()
    value['input_image_sha256'] = common.sha(image)
    token_file = tmp_path / 'tokens.json'
    common.write_json(token_file, value)
    return SimpleNamespace(html=str(html), table_image=str(image),
        ocr_tokens=str(token_file), model_dir=str(tmp_path / 'fake-models'),
        output=str(tmp_path / 'output'))


def test_complete_wrapper_with_fake_recognizers_preserves_structure(tmp_path, monkeypatch):
    args = make_inputs(tmp_path)
    monkeypatch.setattr(correct_table, 'verify_runtime', lambda _: {'fake': True})
    observed = []

    def fake(image, grounded, runtime, output):
        observed.append((image.size, grounded['proposals'][0]['cell_index']))
        return {1: [{'engine_id': 'fake-a', 'text': '120.5'},
                    {'engine_id': 'fake-b', 'text': '120.5'}]}, [{'test_fake': True}]

    monkeypatch.setattr(correct_table, 'recognize_proposals', fake)
    correct_table.run(args)
    out = Path(args.output)
    assert observed == [((300, 30), 1)]
    assert (out / 'raw.html').read_bytes() == RAW.encode()
    assert (out / 'final.html').read_text() == RAW.replace('12O.5', '120.5')
    assert (out / 'READY.json').is_file()
    assert not (out / 'FAILURE.json').exists()


def test_runtime_failure_is_not_raw_success(tmp_path, monkeypatch):
    args = make_inputs(tmp_path)

    def fail(_):
        raise RuntimeError('deliberate fake runtime failure')

    monkeypatch.setattr(correct_table, 'verify_runtime', fail)
    with pytest.raises(RuntimeError, match='fake runtime'):
        correct_table.run(args)
    out = Path(args.output)
    assert (out / 'FAILURE.json').is_file()
    assert not (out / 'READY.json').exists()
    assert not (out / 'final.html').exists()


def test_unsupported_html_is_explicit_raw_preservation(tmp_path, monkeypatch):
    raw = '<table><tr><td>unclosed'
    args = make_inputs(tmp_path, raw)
    monkeypatch.setattr(correct_table, 'verify_runtime', lambda _: {'fake': True})

    def never(*args):
        raise AssertionError('Unsupported HTML must not trigger OCR')

    monkeypatch.setattr(correct_table, 'recognize_proposals', never)
    correct_table.run(args)
    out = Path(args.output)
    assert (out / 'final.html').read_bytes() == raw.encode()
    assert 'UNSUPPORTED_HTML_RAW_PRESERVED' in (out / 'RECEIPT.json').read_text()


def test_input_mutation_stops_before_final(tmp_path, monkeypatch):
    args = make_inputs(tmp_path)
    monkeypatch.setattr(correct_table, 'verify_runtime', lambda _: {'fake': True})

    def mutate(image, grounded, runtime, output):
        Path(args.html).write_bytes(b'changed')
        return {}, []

    monkeypatch.setattr(correct_table, 'recognize_proposals', mutate)
    with pytest.raises(RuntimeError, match='Input changed'):
        correct_table.run(args)
    out = Path(args.output)
    assert (out / 'FAILURE.json').is_file()
    assert not (out / 'READY.json').exists()
    assert not (out / 'final.html').exists()
