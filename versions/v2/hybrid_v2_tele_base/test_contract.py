import os
"""CPU fault-injection contracts, with the original upstream stepping method as oracle."""
import ast
import copy
import importlib.util
from pathlib import Path
import types
import unittest
from PIL import Image
from .tele_adapter import make_client_class
from .region_protocol import (CONFIG_SHA, IDENTITY_FIELDS, eligible, request, decode_request, termination,
                              validate_response, RunBlocked, ExpertTimeout, PauseAfterFallback)

SOURCE = Path(os.environ.get('HYBRID_TELE_SOURCE', '/srv/hybrid-assets/TeleOCR'))


class Block(dict):
    def __init__(self, kind, bbox=None, angle=0, content=None):
        super().__init__(type=kind, bbox=[0, 0, 1, 1] if bbox is None else bbox, angle=angle, content=content)
    type = property(lambda self: self['type'])
    bbox = property(lambda self: self['bbox'])
    angle = property(lambda self: self['angle'])
    content = property(lambda self: self['content'], lambda self, value: self.__setitem__('content', value))


def original_method():
    packaged = Path(__file__).resolve().parents[1] / 'SOURCE_METHOD.txt'
    if packaged.is_file():
        tree = ast.parse(packaged.read_text(encoding='utf-8'))
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'stepping_two_step_extract')
    else:
        tree = ast.parse((SOURCE / 'TeleOCR/vlm_utils/TeleOCR_client.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TeleOCRClient')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'stepping_two_step_extract')
    namespace = {}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), '<original-Tele-stepping>', 'exec'), namespace)
    return namespace[method.name]


class Native:
    batch_size = 1
    def __init__(self): self.calls = []
    def batch_predict(self, images, prompts, params, priority):
        self.calls.append((images, prompts, params, priority))
        return [f'Tele:{im.getpixel((0,0))[0]}:{prompt}' for im, prompt in zip(images, prompts)]


class Helper:
    def __init__(self): self.post_calls = 0
    def batch_prepare_for_extract(self, executor, images, blocks, skip):
        result = []
        for page, entries in enumerate(blocks):
            # Nonconsecutive indices, duplicate boxes, skipped invalid crop and equation_block.
            indices = [i for i, b in enumerate(entries) if b.type not in ('image', 'equation_block')
                       and b.get('skip_crop') is not True and b.type not in (skip or [])]
            result.append(([Image.new('RGB', (28, 28), (page*20+i, 2, 3)) for i in indices],
                           ['prompt:'+entries[i].type for i in indices], [{'index':i} for i in indices], indices))
        return result
    def batch_post_process(self, executor, blocks):
        self.post_calls += 1
        return blocks


class Base:
    stepping_two_step_extract = original_method()
    def __init__(self, blocks):
        self.blocks = blocks
        self.backend, self.batching_mode, self.incremental_priority = 'transformers', 'stepping', False
        self.client, self.helper, self.executor, self.layout_calls = Native(), Helper(), None, 0
    def batch_layout_detect(self, images, priority):
        self.layout_calls += 1
        return copy.deepcopy(self.blocks)


def good_response(sent, text='  \\[x=1\\]  ', tokens=None):
    result = {k: sent[k] for k in IDENTITY_FIELDS}
    result.update(status='ok', raw_text=text, generation=termination(tokens or [12, 2], [2]))
    return result


class Expert:
    def __init__(self, operation=good_response): self.calls, self.operation = [], operation
    def recognize(self, sent):
        self.calls.append(sent)
        return self.operation(sent)


class Contracts(unittest.TestCase):
    def fixture(self):
        skipped = Block('equation'); skipped['skip_crop'] = True
        blocks = [[Block('text', content='inline $x$'), Block('image'), Block('equation'), skipped,
                   Block('equation_block'), Block('table'), Block('equation')],
                  [Block('equation'), Block('text'), Block('equation', angle=None)]]
        images = [Image.new('RGB', (100, 100)), Image.new('RGB', (100, 100))]
        return blocks, images
    def adapter(self, blocks, mode, expert=None):
        events = []
        client = make_client_class(Base)(blocks, hybrid_mode=mode, formula_expert=expert, emit=events.append)
        client.set_context('run', 'page', 'a'*64)
        return client, events

    def test_off_and_pass_through_match_original_with_holes_duplicate_boxes_and_pages(self):
        blocks, images = self.fixture()
        raw = Base(blocks); expected = raw.stepping_two_step_extract(images)
        for mode in ('off', 'pass-through'):
            adapter, events = self.adapter(blocks, mode)
            self.assertEqual(adapter.stepping_two_step_extract(images), expected)
            self.assertEqual(adapter.layout_calls, 1)
            self.assertEqual(adapter.helper.post_calls, 1)
            self.assertEqual(len(adapter.client.calls), 1)
            if mode == 'pass-through':
                self.assertEqual([(e['page_ordinal'],e['block_index']) for e in events if e['event']=='route'],
                                 [(0,0),(0,2),(0,5),(0,6),(1,0),(1,1),(1,2)])

    def test_only_equation_replaced_and_native_arguments_preserved(self):
        blocks, images = self.fixture(); expert = Expert()
        raw = Base(blocks); expected = raw.stepping_two_step_extract(images)
        client, events = self.adapter(blocks, 'on', expert)
        actual = client.stepping_two_step_extract(images)
        self.assertEqual(len(expert.calls), 3)
        self.assertEqual(len({r['region_id'] for r in expert.calls}), 3)
        accepted = {(0,2), (0,6), (1,0)}
        for p, entries in enumerate(actual):
            for i, entry in enumerate(entries):
                if (p,i) in accepted:
                    self.assertEqual(entry.content, '  \\[x=1\\]  ')
                    self.assertEqual({k:v for k,v in entry.items() if k!='content'},
                                     {k:v for k,v in expected[p][i].items() if k!='content'})
                else: self.assertEqual(entry, expected[p][i])
        native_prompts = [prompt for call in client.client.calls for prompt in call[1]]
        self.assertEqual(native_prompts, ['prompt:text','prompt:table','prompt:text','prompt:equation'])
        self.assertEqual(client.helper.post_calls, 1)

    def test_route_metadata_boundaries(self):
        image = Image.new('RGB', (28, 28))
        for block in [Block('text'), Block('equation_block'), Block('equation', angle=None),
                      Block('equation', bbox=[0,0,1,0,1,1,0,1]), Block('equation', bbox=[0,0,float('nan'),1]),
                      Block('equation', bbox=[-0.1,0,1,1])]:
            self.assertFalse(eligible(block, image)[0])
        self.assertFalse(eligible(Block('equation'), Image.new('L',(28,28)))[0])
        self.assertTrue(eligible(Block('equation', angle=90), image)[0])

    def test_rgb_roundtrip_and_hash_color_failures_stop(self):
        image = Image.new('RGB', (31,29), (230,17,2))
        sent = request('run','page','a'*64,0,4,image)
        decoded = decode_request(sent)
        self.assertEqual(decoded.tobytes(), image.tobytes())
        self.assertEqual(decoded.getpixel((0,0)), (230,17,2))
        for field, bad in [('crop_sha256','bad'),('shape',[31,29,3]),('image_mode','BGR')]:
            changed = dict(sent); changed[field] = bad
            with self.assertRaises(RunBlocked): decode_request(changed)

    def test_eos_at_cap_length_and_other(self):
        self.assertEqual(termination([1]*4095+[2],[2])['stop'], 'eos')
        self.assertEqual(termination([1]*4096,[2])['stop'], 'length')
        self.assertEqual(termination([1]*30,[2])['stop'], 'other')
        with self.assertRaises(RunBlocked): termination([1]*4096+[2],[2])

    def test_response_wrong_id_missing_id_and_unknown_exception_stop(self):
        image=Image.new('RGB',(28,28)); sent=request('r','p','a'*64,0,0,image)
        for response in [dict(good_response(sent),region_id='wrong'), {'status':'ok'},
                         dict(good_response(sent),status='fatal',error='CUDA OOM'), dict(good_response(sent),status='mystery')]:
            with self.assertRaises(RunBlocked): validate_response(sent,response)

    def test_expected_failure_falls_back_once_with_original_slot_arguments(self):
        for failure in ('empty','length','other','decode_error'):
            blocks=[[Block('equation')]]; images=[Image.new('RGB',(50,50))]
            expert=Expert(lambda sent: dict(good_response(sent),status=failure))
            client,events=self.adapter(blocks,'on',expert)
            result=client.stepping_two_step_extract(images)
            self.assertEqual(result[0][0].content,'Tele:0:prompt:equation')
            self.assertEqual(len(client.client.calls),1)
            self.assertEqual(len(expert.calls),1)
            self.assertEqual(next(e for e in events if e['event']=='route')['route'],'tele_fallback')

    def test_timeout_saves_current_fallback_then_pauses_without_next_request(self):
        def timeout(sent): raise ExpertTimeout('owned worker stopped')
        client,events=self.adapter([[Block('equation'),Block('equation')]],'on',Expert(timeout))
        with self.assertRaises(PauseAfterFallback): client.stepping_two_step_extract([Image.new('RGB',(50,50))])
        self.assertEqual(len(client.formula_expert.calls),1)
        self.assertEqual(events[-1]['tele_fallback_text'],'Tele:0:prompt:equation')
        self.assertEqual(events[-1]['reason'],'timeout')

    def test_unknown_oom_exception_not_silently_fallback(self):
        def fatal(sent): raise RuntimeError('CUDA out of memory')
        client,_=self.adapter([[Block('equation')]],'on',Expert(fatal))
        with self.assertRaises(RuntimeError): client.stepping_two_step_extract([Image.new('RGB',(50,50))])
        self.assertEqual(len(client.client.calls),0)

    def test_zero_equations_no_expert_requests(self):
        client,_=self.adapter([[Block('text'),Block('table')]],'on',Expert())
        client.stepping_two_step_extract([Image.new('RGB',(50,50))])
        self.assertEqual(client.formula_expert.calls,[])


if __name__ == '__main__': unittest.main(verbosity=2)
