"""CPU-only tests for observation transparency and pre-decode configuration stops."""
import unittest
from .audit import install_tele_audit
from .region_protocol import RunBlocked


class Tensor:
    dtype = 'int64'
    def __init__(self, rows): self.rows, self.shape = rows, (len(rows), len(rows[0]))
    def cpu(self): return self
    def detach(self): return self
    def tolist(self): return self.rows
    def numpy(self): raise TypeError('Exercise numeric fallback')


class Config:
    def __init__(self, **changes):
        self.values = dict(max_length=128000, max_new_tokens=None, do_sample=False,
                           repetition_penalty=1.0, no_repeat_ngram_size=100, eos_token_id=2)
        self.values.update(changes)
    def to_dict(self): return dict(self.values)


class Model:
    def __init__(self, **changes):
        self.config, self.decoded = Config(**changes), False
    def _prepare_generation_config(self): return self.config, {}
    def _prepare_generated_length(self, config): return config
    def generate(self, **kwargs):
        config, _ = self._prepare_generation_config()
        self._prepare_generated_length(config)
        self.decoded = True
        self.received = kwargs
        return Tensor([kwargs['input_ids'].tolist()[0] + [7, 2]])


class AuditTests(unittest.TestCase):
    def test_valid_call_preserves_input_identity_return_and_restore(self):
        model, events, inputs = Model(), [], Tensor([[3, 4]])
        original = model.generate
        restore = install_tele_audit(model, events.append)
        result = model.generate(input_ids=inputs, use_cache=True)
        self.assertIs(model.received['input_ids'], inputs)
        self.assertEqual(result.tolist(), [[3, 4, 7, 2]])
        self.assertTrue(events[0]['returned'])
        self.assertEqual(events[0]['effective_generation_config']['max_length'], 128000)
        restore()
        self.assertEqual(model.generate, original)

    def test_each_invalid_setting_stops_before_decode_and_preserves_failure(self):
        for changes in [dict(max_length=4096), dict(max_new_tokens=4096),
                        dict(do_sample=True), dict(repetition_penalty=1.1),
                        dict(no_repeat_ngram_size=0)]:
            with self.subTest(changes=changes):
                model, events = Model(**changes), []
                install_tele_audit(model, events.append)
                with self.assertRaises(RunBlocked): model.generate(input_ids=Tensor([[3, 4]]))
                self.assertFalse(model.decoded)
                self.assertEqual(len(events), 1)
                self.assertFalse(events[0]['returned'])
                self.assertEqual(events[0]['error_type'], 'RunBlocked')
                for key, value in changes.items():
                    self.assertEqual(events[0]['effective_generation_config'][key], value)


if __name__ == '__main__': unittest.main()
