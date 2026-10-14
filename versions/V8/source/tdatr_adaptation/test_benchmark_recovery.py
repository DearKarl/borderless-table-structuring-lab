"""Regression checks for native keyword dispatch and duplicate-call exclusion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from .benchmark_recovery_contract import ERROR, digest, native_keyword_adapter, validate_prior


class RecoveryTests(unittest.TestCase):
    def test_native_keyword_dispatch_preserves_objects_and_extra_arguments(self):
        images, prompts, parameters, extra = [object()], ['frozen chat prompt'], object(), object()
        received = []
        def internal(images, prompts, parameters, **kwargs):
            received.append((images, prompts, parameters, kwargs))
            return ['unchanged native output']
        native_call = dict(image_objs=images, chat_prompts=prompts, sampling_params=parameters, extra=extra)
        with self.assertRaises(TypeError):
            internal(**native_call)
        result = native_keyword_adapter(internal)(**native_call)
        self.assertEqual(result, ['unchanged native output'])
        self.assertIs(received[0][0], images)
        self.assertIs(received[0][1], prompts)
        self.assertIs(received[0][2], parameters)
        self.assertIs(received[0][3]['extra'], extra)
        self.assertEqual(len(received), 1)

    def test_recovery_uses_compact_diagnosis_and_rejects_any_prior_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root/'complete.md'; complete.write_text('preserved non-table result')
            empty = root/'empty.md'; empty.write_text('')
            pages = [dict(page_id=str(i), input_sha256=str(i)) for i in range(1651)]
            rows = []
            for i, page in enumerate(pages):
                path = complete if i < 1189 else empty
                row = dict(page, model_calls=0, tables=[], outputs={'candidate': {'file':str(path), 'sha256':digest(path)}})
                if i < 1189: row.update(status='complete', predicted_tables=0)
                else: row.update(status='failed', error_type='TypeError', reason=ERROR)
                rows.append(row)
            prior = dict(status='failed', model_calls=0, full_denominator=1651, pages=rows,
                version='V8.4', checkpoint_sha256='checkpoint', benchmark_inputs_sha256='inputs',
                shared_selection_sha256='shared', shared_cache_index_sha256='cache', selection_sha256='nomination',
                first_generation_receipt_exists=False,calls_file_exists=False,benchmark_sha256='original')
            source = root/'RECOVERY_PLAN.json'
            binding = {key: prior[key] for key in ['version','checkpoint_sha256','benchmark_inputs_sha256','shared_selection_sha256','shared_cache_index_sha256']}
            binding.update(nomination_sha256='nomination', recovery_plan_path=str(source),recovery_from_benchmark_sha256='original')
            def save(value):
                source.write_text(json.dumps({'versions':{'V8.4':value}})); binding['recovery_plan_sha256']=digest(source)
            save(prior)
            checked = validate_prior(binding, pages)
            self.assertEqual(sum(row['status']=='complete' for row in checked.values()),1189)
            self.assertEqual(checked['0']['outputs'],rows[0]['outputs'])
            changed=copy.deepcopy(prior);changed['model_calls']=1;save(changed)
            with self.assertRaisesRegex(ValueError,'zero original model calls'):validate_prior(binding,pages)
            changed=copy.deepcopy(prior);changed['pages'][1189]['reason']='different failure';save(changed)
            with self.assertRaisesRegex(ValueError,'unregistered failure'):validate_prior(binding,pages)
            save(prior)
            with self.assertRaisesRegex(ValueError,'page order'):validate_prior(binding,list(reversed(pages)))
            changed=dict(binding,checkpoint_sha256='another model')
            with self.assertRaisesRegex(ValueError,'frozen identity'):validate_prior(changed,pages)


if __name__ == '__main__':
    unittest.main()
