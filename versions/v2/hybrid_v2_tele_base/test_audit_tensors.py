"""Run in the pinned Tele environment, with real CPU Torch tensors only."""
import json
import unittest
import torch
from tokenizers import AddedToken
from .audit import audit_snapshot, install_tele_audit
from .test_audit import Model


class RealTensorAuditTests(unittest.TestCase):
    def test_real_small_and_scalar_tensors_keep_values_and_leave_objects_unchanged(self):
        eos=torch.tensor([151645,151643]);bos=torch.tensor(151643)
        snapshot=audit_snapshot({'_eos_token_tensor':eos,'_bos_token_tensor':bos,'max_length':128000})
        decoded=json.loads(json.dumps(snapshot,allow_nan=False))
        self.assertEqual(decoded['_eos_token_tensor']['values'],[151645,151643])
        self.assertEqual(decoded['_bos_token_tensor']['values'],151643)
        self.assertEqual(eos.tolist(),[151645,151643]);self.assertEqual(bos.item(),151643)
        self.assertEqual(decoded['max_length'],128000)

    def test_large_bfloat16_tensor_is_hashed_not_expanded(self):
        tensor=torch.arange(1024,dtype=torch.float32).to(torch.bfloat16)
        before=tensor.clone();snapshot=audit_snapshot(tensor)
        self.assertNotIn('values',snapshot)
        self.assertEqual(len(snapshot['sha256']),64)
        self.assertEqual(snapshot['shape'],[1024])
        self.assertTrue(torch.equal(before,tensor))
        json.dumps(snapshot,allow_nan=False)

    def test_real_config_tensor_survives_hook_and_json(self):
        model=Model();model.config.values['_bos_token_tensor']=torch.tensor(3)
        model.config.values['_eos_token_tensor']=torch.tensor([2])
        events=[];install_tele_audit(model,events.append)
        model.generate(input_ids=torch.tensor([[3,4]]))
        decoded=json.loads(json.dumps(events,allow_nan=False))
        self.assertEqual(decoded[0]['effective_generation_config']['_eos_token_tensor']['values'],[2])
        self.assertIsInstance(model.config.values['_eos_token_tensor'],torch.Tensor)

    def test_token_semantics_and_unknown_value_rejection(self):
        token=AddedToken('<image>',special=True,lstrip=True)
        saved=json.loads(json.dumps(audit_snapshot(token)))
        self.assertEqual(saved['content'],'<image>');self.assertTrue(saved['special']);self.assertTrue(saved['lstrip'])
        with self.assertRaises(TypeError):audit_snapshot(object())


if __name__=='__main__':unittest.main()
