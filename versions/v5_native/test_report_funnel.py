"""Quality funnel counts only canonical edits and unions page identities."""
import unittest
from .report_funnel import edited_outcomes


class ReportFunnelContracts(unittest.TestCase):
    def test_canonical_equality_is_excluded_and_unknown_is_retained(self):
        version='GBDT_V6.1.1'
        quality={'versions':{version:{'eligible_native_region_outcomes':{'table':{'TEDS':{}}}}}}
        regions=[{'version':version,'page_id':'p','native_region_index':i,'kind':'table','applied':True,
                  'outcomes':{'TEDS':label}} for i,label in enumerate(('improved','unmatchable','improved'))]
        pages=[{'page_id':'p','regions':[{'family':'gbdt','region_id':i,'before':{'canonical_sha256':'a'},
                 'after':{'canonical_sha256':'a' if i==2 else 'b'}} for i in range(3)]}]
        activation={'versions':{version:{'by_kind':{'table':{'funnel':{'applied_canonical_edit':{'regions':2}}}}}}}
        edited_outcomes(quality,regions,pages,activation)
        result=quality['versions'][version]['applied_canonical_region_outcomes']['table']['TEDS']
        self.assertEqual(result['regions'],{'improved':1,'worsened':0,'tied':0,'unmatchable':1})
        self.assertEqual(result['region_denominator'],2)

    def test_combined_pages_are_unions_across_tasks(self):
        version='MLP_V6.2.3';kinds={'table':'TEDS','equation':'CDM'}
        quality={'versions':{version:{'eligible_native_region_outcomes':{k:{m:{}} for k,m in kinds.items()}}}}
        regions=[{'version':version,'page_id':'same-page','native_region_index':i,'kind':kind,'applied':True,
                  'outcomes':{metric:'improved'}} for i,(kind,metric) in enumerate(kinds.items())]
        pages=[{'page_id':'same-page','regions':[{'family':'mlp','region_id':i,'before':{'canonical_sha256':'a'},
                 'after':{'canonical_sha256':'b'}} for i in range(2)]}]
        activation={'versions':{version:{'by_kind':{k:{'funnel':{'applied_canonical_edit':{'regions':1}}} for k in kinds}}}}
        edited_outcomes(quality,regions,pages,activation)
        result=quality['versions'][version]['applied_canonical_combined_primary_outcomes']
        self.assertEqual(result['regions']['improved'],2)
        self.assertEqual(result['unique_pages_by_outcome']['improved'],1)


if __name__=='__main__':unittest.main()
