"""Describe quality outcomes for the applied canonical-edit activation stage."""
import collections


def edited_outcomes(quality,regions,pages,activation):
    """Describe the requested changed-output funnel without selecting on quality."""
    changed={(p['page_id'],r['family'],r['region_id']) for p in pages for r in p['regions']
             if r.get('after') and r['after']['canonical_sha256']!=r['before']['canonical_sha256']}
    for version,item in quality['versions'].items():
        family='gbdt' if version.startswith('GBDT') else 'mlp';counters={};page_sets={}
        combined=collections.Counter({k:0 for k in ('improved','worsened','tied','unmatchable')})
        union={k:set() for k in combined}
        for kind,metrics in item['eligible_native_region_outcomes'].items():
            counters[kind]={metric:collections.Counter({k:0 for k in combined}) for metric in metrics}
            page_sets[kind]={metric:{k:set() for k in combined} for metric in metrics}
        for row in regions:
            if row['version']!=version or not row['applied'] or (row['page_id'],family,row['native_region_index']) not in changed:continue
            kind=row['kind']
            for metric,label in row['outcomes'].items():
                counters[kind][metric][label]+=1;page_sets[kind][metric][label].add(row['page_id'])
            label=row['outcomes']['TEDS' if kind=='table' else 'CDM'];combined[label]+=1;union[label].add(row['page_id'])
        item['applied_canonical_region_outcomes']={kind:{metric:{'regions':dict(c),'unique_pages_by_outcome':{label:len(ids) for label,ids in page_sets[kind][metric].items()},
            'region_denominator':sum(c.values()),'page_denominator':1651} for metric,c in metrics.items()} for kind,metrics in counters.items()}
        for kind,metrics in counters.items():
            expected=activation['versions'][version]['by_kind'][kind]['funnel']['applied_canonical_edit']['regions']
            assert all(sum(c.values())==expected for c in metrics.values())
        item['applied_canonical_combined_primary_outcomes']={'regions':dict(combined),'unique_pages_by_outcome':{k:len(v) for k,v in union.items()},
            'region_denominator':sum(combined.values()),'page_denominator':1651,
            'definition':'Applied canonical output edits only; table TEDS/formula CDM labels. Diagnostic counts, not a composite score. Page sets can overlap.'}
    quality['edited_subset_note']='Descriptive requested funnel conditioned on applied canonical output changes, not quality; all-eligible and complete official outcomes are also retained.'
