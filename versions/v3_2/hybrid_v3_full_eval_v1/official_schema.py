"""Validate the frozen official result schema without changing its metrics."""
import json
import math
from pathlib import Path

EXPECTED={"text_block":("Edit_dist",),"display_formula":("Edit_dist","CDM"),
          "table":("TEDS","TEDS_structure_only","Edit_dist"),"reading_order":("Edit_dist",)}

def mapping(value,name):
    if not isinstance(value,dict):raise ValueError("Invalid official mapping: "+name)
    return value

def validate_results(metrics,result_dir,n):
    mapping(metrics,"metric result")
    debug=mapping(metrics["match_debug"],"match_debug")
    if type(debug["page_count"]) is not int or debug["page_count"]!=n:
        raise ValueError("Official denominator differs from locked input N")
    root=Path(result_dir)
    report=json.loads((root/"pred_quick_match_run_summary.json").read_text())
    if report["save_name"]!="pred_quick_match":raise ValueError("Official report name differs")
    denoms=mapping(report["page_denominators"],"page_denominators")
    mapping(report["notebook_metric_summary"],"notebook_metric_summary")
    coverage={}; undefined={}; observed={}
    def number(value,reason,allow_undefined):
        if type(value) in (int,float) and math.isfinite(value):
            if not 0<=value<=1:raise ValueError("Official metric outside [0,1]: "+reason)
            return value
        if allow_undefined and (value=="NaN" or type(value) is float and math.isnan(value)):
            return None
        raise ValueError("Invalid/nonempty undefined official metric: "+reason)
    for category,names in EXPECTED.items():
        # Official pipeline writes these files even when samples are [].
        samples=json.loads((root/("pred_quick_match_"+category+"_result.json")).read_text())
        if not isinstance(samples,list) or any(not isinstance(s,dict) for s in samples):
            raise ValueError("Invalid official sample list: "+category)
        for sample in samples:
            if not isinstance(sample.get("img_id"),str):
                raise ValueError("Official sample img_id missing")
            if sample.get("metric") is not None and not isinstance(sample["metric"],dict):
                raise ValueError("Invalid official sample metric")
        payload=mapping(metrics[category],category)
        for layer in ("all","group","page"):mapping(payload[layer],category+"."+layer)
        coverage[category]={"matched_samples":len(samples),"page_denominators":{}}
        for name in names:
            denominator=mapping(denoms[category][name],category+" denominator")["ALL"]
            if type(denominator) is not int or not 0<=denominator<=n:
                raise ValueError("Invalid official page denominator")
            coverage[category]["page_denominators"][name]=denominator
            all_metric=mapping(payload["all"][name],category+".all."+name)
            all_key="ALL_page_avg" if name=="Edit_dist" else "all"
            zero_length=False
            if name=="Edit_dist" and samples:
                lengths=[s.get("upper_len") for s in samples]
                if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in lengths):
                    raise ValueError("Invalid official Edit_dist denominator evidence")
                zero_length=all(x==0 for x in lengths)
            allow_undefined=not samples or (name=="Edit_dist" and zero_length and denominator==0)
            all_value=number(all_metric[all_key],category+"."+name,allow_undefined)
            if not samples and all_value is not None:
                raise ValueError("Empty official samples cannot have a defined all-sample mean")
            if denominator==0 and any(type((s.get("metric") or {}).get(name)) in (int,float)
                and math.isfinite(s["metric"][name]) and s.get("img_id") for s in samples):
                raise ValueError("Official page denominator contradicts metric-bearing samples")
            # A missing page metric is legitimate only when the original page denominator is zero.
            page_metric=payload["page"].get(name)
            if page_metric is None:
                if denominator!=0 and not (name=="Edit_dist" and not samples):
                    raise ValueError("Missing metric with nonzero page denominator")
                page_value=None
            else:
                page_metric=mapping(page_metric,category+".page."+name)
                if "ALL" not in page_metric:
                    raise ValueError("Malformed official page metric")
                page_value=number(page_metric["ALL"],category+".page."+name,
                                  denominator==0 and allow_undefined)
            selected=all_value if name=="Edit_dist" else page_value
            key=category+"_"+name
            observed[key]=selected
            if selected is None:
                if name=="Edit_dist" and not allow_undefined:
                    raise ValueError("Unexplained undefined Edit_dist")
                undefined[key]={"reason":"empty samples" if not samples else "official zero length/page denominator",
                                "sample_count":len(samples),"official_page_denominator":denominator}
    return observed,coverage,undefined,report
