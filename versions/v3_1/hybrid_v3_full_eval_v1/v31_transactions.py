"""Transactions at the original helper post-processing boundary."""
import copy
from .core import ContractError, canonical
from .v31_rules import PROFILE, PROFILE_SHA, proposal, rect, formula_failure, i0

class V31Page:
    def __init__(self, page, profile, mode, experts, special_tokens, table_valid):
        if mode not in ("off","pass-through","on"):raise ContractError("Unknown routing mode")
        if profile not in ("native","v31-layout","v31-formula","v31-both"):
            raise ContractError("Unsupported V31 profile")
        self.page,self.profile,self.mode=page,profile,mode
        self.experts,self.special_tokens,self.table_valid=experts,special_tokens,table_valid
        self.events=[];self.formula_calls=0
        self.layout_on=mode=="on" and profile in ("v31-layout","v31-both")
        self.formula_on=mode=="on" and profile in ("v31-formula","v31-both")
        if (self.layout_on or self.formula_on) and experts is None:
            raise ContractError("Enabled expert provider missing")

    def formula(self, block, crop, binding, render):
        raw=block.content
        event={"slot_id":binding["slot_id"],"native_request_id":binding["request_id"],
               "primary":raw,"candidate":None,"selected":raw,"rollback":None}
        self.events.append(event)
        if not self.formula_on or block.type!="equation":
            event["reason"]="formula_not_enabled_or_not_equation";return
        failure=formula_failure(raw,binding["termination"],self.special_tokens["tele"])
        event["native_predicate"]=failure
        if failure["state"]!="bad":event["reason"]="native_"+failure["state"];return
        if self.formula_calls>=PROFILE["max_formula_requests"]:
            event["reason"]="formula_budget_abstain";return
        self.formula_calls+=1
        # All protocol/identity/audit/worker failures propagate. No second native call.
        response=self.experts.formula(crop,binding,render)
        event["expert_response"]=response
        if response["status"]!="ok":
            event.update(reason="expert_content_rejected",rollback=raw);return
        body=i0(response["raw_text"]);event["candidate"]=body
        if body is None:
            event.update(reason="i0_rejected",rollback=raw);return
        check=formula_failure(body,response["termination"],self.special_tokens["formula"])
        event["expert_predicate"]=check
        if check["state"]!="valid":
            event.update(reason="expert_structure_rejected",rollback=raw);return
        block.content=body
        event.update(selected=body,reason="explicit_native_failure_replaced",
                     actual_content_source="paddle",quality_selected=False)

    def apply(self, image, blocks, bindings, crops, fresh, make_block):
        """Called only after the original native assignment, before its one final post."""
        from .request_binding import rgb_binding
        render=rgb_binding(image)
        originals=list(blocks);before=[copy.deepcopy(dict(b)) for b in blocks]
        ids=[self.page["page_id"]+":native:"+str(i) for i in range(len(blocks))]
        for i,b in enumerate(blocks):
            binding=bindings.get(ids[i])
            if binding is not None:
                if binding["stage"]!="completed" or binding["raw_content"]!=b.content:
                    raise ContractError("Native raw assignment is not bound")
                self.formula(b,crops[ids[i]],binding,render)
        if not self.layout_on:
            return blocks,[{"slot_id":s,"block":copy.deepcopy(dict(b))} for s,b in zip(ids,blocks)]
        width,height=image.size
        native=[{"slot_id":ids[i],"type":b.type,"angle":b.angle,
                 "bbox":[v*(width if j%2==0 else height) for j,v in enumerate(rect(b.bbox))]}
                for i,b in enumerate(originals)]
        response=self.experts.layout(image.copy(),self.page,render)
        candidates=response["candidates"]
        indices=[c["index"] for c in candidates]
        if len(indices)!=len(set(indices)):raise ContractError("Duplicate detector indices")
        additions=[];expanded=set();occupied=[];counts={"expand":0,"add":0}
        for c in sorted(candidates,key=lambda x:(-x["score"],x["index"])):
            p=proposal(c,native,occupied+[a["geometry"] for a in additions],width,height,self.page["file_sha256"])
            event={"proposal":p,"detector_candidate":c,"layout_request_id":response['request_id'],"draft":None,"primary":None,
                   "candidate":None,"selected":None,"rollback":None}
            self.events.append(event)
            if not p["accepted"]:continue
            kind=p["kind"]
            if counts[kind]>=2:
                event["reason"]="layout_budget_abstain";continue
            if kind=="expand" and p["slot_id"] in expanded:
                event["reason"]="already_expanded";continue
            # Bound attempts as well as successful commits; no retry hunting.
            counts[kind]+=1
            stable=p["slot_id"]
            if kind=="add":stable=self.page["page_id"]+":"+stable
            normalized=[v/(width if j%2==0 else height) for j,v in enumerate(p["bbox"])]
            draft=make_block(p["type"],normalized,0,None)
            event["slot_id"]=stable;event["draft"]=copy.deepcopy(dict(draft))
            target=ids.index(stable) if kind=="expand" else None
            event["primary"]=copy.deepcopy(dict(blocks[target])) if target is not None else None
            crop,binding=fresh(image,draft,stable,p["detector_index"])
            event["fresh_request_id"]=binding["request_id"]
            event["candidate"]=copy.deepcopy(dict(draft))
            if draft.type=="equation":self.formula(draft,crop,binding,render)
            valid=binding["termination"]["stop"]=="eos" and bool(draft.content and draft.content.strip())
            if draft.type=="equation":
                # If a bad native completion was replaced, use the expert completion gate.
                f=self.events[-1] if self.events[-1] is not event else {}
                valid=(f.get("actual_content_source")=="paddle" or valid)
                valid=valid and formula_failure(draft.content, 
                    (f.get("expert_response") or {}).get("termination",binding["termination"]),
                    self.special_tokens["formula"] if f.get("actual_content_source")=="paddle" else self.special_tokens["tele"])["state"]=="valid"
            elif draft.type=="table":
                valid=valid and self.table_valid(draft.content)
            if not valid:
                event.update(reason="fresh_content_rejected",rollback=copy.deepcopy(event["primary"]));continue
            event.update(reason="committed",selected=copy.deepcopy(dict(draft)))
            if target is not None:
                # Replace atomically only after successful fresh recognition.
                blocks[target]=draft;expanded.add(stable)
                occupied.append({**native[target],"bbox":p["bbox"]})
            else:
                additions.append({"block":draft,"slot_id":stable,"proposal":p,
                    "geometry":{"slot_id":stable,"bbox":p["bbox"],"type":draft.type,"angle":0}})
        final=[];records=[]
        for index,(stable,b) in enumerate(zip(ids,blocks)):
            final.append(b);records.append({"slot_id":stable,"block":copy.deepcopy(dict(b))})
            following=[a for a in additions if a["proposal"]["anchors"][0]==stable]
            for a in sorted(following,key=lambda a:(a["proposal"]["bbox"][1],a["proposal"]["bbox"][0],a["proposal"]["detector_index"])):
                final.append(a["block"]);records.append({"slot_id":a["slot_id"],"block":copy.deepcopy(dict(a["block"]))})
        for i,stable in enumerate(ids):
            changed=stable in expanded or any(e.get("slot_id")==stable and e.get("actual_content_source")=="paddle" for e in self.events)
            if not changed and canonical(before[i])!=canonical(dict(blocks[i])):
                raise ContractError("Non-target native slot changed")
        return final,records

def original_table_validator(module):
    """Require a complete rectangular native OTSL grid before the original converter."""
    def valid(raw):
        tokens,texts=module.otsl_extract_tokens_and_text(raw)
        if not tokens or tokens[-1]!="<nl>":return False
        rows=[];row=[]
        for token in tokens:
            if token=="<nl>":
                if not row:return False
                rows.append(row);row=[]
            else:row.append(token)
        if not rows or len({len(r) for r in rows})!=1:return False
        for y,row in enumerate(rows):
            for x,token in enumerate(row):
                if token=="<lcel>" and x==0:return False
                if token=="<ucel>" and y==0:return False
                if token=="<xcel>" and (x==0 or y==0):return False
        try:
            cells,parsed_rows=module.otsl_parse_texts(texts,tokens)
            if parsed_rows!=rows:return False
            coverage=[[0]*len(rows[0]) for _ in rows]
            for cell in cells:
                for y in range(cell.start_row_offset_idx,cell.end_row_offset_idx):
                    for x in range(cell.start_col_offset_idx,cell.end_col_offset_idx):
                        if not 0<=y<len(rows) or not 0<=x<len(rows[0]):return False
                        coverage[y][x]+=1
            if any(value!=1 for row in coverage for value in row):return False
            html=module.convert_otsl_to_html(raw)
        except (IndexError,ValueError):
            return False
        return bool(html.startswith("<table") and html.endswith("</table>") and "<td" in html)
    return valid
