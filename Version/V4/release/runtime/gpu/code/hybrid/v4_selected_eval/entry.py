"""Socket entry for the existing root-owned native lease/supervisor.

This entry never acquires a GPU or invents a lease. Root binds the accepted
native contract and its external supervisor. A separate input-only projection
removes the old worker contract's fixed action limitation for selected runs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import time
from hybrid.v4_input_selector.tele_worker import load_native
from hybrid.v4_input_selector.worker_contract import read_contract, recv_message, send_message
from .contracts import (checked_json, verify_code, prepare_adaptive_model,
                        prepare_frozen_model, policy_from_memory)
from .worker import CalibrationWorker, SelectedWorker
from .dependencies import install_path, require_receipt, native_origins

def initialize_worker(a, contract, items, rounding, prepared, experiment=None):
    native=load_native(contract,a.vendor_root,a.model_root,a.aux_root,a.output_root)
    native.runtime_binding.assert_clean()
    # All bundle/certificate reads preceded guard installation. This is the sole
    # estimator construction, from the retained verified bytes, under the guard.
    policy=policy_from_memory(prepared)
    native.selector_rounding_evidence=rounding
    native.selector_rounding_evidence_sha256=a.rounding_evidence_sha
    if a.mode=='selected':worker=SelectedWorker(native,policy=policy)
    elif a.mode=='experimental':
        from .experimental import ExperimentalWorker
        worker=ExperimentalWorker(native,policy=policy,experiment=experiment)
    else:worker=CalibrationWorker(native,items,policy=policy,rounding_evidence_sha256=a.rounding_evidence_sha)
    native.runtime_binding.assert_clean()
    return native,worker

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('selected','calibration','experimental'),required=True)
    for k in ('contract','contract-sha','projection','projection-sha','model','code-root',
              'vendor-root','model-root','aux-root','input-root','output-root','owner-token',
              'overlay','overlay-manifest','overlay-sha','dependency-receipt','dependency-receipt-sha','fixture-sha'):
        p.add_argument('--'+k,required=True)
    for k in ('manifest-sha','certificate','certificate-sha','plan','plan-sha','rounding-evidence','rounding-evidence-sha','experiment-auth','experiment-auth-sha'):
        p.add_argument('--'+k)
    p.add_argument('--socket-fd',type=int,required=True)
    a=p.parse_args();sock=socket.socket(fileno=a.socket_fd)
    try:
        dependency=require_receipt(checked_json(a.dependency_receipt,a.dependency_receipt_sha),
            overlay_sha=a.overlay_sha,fixture_sha=a.fixture_sha)
        install_path(a.overlay,a.overlay_manifest,a.overlay_sha)
        if native_origins()!=dependency['native_origins']:raise ValueError('Native dependency origins/bytes differ from CPU probe')
        contract=read_contract(a.contract,a.contract_sha)
        projection=checked_json(a.projection,a.projection_sha)
        rounding=None
        if a.rounding_evidence or a.rounding_evidence_sha:
            if not a.rounding_evidence or not a.rounding_evidence_sha:raise ValueError('Both rounding evidence bindings required')
            rounding=checked_json(a.rounding_evidence,a.rounding_evidence_sha)
        experiment=None
        if a.mode=='experimental':
            from .experimental import prepare_experimental,validate_batch
            if not a.experiment_auth or not a.experiment_auth_sha or a.certificate or a.certificate_sha or a.manifest_sha:
                raise ValueError('Separate experimental authorization required')
            prepared,experiment,auth=prepare_experimental(a.model,a.experiment_auth,a.experiment_auth_sha,code_root=a.code_root)
            validate_batch(projection,auth,a.experiment_auth_sha)
            if auth['rounding_evidence_sha256']!=a.rounding_evidence_sha:raise ValueError('Experimental rounding differs')
        elif a.mode=='selected':
            if a.experiment_auth or a.experiment_auth_sha:raise ValueError('Experimental authorization cannot replace production gate')
            if projection.get('schema')!='v4_policy_selected_items_v1' or projection.get('mode')!='policy_selected':
                raise ValueError('Single-version selected projection required')
            if not all((a.manifest_sha,a.certificate,a.certificate_sha)):
                raise ValueError('Adaptive evidence required before native startup')
            prepared=prepare_adaptive_model(a.model,a.manifest_sha,a.certificate,a.certificate_sha,code_root=a.code_root)
            if json.loads(prepared.deployment_json)['rounding_evidence_sha256']!=a.rounding_evidence_sha:
                raise ValueError('Rounding evidence not calibrated')
            if not 1<=len(projection['items'])<=1651:raise ValueError('Selected item cap exceeded')
        else:
            if not a.plan or not a.plan_sha:raise ValueError('Calibration plan binding required')
            plan=checked_json(a.plan,a.plan_sha)
            if projection.get('schema')!='v4_raster_roster_v1' or projection.get('plan_sha256')!=a.plan_sha:
                raise ValueError('Raster roster binding differs')
            if plan['known_overlap_reviewed'] is not True:raise ValueError('Known overlap review incomplete')
            if plan.get('rounding_evidence_sha256')!=a.rounding_evidence_sha:raise ValueError('Rounding evidence not planned')
            verify_code(Path(a.code_root),plan['code_files']);prepared=prepare_frozen_model(a.model)
        items={i['item_id']:i for i in projection['items']}
        if len(items)!=len(projection['items']):raise ValueError('Duplicate item identity')
        base={i['item_id']:i for i in contract['items']}
        if set(base)!=set(items):raise ValueError('Native contract item projection differs')
        for key,item in items.items():
            if {k:v for k,v in item.items() if k!='action'}!={k:v for k,v in base[key].items() if k!='action'}:
                raise ValueError('Native contract source projection differs')
            if a.mode in ('selected','experimental') and item['action']!='policy_selected':raise ValueError('Fixed arm in selected task')
            if a.mode=='calibration' and item['action']!=base[key]['action']:raise ValueError('Calibration action differs')
        # No resource admission here: caller must already own accepted binding.
        started=time.monotonic()
        native,worker=initialize_worker(a,contract,list(items.values()),rounding,prepared,experiment)
        send_message(sock,dict(kind='ready',owner_token=a.owner_token,mode=a.mode,
            contract_sha256=a.contract_sha,pid=os.getpid(),pgid=os.getpgrp(),
            initialization_seconds=time.monotonic()-started,
            projection_sha256=a.projection_sha,runtime=native.identity))
        while True:
            msg=recv_message(sock)
            if msg=={'kind':'stop'}:return 0
            if (set(msg)!={'kind','item_id','deadline'} or msg['kind']!='page'
                    or msg['item_id'] not in items or type(msg['deadline']) not in (int,float)
                    or not math.isfinite(msg['deadline']) or msg['deadline']<=time.monotonic()):
                raise ValueError('Invalid original page deadline/request')
            result=worker.run(items[msg['item_id']],a.input_root,a.output_root,msg['deadline'])
            send_message(sock,result)
            if result['status']!='completed':return 1
    except BaseException as exc:
        try:send_message(sock,dict(kind='fatal',error=repr(exc)))
        except BaseException:pass
        return 1
    finally:sock.close()

if __name__=='__main__':raise SystemExit(main())
