"""Independent full-run ledger. Consumed pages can never return to pending."""
import uuid
from .host_state import State as BaseState
from .experimental import LIMITS
from .dependencies import need

class State(BaseState):
    def create(self,items):
        need(not self.path.exists() and len(items)==len({i['item_id'] for i in items})==1651,'New1651 state required')
        self.save(dict(schema='v4_experimental_state_v1',manifest_sha256=self.sha,boot_id=self.boot(),
            limits=LIMITS,submit=None,active=None,allocations=[],started_wall=None,started_monotonic=None,
            gpu_seconds=0,item_starts=0,stopped=False,terminal=False,terminal_reason=None,events=[],
            order=[i['item_id'] for i in items],slots={i['item_id']:dict(status='pending',reservation=None) for i in items}),'created')

    def claim_submit(self):
        with self.lock():
            s=self.snapshot()
            if s['submit'] is not None:return False,s
            need(s['boot_id']==self.boot() and s['active'] is None,'Changed boot/unresolved allocation')
            s.update(started_wall=self.wall(),started_monotonic=self.clock(),
                submit=dict(status='intent_persisted',boot_id=self.boot()))
            self.save(s,'controller_intent_before_spawn');return True,s

    def remaining(self,s):
        need(s['boot_id']==self.boot(),'Reboot; original run clocks cannot reset')
        elapsed=max(self.clock()-s['started_monotonic'],self.wall()-s['started_wall'])
        need(elapsed>=0,'Clock moved backwards')
        return min(LIMITS['gpu_seconds']-s['gpu_seconds'],LIMITS['wall_seconds']-elapsed,LIMITS['batch_seconds'])

    def begin(self):
        with self.lock():
            s=self.snapshot();need(not s['stopped'] and s['active'] is None,'Unresolved or stopped run; no allocation')
            need(s['submit'] is not None and all(a.get('reconciled') is True for a in s['allocations']),
                'All earlier allocations require positive release and reconciliation')
            pending=[i for i,k in enumerate(s['order']) if s['slots'][k]['status']=='pending']
            if not pending:
                s.update(terminal=True,terminal_reason='all_items_terminal');self.save(s,'inference_complete');return None
            seconds=int(self.remaining(s))
            if seconds<1200:
                for i in pending:s['slots'][s['order'][i]]['status']='not_run_budget_exhausted'
                s.update(terminal=True,terminal_reason='budget_exhausted');self.save(s,'budget_stop_without_refill');return None
            indices=pending[:min(64,(seconds-600)//600)];now=self.clock();a=dict(id=uuid.uuid4().hex,indices=indices,
                items=[s['order'][i] for i in indices],started_monotonic=now,started_wall=self.wall(),
                global_end=now+seconds,seconds=seconds,had_failure=False,release=None)
            s['active']=a;s['allocations'].append(dict(a));self.save(s,'allocation_before_prepare');return a

    def reserve(self,allocation,item):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation);a=s['active']
            need(not a['had_failure'] and not s['stopped'] and s['item_starts']<1651,'Dispatch stopped/cap reached')
            pending=[k for k in a['items'] if s['slots'][k]['status']=='pending']
            need(pending and pending[0]==item and s['slots'][item]['reservation'] is None,'Repeated/reordered dispatch')
            need(self.clock()+600<=a['global_end'],'Full original page envelope unavailable')
            s['slots'][item].update(status='uncertain',reservation=dict(allocation=allocation,wall=self.wall(),monotonic=self.clock()))
            s['item_starts']+=1;self.save(s,'reserved_before_send:'+item)

    def bind_batch(self,allocation,refs):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation)
            need('batch_refs' not in s['active'],'Batch already bound')
            s['active']['batch_refs']=refs;self.save(s,'batch_bytes_before_host_spawn')

    def result(self,allocation,item,result,output):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation);r=s['slots'][item]
            need(r['status']=='uncertain' and r['reservation']['allocation']==allocation,'Unreserved result')
            r.update(status=result['status'],result=result,output=str(output))
            if result['status']!='completed':s['active']['had_failure']=True
            self.save(s,'result:'+item)

    def failure(self,allocation,item,error):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation);s['active']['had_failure']=True
            s['active']['error']=repr(error)
            if item is not None:s['slots'][item].update(error=repr(error),failure_monotonic=self.clock())
            self.save(s,'failure_await_release_and_classification')

    def finish(self,allocation,release):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation);a=s['active'];now=self.clock()
            need(release.get('allocation')==allocation and release.get('container_absent') is True
                and release.get('lease_released') is True and not release.get('error'),'Positive owned release required')
            need(a['started_monotonic']<=release.get('finished',float('inf'))<=now,'Original release clock differs')
            elapsed=now-a['started_monotonic'];s['gpu_seconds']+=elapsed
            s['allocations'][-1]=dict(a,release=release,elapsed_seconds=elapsed,reconciled=False)
            s['active']=None;self.save(s,'release_and_charge')

    def reconcile(self,allocation,classified):
        with self.lock():
            s=self.snapshot();need(s['active'] is None and s['allocations'][-1]['id']==allocation,'Release before classification required')
            a=s['allocations'][-1];need(a['release'] and not a['reconciled'],'No release or repeated reconciliation')
            for k,kind in classified.items():
                r=s['slots'][k];need(k in a['items'] and r['reservation']['allocation']==allocation
                    and r['status'] in ('uncertain','completed','failed','timeout') and kind in ('failed','timeout'),'Invalid terminal failure classification')
                r['status']=kind;r['terminal_reason']='ordinary_model_failure' if kind=='failed' else 'original_540s_timeout'
            need(all(s['slots'][k]['status'] in ('pending','completed','failed','timeout') for k in a['items']),'Unknown started page; stop rather than retry')
            need(all(k in classified for k in a['items'] if s['slots'][k]['status'] in ('failed','timeout')),
                'Every noncompleted consumed item needs explicit classification')
            a['reconciled']=True;self.save(s,'batch_terminal_failures_classified')

    def stop_unknown(self,error):
        with self.lock():
            s=self.snapshot();s.update(stopped=True,terminal=False,terminal_reason='unknown_fault',last_error=repr(error))
            self.save(s,'unknown_fault_no_resubmit')
