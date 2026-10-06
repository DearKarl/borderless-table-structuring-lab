"""One calibration allocation, durable send reservations, original boot/clocks."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import uuid
from hybrid.v4_input_selector.worker_contract import atomic_json
from .dependencies import need

LIMITS=dict(gpu_seconds=14400,wall_seconds=21600,cpu_seconds=1800,item_starts=16,page_work=540,page_cleanup=60)

def boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

class State:
    def __init__(self,root,manifest_sha,*,clock=time.monotonic,wall=time.time,boot_id=boot):
        self.root=Path(root);self.path=self.root/'STATE.json';self.sha=manifest_sha
        self.clock,self.wall,self.boot=clock,wall,boot_id

    @contextmanager
    def lock(self):
        import fcntl
        with (self.root/'state.lock').open('a+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)

    def snapshot(self):
        s=json.loads(self.path.read_bytes());need(s['manifest_sha256']==self.sha,'State manifest differs');return s

    def save(self,s,event):
        s['events'].append(dict(event=event,wall=self.wall(),monotonic=self.clock()))
        atomic_json(self.path,s)

    def create(self,items,cpu_seconds):
        need(not self.path.exists() and len(items)==16 and len({i['item_id'] for i in items})==16,'New exact sixteen-item state required')
        need(0<=cpu_seconds<1800,'CPU budget already exhausted')
        self.save(dict(schema='v4_calibration_state_v1',manifest_sha256=self.sha,boot_id=self.boot(),
            limits=LIMITS,submit=None,active=None,release=None,started_wall=None,started_monotonic=None,
            gpu_seconds=0,cpu_seconds=cpu_seconds,item_starts=0,stopped=False,collection='not_started',
            finalize=None,events=[],order=[i['item_id'] for i in items],
            slots={i['item_id']:dict(status='pending',reservation=None) for i in items}),'created')

    def require_allocation(self,s,allocation):
        need(s['boot_id']==self.boot(),'Host reboot; original allocation cannot continue')
        need(s['active'] is not None and s['active']['id']==allocation,'Active allocation differs')

    def claim_submit(self):
        with self.lock():
            s=self.snapshot()
            if s['submit'] is not None:return False,s
            need(s['boot_id']==self.boot() and s['active'] is None,'Unresolved state or changed boot')
            now=self.clock();wall=self.wall();allocation=uuid.uuid4().hex
            s.update(started_monotonic=now,started_wall=wall,collection='submitted',
                active=dict(id=allocation,started_monotonic=now,started_wall=wall,
                    global_end=now+min(LIMITS['gpu_seconds'],LIMITS['wall_seconds'])),
                submit=dict(status='intent_persisted',allocation=allocation,boot_id=self.boot()))
            self.save(s,'submit_intent_before_spawn');return True,s

    def submission(self,record):
        with self.lock():
            s=self.snapshot();need(s['submit'] is not None,'No submit intent')
            s['submit'].update(record);self.save(s,'spawn_observation')

    def reserve(self,allocation,item):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation)
            need(not s['stopped'] and s['item_starts']<16,'Further dispatch stopped')
            need(s['order'][s['item_starts']]==item and s['slots'][item]['status']=='pending','Item reordered or consumed')
            need(self.clock()+60<s['active']['global_end'] and self.wall()-s['started_wall']<21600-60,'Original budget exhausted')
            s['slots'][item].update(status='uncertain',reservation=dict(allocation=allocation,monotonic=self.clock(),wall=self.wall()))
            s['item_starts']+=1;self.save(s,'reserved_before_send:'+item)

    def result(self,allocation,item,result,output):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation);row=s['slots'][item]
            need(row['status']=='uncertain' and row['reservation']['allocation']==allocation,'Unreserved/repeated result')
            row.update(status=result['status'],result=result,output=str(output))
            if result['status']!='completed':s['stopped']=True
            self.save(s,'result:'+item)

    def failure(self,allocation,item,error):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation);s['stopped']=True
            s['last_error']=repr(error)
            if item is not None:s['slots'][item]['error']=repr(error)
            self.save(s,'failure_no_retry')

    def finish(self,allocation,release):
        with self.lock():
            s=self.snapshot();self.require_allocation(s,allocation)
            need(release.get('allocation')==allocation and release.get('container_absent') is True
                and release.get('lease_released') is True and not release.get('error'),'Positive owned release required')
            now=self.clock();elapsed=now-s['active']['started_monotonic']
            need(elapsed>=0 and s['active']['started_monotonic']<=release.get('finished',float('inf'))<=now,'Original release clock differs')
            s['gpu_seconds']+=elapsed;s['release']=release;s['active']=None
            if elapsed>14400 or self.wall()-s['started_wall']>21600:s['stopped']=True
            s['collection']='completed' if (not s['stopped'] and s['item_starts']==16 and
                all(r['status']=='completed' for r in s['slots'].values())) else 'failed_or_partial'
            self.save(s,'release_charged')

    def claim_finalize(self):
        with self.lock():
            s=self.snapshot();need(s['active'] is None and s['release'] is not None,'Release unresolved; no host scoring')
            need(s['boot_id']==self.boot(),'Reboot; original deadline cannot reset')
            if s['finalize'] is not None:return False,s
            remaining=min(1800-s['cpu_seconds'],s['started_monotonic']+21600-self.clock(),21600-(self.wall()-s['started_wall']))
            need(remaining>=1,'No remaining bounded CPU time')
            seconds=int(remaining)
            s['finalize']=dict(status='intent_persisted',seconds=seconds,boot_id=self.boot())
            # Reserve the entire possible CPU spend before spawn; never reissue it.
            s['cpu_seconds']+=seconds;self.save(s,'finalize_intent_before_spawn');return True,s

    def finalized(self,record):
        with self.lock():
            s=self.snapshot();need(s['finalize'] is not None,'No finalize intent')
            s['finalize'].update(record);self.save(s,'finalize_observation')
