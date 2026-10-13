"""Single pending Ovis mailbox; deadlines are fatal and never retried."""
import io
import hashlib
import signal
import time
from pathlib import Path
from .core import ContractError, DeadlineError, read, write
from .request_binding import rgb_binding

class ResidentClient:
    def __init__(self,control,out,ledger):
        self.control,self.out,self.ledger=control,Path(out),ledger;self.sequence=0;self.counts={}
    def text(self,image,binding):
        self.sequence+=1;p=binding['page_id'];self.counts[p]=self.counts.get(p,0)+1
        b=self.control['budget'];crop=self.control['runtime']['v32'].get('execution')=='crop'
        if self.counts[p]>min(256,b['expert_calls_per_page']) or self.sequence>(1 if crop else 512):raise ContractError('Ovis request budget exhausted')
        root=self.out/'experts/ipc';name=f'{self.sequence:08d}'
        stream=io.BytesIO();image.save(stream,format='PNG');data=stream.getvalue()
        with (root/(name+'.png')).open('xb') as f:f.write(data)
        v=self.control['runtime']['v32'];request={k:binding[k] for k in ('run_id','page_id','slot_id','input_sha256')}
        request.update(request_id=binding['request_id']+'/ovis',crop=rgb_binding(image),png_sha256=hashlib.sha256(data).hexdigest(),
            png_name=name+'.png',model_sha256=v['model_sha256'],config_sha256=v['config_sha256'],source_sha256=v['source_sha256'])
        if request['crop']!=binding['crop']:raise ContractError('Ovis actual prepared crop changed')
        self.ledger('CALL_START',engine='ovis',request_id=request['request_id'],page_id=p)
        pending=root/(name+'.pending');write(pending,request);pending.rename(root/(name+'.request.json'))
        remaining,_=signal.getitimer(signal.ITIMER_REAL)
        if remaining<=0:raise ContractError('Ovis outside bounded page/crop')
        end=min(time.monotonic()+min(120,remaining),self.control['absolute_stop_monotonic'])
        answer=root/(name+'.response.json')
        while not answer.exists():
            if (self.out/'experts/ERROR.json').exists():raise ContractError('Ovis resident system failure')
            if time.monotonic()>=end:raise DeadlineError('Ovis request timeout is fatal')
            time.sleep(.05)
        response=read(answer)
        self.ledger('CALL_RESULT',engine='ovis',request_id=request['request_id'],page_id=p,returned=True)
        return request,response
