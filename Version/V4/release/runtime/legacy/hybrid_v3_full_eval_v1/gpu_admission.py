"""Persistent admission markers survive host exit; explicit recovery checks before clearing."""
import json
import os
from pathlib import Path
import subprocess
import time
from .core import ContractError, identifier, read, sha, write, Deadline
from .bounded_cleanup import inspect_absent

def block_path(directory,key):
    return Path(directory)/(identifier(key)+".gpu-block.json")

def assert_unblocked(directory,keys):
    blocked=[str(block_path(directory,k)) for k in keys if block_path(directory,k).exists()]
    if blocked:raise ContractError("GPU admission blocked by unresolved owned-resource record: "+", ".join(blocked))

def durable(path,data,exclusive=False):
    path=Path(path)
    if exclusive:
        with path.open("x",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    else:
        temp=path.with_suffix(".pending")
        with temp.open("w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    if os.name=="posix":
        fd=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

class Admission:
    """Called only while all relevant interprocess leases are held."""
    def __init__(self,directory,keys):
        self.directory=Path(directory);self.keys=list(keys);self.record=None
    def reserve(self,run_id,owners):
        if self.record is not None:raise ContractError("Admission already reserved")
        assert_unblocked(self.directory,self.keys)
        self.record={"schema":1,"run_id":run_id,"gpu_uuids":self.keys,"state":"pending_owned_cleanup",
                     "created_unix":time.time(),"resources":{name:{"owner":name,"cid":None,
                     "container_name":"hybrid-v3-"+name,"creation_attempted":False,"confirmed":False} for name in owners}}
        for key in self.keys:durable(block_path(self.directory,key),self.record,exclusive=True)
    def observe(self,owned):
        if self.record is None or owned.token not in self.record["resources"]:
            raise ContractError("Unregistered resource owner")
        self.record["resources"][owned.token].update(cid=owned.cid,
            creation_attempted=owned.creation_attempted,confirmed=owned.last_verified is not None)
        self._persist()
    def _persist(self):
        for key in self.keys:
            path=block_path(self.directory,key)
            if not path.is_file() or read(path)["run_id"]!=self.record["run_id"]:
                raise ContractError("Persistent admission owner changed or record missing")
            durable(path,self.record)
    def finish(self,group):
        if self.record is None:return {"reserved":False,"released":True,"blocked":False}
        if group.get("resource_release_verified") and not group.get("budget_exceeded"):
            for key in self.keys:
                path=block_path(self.directory,key)
                if path.exists():
                    if read(path)["run_id"]!=self.record["run_id"]:raise ContractError("Refuse foreign admission clear")
                    path.unlink()
            return {"reserved":True,"released":True,"blocked":False,"run_id":self.record["run_id"],
                    "physical_release_verified":True,"gpu_release":group["release"]}
        self.record.update(state="blocked_unresolved_cleanup",cleanup=group)
        self._persist()
        return {"reserved":True,"released":False,"blocked":True,"run_id":self.record["run_id"],
                "records":[str(block_path(self.directory,k)) for k in self.keys]}

def clear(directory,keys,run_id,idle_memory_mib,output,runner=subprocess.run):
    """Explicit operator command: require every recorded container absent AND fresh GPU idleness."""
    from .lifecycle import Leases,gpu_idle
    keys=sorted(keys);start=time.monotonic();clock=Deadline(start,60,0)
    # Recovery acquires the same leases but may inspect existing blocks; it never launches a model.
    with Leases(directory,keys,recovery=True):
        paths=[block_path(directory,k) for k in keys]
        records=[read(p) for p in paths]
        if not records or any(r["run_id"]!=run_id or sorted(r["gpu_uuids"])!=keys for r in records):
            raise ContractError("Recovery must name exact blocked run and complete GPU set")
        resources={}
        for record in records:
            for owner,item in record["resources"].items():
                prior=resources.get(owner)
                if prior is not None and prior!=item:raise ContractError("Persistent resource records disagree")
                resources[owner]=item
        proof=[]
        for owner,item in resources.items():
            if not item["creation_attempted"]:continue
            for target in dict.fromkeys([item["container_name"],item["cid"]] if item["cid"] else [item["container_name"]]):
                p=runner(["docker","inspect",target],capture_output=True,text=True,check=False,timeout=clock.bound(5))
                if not inspect_absent(p,target):
                    raise ContractError("Recorded resource is present or daemon state unknown; block retained")
                proof.append({"owner":owner,"target":target,"absence_verified":True})
        memory=gpu_idle(keys,idle_memory_mib,clock,runner=runner)
        result={"run_id":run_id,"gpu_uuids":keys,"blocks":{p.name:sha(p) for p in paths},
                "container_absence":proof,"fresh_gpu_memory":memory,"explicit_recovery":True,
                "verified_at_unix":time.time()}
        # Persist verification before unlinking; any evidence failure keeps admission blocked.
        write(output,result)
        for path in paths:path.unlink()
        return {**result,"admission_released":True}
