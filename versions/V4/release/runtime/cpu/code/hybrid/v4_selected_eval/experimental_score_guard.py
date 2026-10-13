"""CPU scoring container PID 1; absolute wall deadline survives host/SSH loss."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from hybrid.v4_input_selector.supervisor import process_identity
from hybrid.v4_input_selector.worker_contract import atomic_json

def remaining(deadline,now):
    if deadline-now<=20:raise TimeoutError('Original CPU evaluation deadline exhausted')
    return deadline-now-15

def main():
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    if os.getpid()!=1:raise RuntimeError('Independent scoring guardian must be container PID 1')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    budget=remaining(args.deadline,time.time());child=subprocess.Popen(command,start_new_session=True)
    identity=process_identity(child.pid);reaped=False;error=None;code=None
    atomic_json('/work/SCORER_PROCESS.json',dict(identity=identity,absolute_deadline=args.deadline,command=command))
    try:
        code=child.wait(timeout=max(.001,min(budget,args.deadline-time.time()-15)));reaped=True
    except BaseException as exc:error=repr(exc)
    finally:
        if not reaped:
            if process_identity(child.pid)!=identity:raise RuntimeError('CPU scorer child identity changed')
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=max(.001,min(5,args.deadline-time.time()-5)));reaped=True
            except subprocess.TimeoutExpired:
                if process_identity(child.pid)!=identity:raise RuntimeError('CPU scorer child identity changed before kill')
                os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=max(.001,min(3,args.deadline-time.time()-1)));reaped=True
        atomic_json('/work/SCORER_TERMINAL.json',dict(returncode=code,error=error,finished_wall=time.time(),absolute_deadline=args.deadline,reaped=reaped))
    return 0 if code==0 and error is None else 1

if __name__=='__main__':raise SystemExit(main())
