"""Container PID 1: local socket bridge to the unchanged persistent OwnedWorker.

The explicit lifecycle-probe mode only echoes two protocol requests; it neither
imports tele_worker nor constructs OwnedWorker, queries a GPU, or reads assets.
"""
import argparse
import os
from pathlib import Path
import signal
import socket
import time
from types import SimpleNamespace

from .provider_wire import reply, validate_request
from .supervisor import OwnedWorker, work_alarm
from .worker_contract import read_contract, recv_message


class Bridge:
    def __init__(self, roots, global_end, *, probe=False, worker_factory=OwnedWorker, alarm=work_alarm):
        self.roots, self.global_end = roots, global_end
        self.probe, self.worker_factory, self.alarm = probe, worker_factory, alarm
        self.worker = None
        self.started = False
        self.stopped = False
        self.pages = set()
        self.work_end = min(time.monotonic() + 540, global_end - 60)

    def within(self, path, key):
        path, root = Path(path).resolve(), Path(self.roots[key]).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError('Bridge path escapes ' + key)
        return path

    def dispatch(self, operation, args, deadline):
        if operation in ('start', 'page'):
            if deadline > self.global_end - 60 or deadline > time.monotonic() + 540:
                raise ValueError('Bridge work deadline exceeds original budget')
            self.work_end = deadline
        if operation == 'start':
            if self.started or self.stopped:
                raise ValueError('Bridge can initialize only once')
            self.started = True
            if self.probe:
                if args != {'lifecycle_probe': True}:
                    raise ValueError('Probe cannot accept a model contract')
                return dict(pid=os.getpid(), model_loads=0, probe=True)
            if set(args) != {'contract', 'contract_sha256', 'owner', 'logs', 'output_root'}:
                raise ValueError('Unexpected bridge startup fields')
            with self.alarm(deadline):
                contract = self.within(args['contract'], 'control')
                output = self.within(args['output_root'], 'output')
                logs = self.within(args['logs'], 'output')
                read_contract(contract, args['contract_sha256'])
                native_args = SimpleNamespace(contract=str(contract), vendor_root=self.roots['vendor'],
                    model_root=self.roots['model'], aux_root=self.roots['aux'], input_root=self.roots['input'])
                self.worker = self.worker_factory(native_args, output, logs, args['contract_sha256'], args['owner'])
                return self.worker.start(deadline)
        if operation == 'page':
            if not self.started or self.stopped or set(args) != {'item_id'} or args['item_id'] in self.pages:
                raise ValueError('Invalid/repeated bridge page')
            self.pages.add(args['item_id'])
            if self.probe:
                if args['item_id'] not in ('echo_1', 'echo_2'):
                    raise ValueError('Unexpected lifecycle-probe item')
                return dict(item_id=args['item_id'], pid=os.getpid(), ordinal=len(self.pages), model_loads=0, probe=True)
            with self.alarm(deadline):
                return self.worker.page(args['item_id'], deadline)
        if operation == 'stop':
            if args or self.stopped or deadline > self.work_end + 25 or deadline > time.monotonic() + 25:
                raise ValueError('Bridge stop budget or state differs')
            self.stopped = True
            if self.worker is None:
                return dict(group_absent=True, cleanup_error=None, signals=[], probe=self.probe)
            with self.alarm(deadline):
                return self.worker.stop(deadline)
        raise ValueError('Unknown bridge operation')

    def abandoned(self):
        if self.worker is not None and not self.stopped:
            self.stopped = True
            deadline = min(time.monotonic() + 25, self.work_end + 25, self.global_end - 35)
            with self.alarm(deadline):
                return self.worker.stop(deadline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global-end', required=True, type=float)
    parser.add_argument('--initial-work-end', required=True, type=float)
    parser.add_argument('--lifecycle-probe', action='store_true')
    for name in ('control', 'output', 'input', 'vendor', 'model', 'aux'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if os.getpid() != 1:
        raise RuntimeError('Provider bridge must be the inspected container PID 1')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Container terminated')))
    bridge = Bridge({k: getattr(args, k) for k in ('control','output','input','vendor','model','aux')}, args.global_end, probe=args.lifecycle_probe)
    bridge.work_end = args.initial_work_end
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    previous = 0
    try:
        sock.settimeout(max(.001, args.initial_work_end - time.monotonic()))
        sock.connect('/v4-ipc/control.sock')
        while not bridge.stopped:
            # Idle bridge cannot outlive its current work plus inner cleanup tail.
            try:
                request = recv_message(sock, deadline=min(bridge.work_end + 25, args.global_end - 35))
            except EOFError:
                # The CPU host-loss fixture closes this connection after both
                # echoes. Keep native/early EOF and every other failure visible;
                # guardian still proves owned absence before releasing admission.
                if bridge.probe and bridge.started and bridge.pages == {'echo_1', 'echo_2'}:
                    break
                raise
            previous = validate_request(request, previous)
            try:
                result = bridge.dispatch(request['operation'], request['arguments'], request['deadline'])
                reply(sock, request, result=result)
            except BaseException as exc:
                reply(sock, request, error=repr(exc))
    finally:
        try:
            bridge.abandoned()
        finally:
            sock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
