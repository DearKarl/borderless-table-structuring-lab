"""A single serial Paddle worker in its frozen environment, with bounded lifetime."""
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from .region_protocol import RunBlocked, ExpertTimeout


class ExpertProcess:
    def __init__(self, python, model_dir, output, root, env=None, load_timeout=600, region_timeout=180, smoke_controls=False, manifest=None):
        self.output = Path(output); self.output.mkdir(exist_ok=False)
        self.region_timeout, self.requests, self.closed = region_timeout, 0, False
        self.messages = queue.Queue(maxsize=2)
        self.log = (self.output / 'console.log').open('xb')
        runtime_env = dict(os.environ if env is None else env, PYTHONPATH=str(root),
                           PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [str(python), '-B', '-m', 'hybrid_v2_tele_base.paddle_formula_worker',
                   '--model-dir', str(model_dir), '--audit-dir', str(self.output / 'native')]
        if smoke_controls: command.append('--smoke-controls')
        if manifest: command += ['--manifest',str(manifest)]
        self.process = subprocess.Popen(command, cwd=root, env=runtime_env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.log, text=True, encoding='utf-8', bufsize=1)
        (self.output / 'PROCESS.json').write_text(json.dumps({'pid': self.process.pid, 'command': command,
                  'started_at': time.time(), 'automatic_restart': False}), encoding='utf-8')
        def read():
            try:
                while True:
                    line = self.process.stdout.readline(8 * 1024 * 1024 + 1)
                    if not line: self.messages.put(None); return
                    if len(line) > 8 * 1024 * 1024: raise RunBlocked('Expert response exceeds bound')
                    self.messages.put(json.loads(line))
            except Exception as exc:
                try: self.messages.put_nowait(exc)
                except queue.Full: pass
        self.reader = threading.Thread(target=read, daemon=True); self.reader.start()
        try:
            self.load = self.receive(load_timeout)
            if self.load.get('event') != 'ready' or self.load.get('model_loads') != 1:
                raise RunBlocked('Expert did not initialize exactly once')
        except BaseException:
            self.close(force=True); raise

    def receive(self, timeout):
        result = self.messages.get(timeout=timeout)
        if result is None: raise RunBlocked('Expert process exited unexpectedly')
        if isinstance(result, Exception): raise RunBlocked('Expert protocol error: ' + str(result))
        return result

    def recognize(self, request):
        if self.closed: raise RunBlocked('Expert worker is unavailable; explicit new session required')
        self.requests += 1
        prefix = self.output / ('region-%06d' % self.requests)
        prefix.with_suffix('.png').write_bytes(base64.b64decode(request['png_base64'], validate=True))
        prefix.with_suffix('.request.json').write_text(json.dumps({k:v for k,v in request.items() if k != 'png_base64'},
                                                                  ensure_ascii=False, indent=2), encoding='utf-8')
        started = time.monotonic()
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n'); self.process.stdin.flush()
            response = self.receive(self.region_timeout - (time.monotonic()-started))
        except queue.Empty:
            self.close(force=True)
            prefix.with_suffix('.timeout.json').write_text(json.dumps({'region_id': request['region_id'],
                  'deadline_seconds': self.region_timeout, 'worker_exit_code': self.process.returncode,
                  'worker_confirmed_stopped': self.process.poll() is not None}), encoding='utf-8')
            raise ExpertTimeout('Region deadline; owned worker stopped, no retry')
        except (BrokenPipeError, OSError) as exc:
            self.close(force=True); raise RunBlocked('Expert transport failed') from exc
        prefix.with_suffix('.response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding='utf-8')
        return response

    def close(self, force=False):
        if self.closed: return
        self.closed = True
        try:
            if self.process.poll() is None:
                if not force:
                    try:
                        self.process.stdin.write('{"op":"close"}\n'); self.process.stdin.flush()
                        self.process.wait(timeout=3)
                    except (OSError, subprocess.TimeoutExpired): pass
                if self.process.poll() is None:
                    self.process.terminate()
                    try: self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill(); self.process.wait(timeout=5)
            (self.output / 'PROCESS_EXIT.json').write_text(json.dumps({'pid': self.process.pid,
                 'exit_code': self.process.returncode, 'forced': force, 'time': time.time()}), encoding='utf-8')
        finally:
            self.log.close()
            if self.process.stdin: self.process.stdin.close()
            if self.process.stdout: self.process.stdout.close()
