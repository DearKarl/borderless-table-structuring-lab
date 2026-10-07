"""Container PID 1: enforce hard deadlines independently of the host monitor."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .io_utils import atomic_json, read_json, utc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--worker-id', required=True)
    args = ap.parse_args()
    control = read_json(args.control)
    folder = Path(args.output) / 'workers' / args.worker_id
    folder.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([sys.executable, '-B', '-m', 'versions.v5.worker',
                                '--control', args.control, '--output', args.output, '--worker-id', args.worker_id],
                               start_new_session=True)
    start = time.monotonic()
    reason = None
    try:
        while process.poll() is None:
            now = time.monotonic()
            if time.time() >= control['absolute_deadline_unix']:
                reason = 'absolute_round_deadline'
            status_path = folder / 'STATUS.json'
            if status_path.is_file():
                status = read_json(status_path)
                page_start = status.get('page_start_monotonic')
                if page_start is not None and now - page_start >= control['page_hard_seconds']:
                    reason = 'page_hard_deadline'
                if status['phase'] == 'model_loading' and now - start >= control['startup_hard_seconds']:
                    reason = 'model_startup_deadline'
            elif now - start >= control['startup_hard_seconds']:
                reason = 'worker_startup_deadline'
            if reason:
                atomic_json(folder / 'HARD_STOP.json', {'reason': reason, 'at': utc(), 'child_pid': process.pid})
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                return 124
            time.sleep(0.25)
        return process.returncode
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


if __name__ == '__main__':
    raise SystemExit(main())
