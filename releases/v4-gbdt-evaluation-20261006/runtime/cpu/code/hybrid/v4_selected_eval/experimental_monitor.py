"""Frozen host watchdog with one durable deadline-signal observation."""
from hybrid.v4_input_selector.provider_host import *
from hybrid.v4_input_selector import provider_host as frozen

def monitor(guardian, channel, child, identity):
    previous, interrupted = 0, False
    cause, error = 'host_exit', None
    try:
        while True:
            end = guardian.hard_end if guardian.final is not None else guardian.work_end + (25 if interrupted else 0)
            try:
                request = recv_message(channel, deadline=min(end,guardian.global_end))
            except (TimeoutError,socket.timeout):
                if guardian.final is not None:
                    cause = 'host_persistence_deadline'
                    break
                if not interrupted:
                    atomic_json(guardian.output/'DEADLINE_SIGNAL.json',dict(identity=identity,work_end=guardian.work_end,hard_end=guardian.hard_end,signal_monotonic=time.monotonic()))
                    signal_child(child, identity, signal.SIGTERM)
                    interrupted = True
                    cause = 'original_work_deadline'
                    continue
                cause = 'original_inner_cleanup_deadline'
                break
            previous = validate_request(request, previous)
            try:
                require(not interrupted or request['operation'] == 'release', 'Watchdog interrupted work; cleanup only')
                result = dispatch(guardian,request['operation'],request['arguments'],request['deadline'])
                reply(channel,request,result=result)
            except BaseException as exc:
                reply(channel,request,error=repr(exc))
    except EOFError:
        cause = 'host_channel_eof_after_deadline' if interrupted else 'host_channel_eof'
    except BaseException as exc:
        error = repr(exc)
        cause = 'guardian_control_failure'
    finally:
        channel.close()
        if guardian.allocation is not None and guardian.final is None:
            # Missing host cannot defer cleanup to a new clock/watchdog run.
            guardian.release(guardian.allocation,min(guardian.hard_end-5,guardian.clock()+30))
        remaining = min(guardian.hard_end,guardian.global_end) - time.monotonic()
        if remaining > 0:
            try:
                # If wait succeeds, never signal the reaped PID. On timeout the
                # direct Popen child still reserves its identity for one signal.
                child.wait(timeout=min(.25,remaining))
            except subprocess.TimeoutExpired:
                try:
                    signal_child(child,identity,signal.SIGKILL)
                    child.wait(timeout=max(.001,min(1,min(guardian.hard_end,guardian.global_end)-time.monotonic())))
                except BaseException as exc:
                    error = error or repr(exc)
            except BaseException as exc:
                error = error or repr(exc)
        receipt = dict(schema='v4_guardian_exit_v1',cause=cause,error=error,host_returncode=child.returncode,
                       host_identity=identity,global_end=guardian.global_end,hard_end=guardian.hard_end,
                       finalization=guardian.final,finished_monotonic=time.monotonic())
        with guardian.alarm(min(guardian.hard_end,guardian.global_end)):
            atomic_json(guardian.output/'GUARDIAN_EXIT.json',receipt)
    return receipt

def spawn_host(guardian,argv):
    original=frozen.monitor
    frozen.monitor=monitor
    try:return frozen.spawn_host(guardian,argv)
    finally:frozen.monitor=original
