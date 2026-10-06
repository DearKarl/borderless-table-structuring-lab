"""Exact accepted timeout classifier; no old run state imported."""
def normal_timeout(row,attempt,watch,guardian,failure_monotonic=None):
    """Classify the two watchdogs by bound deadline evidence, not exception name alone."""
    if row.get('status') not in ('failed','timeout'):return False
    if row.get('work_budget_seconds',0)<539.9:return False
    if watch.get('phase')!='page' or watch.get('item_id')!=row['item_id']:return False
    if abs(watch.get('hard_end',0)-watch.get('work_end',0)-60)>.01:return False
    if guardian.get('finished_monotonic',0)<watch.get('work_end',float('inf')):return False
    if row.get('work_elapsed_seconds',540)<539.9:return False
    if failure_monotonic is not None and failure_monotonic<watch['work_end']-.01:return False
    text=row.get('supervisor_error','')+' '+str(attempt.get('error',''))
    if 'InterruptedError' in text:
        return guardian.get('cause') in ('original_work_deadline','host_channel_eof_after_deadline') and 'Guardian deadline' in text
    if 'TimeoutError' in text:return True
    return failure_monotonic is not None and 'EOFError' in text and failure_monotonic>=watch['work_end']
