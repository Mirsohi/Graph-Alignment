"""R15 M2 job queue (two workers share it through O_EXCL lock files).
Order: references for all 40 test instances (TEST order), then pipeline replicates (inst, 1), (inst, 2)
in TEST order, then comparators (inst, wl_ext), (inst, pt_broad_w) once both replicates exist.
Hard stops (pre-registered): no pipeline job starts after Sun 4 Oct 2026 12:00 PT, no comparator job
after Sun 4 Oct 2026 18:00 PT.
Usage: python r15_p1_queue.py <worker_id>"""
import os, sys, time, traceback, datetime
from r15_common import claim, append_jsonl
from r15_p1_common import P1, TEST
import r15_p1_reference as REF
import r15_p1_pipeline as PIPE
import r15_p1_comparators as COMP

PT = datetime.timezone(datetime.timedelta(hours=-7))
STOP_PIPE = datetime.datetime(2026, 10, 4, 12, 0, tzinfo=PT).timestamp()
STOP_COMP = datetime.datetime(2026, 10, 4, 18, 0, tzinfo=PT).timestamp()
LOCK = os.path.join(P1, 'locks')


def jobs():
    J = [('ref', i, None) for i in TEST]
    J += [('pipe', i, r) for i in TEST for r in (1, 2)]
    J += [('comp', i, m) for i in TEST for m in ('wl_ext', 'pt_broad_w')]
    return J


def out_path(kind, i, x):
    return {'ref': os.path.join(P1, 'ref', f'{i}.json'), 'pipe': os.path.join(P1, 'pipe', f'{i}_r{x}.json'),
            'comp': os.path.join(P1, 'comp', f'{i}_{x}.json')}[kind]


def main(wid):
    while True:
        pending = False
        did = False
        for kind, i, x in jobs():
            name = f'{kind}_{i}_{x}'
            if os.path.exists(out_path(kind, i, x)) or os.path.exists(os.path.join(LOCK, name + '.lock')):
                continue
            now = time.time()
            if kind == 'pipe' and now > STOP_PIPE:
                continue
            if kind == 'comp' and now > STOP_COMP:
                continue
            if kind == 'comp' and not all(os.path.exists(out_path('pipe', i, r)) for r in (1, 2)):
                pending = True
                continue
            if not claim(LOCK, name):
                continue
            t = time.time()
            try:
                if kind == 'ref':
                    REF.run(i)
                elif kind == 'pipe':
                    PIPE.run(i, x)
                else:
                    COMP.run(i, x)
                append_jsonl(dict(job=name, worker=wid, secs=time.time() - t, ok=True, end=time.ctime()),
                             os.path.join(P1, 'jobs_done.jsonl'))
            except Exception as e:
                traceback.print_exc()
                append_jsonl(dict(job=name, worker=wid, ok=False, err=repr(e), end=time.ctime()),
                             os.path.join(P1, 'jobs_done.jsonl'))
            did = True
            break           # restart scan from the top (keeps the pre-registered order)
        if not did:
            if not pending:
                print('queue empty', time.ctime(), flush=True)
                return
            time.sleep(30)


if __name__ == '__main__':
    main(sys.argv[1])
