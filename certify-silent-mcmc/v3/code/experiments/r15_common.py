"""Shared helpers for the R15 experiments (provenance, atomic JSON writes, a two-worker
file-lock job queue).  Every R15 result JSON carries `prov`: script SHA-256, gacert2 SHA-256s,
hostname, start/end time and wall-clock seconds."""
import hashlib, json, os, socket, sys, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(os.path.dirname(HERE))
CODE = os.path.join(V3, 'code')
RES = os.path.join(V3, 'results', 'r15')
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def code_hashes(extra=()):
    out = {}
    for p in sorted(glob.glob(os.path.join(CODE, 'gacert2', '*.py'))) + list(extra):
        out[os.path.relpath(p, V3)] = sha256(p)
    return out


def prov(script, t0, extra=()):
    return dict(script=os.path.relpath(os.path.abspath(script), V3), script_sha256=sha256(script),
                code_sha256=code_hashes(extra), hostname=socket.gethostname(),
                started=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(t0)),
                finished=time.strftime('%Y-%m-%d %H:%M:%S %Z'), wall_seconds=time.time() - t0,
                python=sys.version.split()[0], numpy=np.__version__)


class Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def dump(obj, path):
    """Atomic write; refuses to overwrite (R15 rule: no result file is overwritten)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(path)
    tmp = path + '.tmp%d' % os.getpid()
    with open(tmp, 'w') as f:
        json.dump(obj, f, cls=Enc)
    os.replace(tmp, path)


def append_jsonl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(obj, cls=Enc) + '\n')


def claim(lockdir, name):
    """Atomically claim a job (O_EXCL lock file). Returns True if this worker owns it."""
    os.makedirs(lockdir, exist_ok=True)
    try:
        fd = os.open(os.path.join(lockdir, name + '.lock'), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{socket.gethostname()} pid {os.getpid()} {time.ctime()}\n'.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def progress(msg):
    p = os.path.join(V3, 'notes', 'R15_PROGRESS.md')
    with open(p, 'a') as f:
        f.write(f"- {time.strftime('%a %d %b %H:%M %Z')} [agent3] {msg}\n")
