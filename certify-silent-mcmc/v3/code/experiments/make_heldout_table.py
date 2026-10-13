"""Per-rule table of the held-out validity test (results/heldout -> paper/tables/heldout.tex)."""
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(os.path.dirname(HERE))
R = os.path.join(V2, 'results', 'heldout')
T = os.path.join(V2, 'paper', 'tables')
yes = lambda b: '$\\times$' if b else ''
rows, cnt = [], dict(edge=0, meltD=0, meltC=0, profile=0, mid=0, n=0)
single_flagged = 0
for f in sorted(glob.glob(os.path.join(R, '*.json')), key=lambda f: (f.split('_')[-3], int(f.split('_s')[-1][:-5]))):
    d = json.load(open(f))
    if 'flagged' not in d:
        continue
    if d['kind'] == 'single':
        single_flagged += int(d['flagged'])
        continue
    r = dict(edge=d['edge'] >= 0.01, meltD=not d['meltD_ok'], meltC=not d['meltC_ok'], profile=not d['profile_ok'],
             mid=d['middle_basin'])
    for k in r:
        cnt[k] += int(r[k])
    cnt['n'] += 1
    rows.append(f"s{d['seed']} & {d['witness_agree']:.2f} & {d['w_window']:.2f} & {yes(r['edge'])} & {yes(r['meltD'])} & "
                f"{yes(r['meltC'])} & {yes(r['profile'])} & {yes(r['mid'])} \\\\")
lines = ['\\begin{tabular}{lccccccc}', '\\toprule',
         'seed & agreement & $\\mu_{\\rm win}(C)$ & edge & $D$ two-start & $C$ two-start & window & middle basin \\\\',
         '\\midrule'] + rows + ['\\midrule',
         f"fired & & & {cnt['edge']}/{cnt['n']} & {cnt['meltD']}/{cnt['n']} & {cnt['meltC']}/{cnt['n']} & {cnt['profile']}/{cnt['n']} & {cnt['mid']}/{cnt['n']} \\\\",
         '\\bottomrule', '\\end{tabular}']
open(os.path.join(T, 'heldout.tex'), 'w').write('\n'.join(lines) + '\n')
print(cnt, 'single flagged', single_flagged)
