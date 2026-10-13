"""Analysis of protein and synthetic results: certificates, tables, figures.

Writes results/summary_protein.json, results/summary_synth.json,
paper/tables/*.tex and figures/*.pdf.
"""
import json, os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from gacert.core import crude_log_bound, tilted_log_bound

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LN01 = np.log(0.01)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6, 'lines.linewidth': 1.5,
                     'legend.frameon': False, 'pdf.fonttype': 42})
TAB = os.path.join(ROOT, 'paper', 'tables')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def beta_c(rec, witness=None):
    t = rec['ti']
    lams, L, m = np.array(t['lams']), np.array(t['Lambda']), np.array(t['mean_S'])
    sw = rec['best_graph_only_score'] if witness is None else witness
    for b in np.arange(0.5, 6.0001, 0.005):
        Lb = np.interp(b, lams, L) if b <= lams[-1] else L[-1] + (b - lams[-1]) * m[-1]
        if b * sw >= rec['log_nfact'] + Lb:
            return float(b)
    return float('nan')


def certs(rec, s_list, beta, witness):
    t = rec['ti']
    out = []
    for s in s_list:
        cr = crude_log_bound(rec['n'], beta, s, witness)
        tl, lam = tilted_log_bound(rec['n'], beta, s, witness, t['lams'], t['Lambda'])
        out.append(dict(s=s, crude=cr, tilted=tl, lam=lam,
                        crude_cert=cr < LN01, tilted_cert=tl < LN01,
                        margin=(LN01 - tl) if tl < LN01 else 0.0))
    return out


def enrich(rec):
    rec['beta_c'] = beta_c(rec)
    rec['beta_c_known'] = beta_c(rec, rec['known_score'])
    for g in rec['groups']:
        b = g['beta']
        for wname, w in (('graph', rec['best_graph_only_score']), ('known', rec['known_score'])):
            g['cert_chain_' + wname] = certs(rec, g['chain_max_retained'], b, w)
            g['cert_pooled_' + wname] = certs(rec, [max(g['chain_max_retained'])], b, w)[0]
        g['d_ratio'] = g['d_marg'] / max(g['floor_plugin'], 1e-12)
        g['std_pass'] = bool(g['rhat'] < 1.01 and g['d_ratio'] < 1.5)
    return rec


def med(x):
    return float(np.median(x))


def iqr(x):
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))


# ------------------------------------------------------------------ protein
def protein():
    recs = [enrich(json.load(open(f))) for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'protein', '*.json')))]
    betas = sorted({g['beta'] for r in recs for g in r['groups'] if g['label'].startswith('mh_')})
    S = dict(n_cases=len(recs))
    S['beta_c'] = {r['case']: r['beta_c'] for r in recs}
    bc = np.array(list(S['beta_c'].values()))
    S['beta_c_range'] = [float(bc.min()), float(bc.max()), float(np.median(bc))]
    S['graph_only_equals_known'] = int(sum(r['best_graph_only_score'] >= r['known_score'] for r in recs))
    S['graph_only_match'] = {r['case']: r['best_graph_only_match'] for r in recs}
    # sweep
    sweep = {}
    for lab in ('mh_uniform', 'mh_oracle'):
        for b in betas:
            gs = [g for r in recs for g in r['groups'] if g['label'] == lab and g['beta'] == b]
            chains = [c for g in gs for c in g['cert_chain_graph']]
            sweep[f'{lab}@{b}'] = dict(
                match_med=med([g['match'] for g in gs]), match_iqr=iqr([g['match'] for g in gs]),
                rhat_med=med([g['rhat'] for g in gs]), dratio_med=med([g['d_ratio'] for g in gs]),
                chains=len(chains), crude=int(sum(c['crude_cert'] for c in chains)),
                tilted=int(sum(c['tilted_cert'] for c in chains)),
                std_pass=int(sum(g['std_pass'] for g in gs)),
                silent=int(sum(g['std_pass'] and any(c['tilted_cert'] for c in g['cert_chain_graph']) for g in gs)),
                silent_crude=int(sum(g['std_pass'] and any(c['crude_cert'] for c in g['cert_chain_graph']) for g in gs)),
                min_margin=float(min([c['margin'] for c in chains if c['tilted_cert'] and not c['crude_cert']], default=np.nan)),
                med_margin=float(np.median([c['margin'] for c in chains if c['tilted_cert'] and not c['crude_cert']]) if any(c['tilted_cert'] and not c['crude_cert'] for c in chains) else np.nan))
    S['sweep'] = sweep
    # beta = 4 comparison
    comp = {}
    oracle = {r['case']: [g for g in r['groups'] if g['label'] == 'mh_oracle' and g['beta'] == 4.0][0] for r in recs}
    for lab in ('mh_uniform', 'annealed_mh', 'pt_quadratic8', 'pt_geometric16', 'mh_oracle'):
        gs = [(r, [g for g in r['groups'] if g['label'] == lab and g['beta'] == 4.0][0]) for r in recs]
        chains = [c for _, g in gs for c in g['cert_chain_graph']]
        chains_k = [c for _, g in gs for c in g['cert_chain_known']]
        comp[lab] = dict(match_med=med([g['match'] for _, g in gs]), match_iqr=iqr([g['match'] for _, g in gs]),
                         rhat_med=med([g['rhat'] for _, g in gs]), dratio_med=med([g['d_ratio'] for _, g in gs]),
                         chains=len(chains), crude=int(sum(c['crude_cert'] for c in chains)),
                         tilted=int(sum(c['tilted_cert'] for c in chains)),
                         crude_known=int(sum(c['crude_cert'] for c in chains_k)),
                         tilted_known=int(sum(c['tilted_cert'] for c in chains_k)),
                         cases_any_cert=int(sum(any(c['tilted_cert'] for c in g['cert_chain_graph']) for _, g in gs)),
                         cases_match_gt_08=int(sum(g['match'] > 0.8 for _, g in gs)),
                         work=int(gs[0][1].get('work_per_chain', 2 ** 21)),
                         min_swap=float(med([g['min_swap_rate'] for _, g in gs])) if 'min_swap_rate' in gs[0][1] else None)
    S['beta4'] = comp
    prune = {}
    for lab in ('annealed_mh', 'pt_quadratic8', 'pt_geometric16', 'mh_uniform'):
        keep, drop = [], []
        for r in recs:
            g = [g for g in r['groups'] if g['label'] == lab and g['beta'] == 4.0][0]
            for c, m in zip(g['cert_chain_graph'], g['match_chain']):
                (drop if c['tilted_cert'] else keep).append(m)
        prune[lab] = dict(n_keep=len(keep), n_drop=len(drop), match_keep_med=med(keep) if keep else None,
                          match_drop_med=med(drop) if drop else None,
                          keep_gt_08=int(sum(m > 0.8 for m in keep)), drop_gt_08=int(sum(m > 0.8 for m in drop)))
    S['prune_beta4'] = prune
    json.dump(S, open(os.path.join(ROOT, 'results', 'summary_protein.json'), 'w'), indent=1)

    # ---- per-protein table (appendix)
    import csv
    meta = {row['case_id']: row for row in csv.DictReader(open(os.path.join(ROOT, 'data', 'structures.csv')))}
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Case & Pair & $n$ & $|E_A|$ & $S(\pi^\star)$ & $\hat S_{\mathrm{graph}}$ & $\hat\beta_c$ \\', r'\midrule']
    for r in recs:
        m = meta.get(r['case'], {})
        pair = f"{m.get('source', '')}/{m.get('target', '')}".replace('_', ':')
        name = r['case'].replace('pscdb_', '').replace('_', '.').replace('adk.', 'AdK ').replace('crystal.control', 'control').replace('open.closed', 'open/closed')
        lines.append(f"{name} & {pair} & {r['n']} & {r['edges_a']} & {r['known_score']} & {r['best_graph_only_score']} & {r['beta_c']:.2f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(TAB, 'proteins.tex'), 'w').write('\n'.join(lines))

    # ---- beta=4 table (main)
    names = dict(mh_uniform='Local MH, random start', annealed_mh='Annealed MH', pt_quadratic8='PT, quadratic ladder (8)',
                 pt_geometric16='PT, geometric ladder (16)', mh_oracle='Local MH, oracle start$^\\dagger$')
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Sampler & Match & $\widehat R$ & $D/D_{\mathrm{iid}}$ & \multicolumn{2}{c}{Certified chains} \\',
             r' & (median) & (median) & (median) & crude & tilted \\', r'\midrule']
    for lab in names:
        c = comp[lab]
        lines.append(f"{names[lab]} & {c['match_med']:.2f} & {c['rhat_med']:.2f} & {c['dratio_med']:.1f} & {c['crude']}/{c['chains']} & {c['tilted']}/{c['chains']} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(TAB, 'beta4.tex'), 'w').write('\n'.join(lines))

    # ---- sweep table (appendix)
    lines = [r'\begin{tabular}{rrrrrrrr}', r'\toprule',
             r'$\beta$ & Match (rand.) & Match (oracle) & $\widehat R$ & $D/D_{\mathrm{iid}}$ & Std.\ pass & Cert.\ chains (crude/tilted) & Silent \\', r'\midrule']
    for b in betas:
        u, o = sweep[f'mh_uniform@{b}'], sweep[f'mh_oracle@{b}']
        lines.append(f"{b:g} & {u['match_med']:.2f} & {o['match_med']:.2f} & {u['rhat_med']:.2f} & {u['dratio_med']:.1f} & {u['std_pass']}/22 & {u['crude']}/{u['tilted']} of {u['chains']} & {u['silent']} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(TAB, 'sweep.tex'), 'w').write('\n'.join(lines))

    # ---- Figure: hysteresis + certificates
    fig, ax = plt.subplots(1, 2, figsize=(6.75, 2.2))
    xs = np.array(betas)
    for lab, col, mk, nm in (('mh_uniform', C[0], 'o', 'random start'), ('mh_oracle', C[1], 's', 'oracle start')):
        m = np.array([sweep[f'{lab}@{b}']['match_med'] for b in betas])
        lo = np.array([sweep[f'{lab}@{b}']['match_iqr'][0] for b in betas])
        hi = np.array([sweep[f'{lab}@{b}']['match_iqr'][1] for b in betas])
        ax[0].plot(xs, m, marker=mk, ms=4, color=col, label=nm)
        ax[0].fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0)
    ax[0].axvspan(bc.min(), bc.max(), color='0.85', lw=0, zorder=0)
    ax[0].axvline(np.median(bc), color='0.4', lw=0.8, ls='--')
    ax[0].set_xlabel(r'inverse temperature $\beta$')
    ax[0].set_ylabel('posterior mass on known match')
    ax[0].legend(loc='center right')
    ax[0].set_title(r'(a) hysteresis across $\hat\beta_c$ (22 pairs)', fontsize=8)
    cr = np.array([sweep[f'mh_uniform@{b}']['crude'] for b in betas]) / 88
    tl = np.array([sweep[f'mh_uniform@{b}']['tilted'] for b in betas]) / 88
    pass_ = np.array([sweep[f'mh_uniform@{b}']['std_pass'] for b in betas]) / 22
    ax[1].plot(xs, tl, marker='o', ms=4, color=C[0], label='tilted cert. (chains)')
    ax[1].plot(xs, cr, marker='^', ms=4, color=C[2], label='crude cert. (chains)')
    ax[1].plot(xs, pass_, marker='D', ms=3.5, color=C[3], label=r'$\widehat R$, $D$ pass (pairs)')
    ax[1].axvspan(bc.min(), bc.max(), color='0.85', lw=0, zorder=0)
    ax[1].set_xlabel(r'inverse temperature $\beta$')
    ax[1].set_ylabel('fraction')
    ax[1].set_ylim(-0.03, 1.03)
    ax[1].legend(loc='center left', bbox_to_anchor=(0.0, 0.45), fontsize=6.5)
    ax[1].set_title('(b) random-start local MH', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'protein_phase.pdf'))
    plt.close(fig)
    return S, recs


# ---------------------------------------------------------------- synthetic
def synth():
    files = sorted(glob.glob(os.path.join(ROOT, 'results', 'synth', '*.json')))
    if not files:
        return None
    recs = []
    for f in files:
        r = json.load(open(f))
        r['groups'] = [dict(g, label=k, beta=r['beta']) for k, g in r['groups'].items()]
        recs.append(enrich(r))
    rows = {}
    for r in recs:
        key = (r['family'], r['q'])
        rows.setdefault(key, []).append(r)
    out = []
    for (fam, q), rs in sorted(rows.items()):
        d = dict(family=fam, q=q, beta=rs[0]['beta'], beta_c=med([r['beta_c_known'] for r in rs]),
                 beta_c_graph=med([r['beta_c'] for r in rs]),
                 graph_witness_ratio=med([r['best_graph_only_score'] / r['known_score'] for r in rs]),
                 ti_melts=int(sum(r['ti']['oracle_mean_S_at_top'] < 2 * r['ti']['mean_S'][-1] for r in rs)))
        for lab in ('mh_uniform', 'mh_oracle', 'annealed_mh', 'pt_geometric16'):
            gs = [[g for g in r['groups'] if g['label'] == lab][0] for r in rs]
            ch = [c for g in gs for c in g['cert_chain_known']]
            chg = [c for g in gs for c in g['cert_chain_graph']]
            d[lab] = dict(tilted_graph=int(sum(c['tilted_cert'] for c in chg)), crude_graph=int(sum(c['crude_cert'] for c in chg)),match=med([g['match'] for g in gs]), tv_oracle=med([g['tv_to_oracle'] for g in gs]),
                          rhat=med([g['rhat'] for g in gs]), dratio=med([g['d_ratio'] for g in gs]),
                          tilted=int(sum(c['tilted_cert'] for c in ch)), crude=int(sum(c['crude_cert'] for c in ch)),
                          chains=len(ch), std_pass=int(sum(g['std_pass'] for g in gs)),
                          silent=int(sum(g['std_pass'] and any(c['tilted_cert'] for c in g['cert_chain_known']) for g in gs)))
        out.append(d)
    json.dump(out, open(os.path.join(ROOT, 'results', 'summary_synth.json'), 'w'), indent=1)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.75, 2.2), sharey=True)
    for k, fam in enumerate(('er', 'sbm')):
        ds = [d for d in out if d['family'] == fam]
        qs = [d['q'] for d in ds]
        for lab, col, mk, nm in (('mh_oracle', C[1], 's', 'oracle-start MH'), ('mh_uniform', C[0], 'o', 'random-start MH'),
                                 ('annealed_mh', C[2], '^', 'annealed MH'), ('pt_geometric16', C[3], 'D', 'PT (16)')):
            ax[k].plot(qs, [d[lab]['match'] for d in ds], marker=mk, ms=3.5, color=col, label=nm)
        ax[k].set_xscale('log')
        ax[k].set_xlabel('edge-flip probability $q$')
        ax[k].set_title(f"({'ab'[k]}) {'Erdős–Rényi' if fam == 'er' else 'two-block SBM'}, $n=120$", fontsize=8)
    ax[0].set_ylabel('posterior mass on planted match')
    ax[1].legend(fontsize=6.5, loc='center left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'synth.pdf'))
    plt.close(fig)
    lines = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'Family & $q$ & $\beta(q)$ & $\hat\beta_c$ & Match: rand./anneal/PT/oracle & Cert.\ rand.\ (tilted) & Cert.\ anneal & Silent \\', r'\midrule']
    for d in out:
        lines.append(f"{d['family'].upper()} & {d['q']:g} & {d['beta']:.2f} & {d['beta_c']:.2f} & "
                     f"{d['mh_uniform']['match']:.2f}/{d['annealed_mh']['match']:.2f}/{d['pt_geometric16']['match']:.2f}/{d['mh_oracle']['match']:.2f} & "
                     f"{d['mh_uniform']['tilted']}/{d['mh_uniform']['chains']} & {d['annealed_mh']['tilted']}/{d['annealed_mh']['chains']} & {d['mh_uniform']['silent']} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(TAB, 'synth.tex'), 'w').write('\n'.join(lines))
    return out


if __name__ == '__main__':
    S, _ = protein()
    print(json.dumps({k: S[k] for k in ('beta_c_range', 'graph_only_equals_known')}, indent=0))
    for k, v in S['sweep'].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'match_iqr'})
    for k, v in S['beta4'].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'match_iqr'})
    out = synth()
    if out:
        for d in out:
            print(d['family'], d['q'], round(d['beta'], 2), round(d['beta_c'], 2), 'melt', d['ti_melts'],
                  {k: (round(d[k]['match'], 2), round(d[k]['tv_oracle'], 2), d[k]['tilted'], d[k]['std_pass'], d[k]['silent'])
                   for k in ('mh_uniform', 'annealed_mh', 'pt_geometric16', 'mh_oracle')})
