from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.stats import rankdata
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'; F=ROOT/'figures';F.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(R/'budget_curves'/'budget_curve_raw.csv')
d=d[(d.method=='local')&(d.local_proposals_per_chain==262144)].copy()
d['bad']=d.mean_row_tv>.1
rows=[]
for col,label in [('score_split_rhat','score split-Rhat'),('chain_marginal_disagreement','marginal disagreement')]:
 x=d[col].replace([np.inf,-np.inf],np.nan);ok=x.notna();rho,p=spearmanr(x[ok],d.loc[ok,'mean_row_tv']); y=d.loc[ok,'bad'].astype(int).to_numpy(); scores=x[ok].to_numpy(); ranks=rankdata(scores); n1=y.sum(); n0=len(y)-n1; auc=float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0));rows.append(dict(diagnostic=label,n_instances=int(ok.sum()),spearman_rho=float(rho),spearman_p=float(p),auc_for_rowtv_gt_0_1=float(auc)))
s=pd.DataFrame(rows);s.to_csv(R/'diagnostic_summary.csv',index=False);d.to_csv(R/'diagnostic_raw.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.4,3.0))
axs[0].scatter(d.score_split_rhat,d.mean_row_tv,alpha=.75);axs[0].set_xlabel('score split-$\\widehat R$');axs[0].set_ylabel('exact row-TV');axs[0].set_xscale('log')
axs[1].scatter(d.chain_marginal_disagreement,d.mean_row_tv,alpha=.75);axs[1].set_xlabel('cross-chain marginal disagreement');axs[1].set_ylabel('exact row-TV')
for ax in axs: ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(F/'diagnostic_power.pdf',bbox_inches='tight');fig.savefig(F/'diagnostic_power.png',dpi=220,bbox_inches='tight');plt.close(fig)
(R/'diagnostic_metadata.json').write_text(json.dumps({'protocol':'60 independent-start local-MH runs, 20 at each n=8,9,10, maximum budget','bad_threshold':.1},indent=2))
print(s.to_string(index=False));print('bad',int(d.bad.sum()),'of',len(d))
