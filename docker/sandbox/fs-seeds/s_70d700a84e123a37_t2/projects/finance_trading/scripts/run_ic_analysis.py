"""Compute IC/IR per factor over full and recent windows, plus walk-forward.
Reads config.yaml + data/factor_panel.csv, prints/returns per-factor stats.
"""
import yaml, numpy as np, pandas as pd

def load_cfg(p='config.yaml'):
    with open(p) as f: return yaml.safe_load(f)

def rank_ic(df, fac, ret):
    out=[]
    for d,g in df.groupby('date'):
        if g[fac].notna().sum()>3:
            out.append(g[fac].corr(g[ret], method='spearman'))
    return np.array(out, dtype=float)

def analyze(cfg):
    df=pd.read_csv(cfg['data']['price_csv'])
    ret=cfg['data']['return_col']
    reg=yaml.safe_load(open(cfg['output']['registry']))['factors']
    rw=cfg['ic_analysis']['recent_window']
    res={}
    for fac in reg:
        if fac not in df.columns: continue
        ics=rank_ic(df, fac, ret)
        hist=np.nanmean(ics)
        recent=np.nanmean(ics[-rw:]) if len(ics)>rw else hist
        ir=hist/(np.nanstd(ics)+1e-9)
        res[fac]={'ic_hist':hist,'ic_recent':recent,'ir':ir}
    return res

if __name__=='__main__':
    print(analyze(load_cfg()))
