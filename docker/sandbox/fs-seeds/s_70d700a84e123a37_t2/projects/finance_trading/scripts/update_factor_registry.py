"""Update factor_registry.yaml from IC analysis results.
Flips direction on sign reversal, refreshes ic/ir, sets status decayed/watch,
and can rebalance weights toward walk-forward recommended weights.
"""
import yaml

def update(reg_path, results, wf_weights=None, decay_thr=0.5):
    reg=yaml.safe_load(open(reg_path))
    for fac,st in results.items():
        f=reg['factors'].get(fac)
        if not f: continue
        f['ic']=round(float(st['ic_hist']),4)
        f['ir']=round(float(st['ir']),3)
        if st['ic_hist']*f['direction']<0:
            f['direction']*=-1; f['status']='flipped'
        elif abs(st['ic_recent'])<abs(st['ic_hist'])*decay_thr:
            f['status']='decayed'
        if wf_weights and fac in wf_weights:
            f['weight']=round(float(wf_weights[fac]),3)
    with open(reg_path,'w') as f: yaml.safe_dump(reg,f,sort_keys=False)
    return reg
