#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

METRICS=('AUROC','AUPRC','AUARC','AURC','ECE','MCE','Brier')


def build_scores(raw,score_type,calibration='minmax'):
    raw=np.asarray(raw,dtype=float)
    if score_type not in {'confidence','uncertainty'}:raise ValueError('Unknown score direction')
    rank=raw if score_type=='confidence' else -raw
    if calibration=='minmax':
        lo,hi=rank.min(),rank.max()
        conf=np.full_like(rank,.5) if np.isclose(lo,hi) else (rank-lo)/(hi-lo)
    elif calibration in {'probability','percent'}:
        unit=raw/(100 if calibration=='percent' else 1)
        if np.any(unit<0) or np.any(unit>1):raise ValueError('Probability/percent calibration scores outside declared bounds')
        conf=unit if score_type=='confidence' else 1-unit
    else:raise ValueError('Unknown calibration mode')
    return rank,conf


def calibration_errors(y,conf,bins=10,strategy='uniform'):
    edges=np.linspace(0,1,bins+1) if strategy=='uniform' else np.unique(np.quantile(conf,np.linspace(0,1,bins+1)))
    indices=np.digitize(conf,edges[1:-1],right=True)
    ece=mce=0.
    for b in np.unique(indices):
        selected=indices==b
        gap=abs(float(y[selected].mean()-conf[selected].mean()))
        ece+=selected.mean()*gap;mce=max(mce,gap)
    return float(ece),float(mce)


def coverage_areas(y,confidence,mode='legacy_trapezoid'):
    
    if mode=='legacy_trapezoid':
        order=np.argsort(-confidence)
        accuracy=np.cumsum(y[order])/np.arange(1,len(y)+1)
        coverage=np.arange(1,len(y)+1)/len(y)
        integrate=getattr(np,'trapezoid',None) or np.trapz
        return float(integrate(accuracy,coverage)),float(integrate(1-accuracy,coverage))
    if mode=='tie_expected_discrete':
        
        order=np.argsort(-confidence,kind='stable');scores=confidence[order];labels=y[order]
        boundaries=np.r_[0,np.flatnonzero(np.diff(scores)!=0)+1,len(y)]
        acc=[];previous_correct=0.
        for start,end in zip(boundaries[:-1],boundaries[1:]):
            size=end-start;correct=labels[start:end].sum()
            for k in range(1,size+1):acc.append((previous_correct+k*correct/size)/(start+k))
            previous_correct+=correct
        area=float(np.mean(acc))
        return area,1-area
    raise ValueError('Unknown coverage mode')


def point_metrics(y,rank,conf,bins=10,binning='uniform',coverage='legacy_trapezoid'):
    auarc,aurc=coverage_areas(y,rank,coverage)
    ece,mce=calibration_errors(y,conf,bins,binning)
    two_classes=len(np.unique(y))==2
    return {'AUROC':float(roc_auc_score(y,rank)) if two_classes else float('nan'),
            'AUPRC':float(average_precision_score(y,rank)) if np.any(y==1) else float('nan'),
            'AUARC':auarc,'AURC':aurc,'ECE':ece,'MCE':mce,'Brier':float(np.mean((conf-y)**2))}


def evaluate(y,raw,score_type,bootstrap_runs=200,seed=42,bins=10,binning='uniform',
             calibration='minmax',coverage='legacy_trapezoid'):
    y=np.asarray(y);raw=np.asarray(raw,dtype=float)
    if len(y)==0 or len(y)!=len(raw) or not np.isfinite(raw).all():raise ValueError('Empty, mismatched, or nonfinite input')
    if not np.isin(y,[0,1]).all():raise ValueError('Correctness must contain only 0/1')
    y=y.astype(int)
    if bootstrap_runs<0 or bins<1:raise ValueError('bootstrap-runs must be >=0 and bins >=1')
    rank,conf=build_scores(raw,score_type,calibration)
    point=point_metrics(y,rank,conf,bins,binning,coverage)
    values={m:[]for m in METRICS};rng=np.random.default_rng(seed)
    for _ in range(bootstrap_runs):
        indices=rng.choice(len(y),len(y),replace=True)
        
        result=point_metrics(y[indices],rank[indices],conf[indices],bins,binning,coverage)
        for metric,value in result.items():
            if np.isfinite(value):values[metric].append(value)
    rows=[]
    for metric in METRICS:
        valid=values[metric]
        if bootstrap_runs:
            mean=float(np.mean(valid)) if valid else None
            std=float(np.std(valid,ddof=0)) if valid else None
        else:
            mean=point[metric] if np.isfinite(point[metric]) else None
            std=0. if mean is not None else None
        scale=1 if metric=='Brier' else 100;digits=4 if metric=='Brier' else 2
        display=f'{mean*scale:.{digits}f} ± {std*scale:.{digits}f}' if mean is not None else 'NaN'
        rows.append({'metric':metric,'mean':mean,'std':std,'result':display,
                     'point_estimate':point[metric] if np.isfinite(point[metric]) else None,
                     'bootstrap_valid':len(valid) if bootstrap_runs else 0,
                     'bootstrap_requested':bootstrap_runs})
    return rows
