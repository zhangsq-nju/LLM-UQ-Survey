#!/usr/bin/env python3

from __future__ import annotations
import argparse
import collections
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile

if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metric_evaluation.fields import FIELDS,FALLBACKS,nested,normalize_method
from metric_evaluation.metrics import METRICS,evaluate


def scalar(value,reducer=None):
    if value is None:return None
    if isinstance(value,(list,dict)):
        if not isinstance(value,list) or not reducer or not value:return None
        parts=[scalar(x) for x in value]
        if any(x is None for x in parts):return None
        value=sum(parts)/(len(parts) if reducer=='mean' else 1)
    if isinstance(value,bool):return None
    try:result=float(value)
    except (ValueError,TypeError):return None
    return result if math.isfinite(result) else None


def labels_from_file(path):
    labels={}
    with Path(path).open() as stream:
        for number,line in enumerate(stream,1):
            if not line.strip():continue
            row=json.loads(line);qid=next((row[k] for k in ['id','question_id','task_id'] if row.get(k) is not None),None)
            correct=row.get('correct')
            if qid is None or correct not in (0,1) or correct is None:raise ValueError(f'Invalid correctness line {number}')
            key=(str(qid),row.get('response_index',row.get('target_response_index')))
            if key in labels:raise ValueError(f'Duplicate external correctness ID/index: {key}')
            labels[key]=int(correct)
    return labels


def load_scores(path,method=None,score_key=None,score_type=None,correct_key=None,correct_file=None,
                reducer=None,invalid_policy='skip'):
    if score_key and not score_type:raise ValueError('--score-key requires --score-type')
    if not score_key and method not in FIELDS:raise ValueError('Specify --method or --score-key and --score-type')
    candidates=[(score_key,score_type)] if score_key else list(FIELDS[method])
    if not score_key:
        candidates += [(p,candidates[0][1]) for p in FALLBACKS.get(method,[])]
        candidates += [('metrics.'+p,d) for p,d in list(candidates) if not p.startswith('metrics.')]
    external=labels_from_file(correct_file) if correct_file else None
    y=[];scores=[];selected=None;counts=collections.Counter();correct_paths=collections.Counter()
    def reject(reason,line):
        counts[reason]+=1
        if invalid_policy=='error':raise ValueError(f'{path}: line {line}: {reason}')
    with Path(path).open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            counts['lines']+=1
            if not line.strip():counts['blank']+=1;continue
            try:row=json.loads(line)
            except (json.JSONDecodeError,ValueError):reject('invalid_json',number);continue
            if not isinstance(row,dict):reject('not_an_object',number);continue
            if row.get('error'):reject('backend_error',number);continue
            if selected is None:
                selected=next(((p,d) for p,d in candidates if scalar(nested(row,p),reducer) is not None),None)
            if selected is None:reject('missing_or_nonfinite_or_nonscalar_score',number);continue
            score=scalar(nested(row,selected[0]),reducer)
            if score is None:reject('missing_or_nonfinite_or_nonscalar_score',number);continue
            if external is not None:
                qid=next((row[k] for k in ['id','question_id','task_id'] if row.get(k) is not None),None)
                index=row.get('target_response_index',row.get('response_index',0))
                correct=external.get((str(qid),index),external.get((str(qid),None)))
                label_path='external_correct_file'
            else:
                paths=[correct_key] if correct_key else ['correct','greedy_correct']
                label_path=next((p for p in paths if nested(row,p) is not None),None)
                correct=nested(row,label_path) if label_path else None
            
            if correct is None or isinstance(correct,(dict,list,str)) or correct not in (0,1):
                reject('missing_or_invalid_correctness',number);continue
            y.append(int(correct));scores.append(score);counts['valid']+=1;correct_paths[label_path]+=1
    if not y:raise ValueError(f'No valid labeled scalar scores: {dict(counts)}')
    direction=score_type or selected[1]
    return y,scores,{'counts':dict(counts),'score_key':selected[0],'score_type':direction,
                     'correct_keys':dict(correct_paths),'score_reducer':reducer,
                     'accuracy':sum(y)/len(y),'score_min':min(scores),'score_max':max(scores)}


def infer_jobs(path,requested=None,explicit_key=None):
    if explicit_key:return [(requested or 'custom',path)]
    stem=path.stem.split('__')[0]
    if requested:
        if stem==requested or stem=='cot_uq_four':return [(requested,path)]
        return []
    if stem=='cot_uq_four':return [(m,path) for m in ['cot_uq_me','cot_uq_mi','cot_uq_sar','cot_uq_sp']]
    return [(stem,path)] if stem in FIELDS else []


def write_csv_atomic(path,rows,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8-sig',newline='',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name);writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def process(path,output,method,args):
    if output.suffix.lower()!='.csv':raise ValueError('Output must end in .csv')
    metadata_path=output.with_suffix('.evaluation.json')
    if output.resolve()==path.resolve() or (args.correct_file and output.resolve()==Path(args.correct_file).resolve()):
        raise ValueError('Output must not overwrite input/labels')
    if not args.overwrite and (output.exists() or metadata_path.exists()):raise ValueError(f'Output exists: {output}')
    labels,scores,report=load_scores(path,method,args.score_key,args.score_type,args.correct_key,args.correct_file,
                                  args.reduce,args.invalid_policy)
    rows=evaluate(labels,scores,report['score_type'],args.bootstrap_runs,args.seed,args.bins,args.binning,
                  args.calibration,args.coverage)
    report.update(input_file=str(path.resolve()),method=method,bootstrap_runs=args.bootstrap_runs,seed=args.seed,
                  calibration=args.calibration,coverage=args.coverage,ece_bins=args.bins,ece_strategy=args.binning,
                  positive_class='correct=1',bootstrap_std_ddof=0,
                  normalization_fitted_once_before_bootstrap=True,metrics=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=metadata_path.with_name(metadata_path.name+'.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv_atomic(output,rows,['metric','mean','std','result','point_estimate','bootstrap_valid','bootstrap_requested'])
    os.replace(temp,metadata_path)
    return rows,report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-file','--input',dest='input_file')
    source.add_argument('--input-dir')
    p.add_argument('--output-file','--output',dest='output_file')
    p.add_argument('--output-dir')
    p.add_argument('--method',help='Infer from normalized filename when omitted')
    p.add_argument('--score-key','--confidence_key',dest='score_key',help='Explicit dotted field; requires score-type')
    p.add_argument('--score-type','--score_type',dest='score_type',choices=['confidence','uncertainty'])
    p.add_argument('--correct-key','--correct_key',dest='correct_key')
    p.add_argument('--correct-file',help='Optional external correctness JSONL, joined by ID and response index')
    p.add_argument('--reduce',choices=['mean','sum'],help='Explicitly reduce vector scores such as PMI')
    p.add_argument('--bootstrap-runs','--bootstrap_runs',dest='bootstrap_runs',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bins',type=int,default=10)
    p.add_argument('--binning',choices=['uniform','quantile'],default='uniform')
    p.add_argument('--calibration',choices=['minmax','probability','percent'],default='minmax')
    p.add_argument('--coverage',choices=['legacy_trapezoid','tie_expected_discrete'],default='legacy_trapezoid')
    p.add_argument('--invalid-policy',choices=['skip','error'],default='skip')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def main():
    p=parser();args=p.parse_args()
    try:
        if args.bootstrap_runs<0 or args.bins<1:raise ValueError('bootstrap-runs >=0 and bins >=1 required')
        if args.method:args.method=normalize_method(args.method)
        if args.score_key and not args.score_type:raise ValueError('--score-key requires --score-type')
        if args.input_file:
            source=Path(args.input_file)
            jobs=[(args.method,source)] if args.method else infer_jobs(source,None,args.score_key)
            if not jobs:raise ValueError('Cannot infer method; supply --method or --score-key and --score-type')
            if len(jobs)>1 and args.output_file:raise ValueError('Combined CoT-UQ input requires --output-dir or an explicit --method')
            if len(jobs)==1 and not args.output_file and not args.output_dir:raise ValueError('Supply --output-file or --output-dir')
            targets=[]
            for method,path in jobs:
                name=path.stem if len(jobs)==1 else path.stem.replace('cot_uq_four',method,1)
                target=Path(args.output_file) if args.output_file else Path(args.output_dir)/(name+'.csv')
                targets.append((method,path,target))
        else:
            if not args.output_dir or args.output_file:raise ValueError('Directory evaluation requires --output-dir')
            root=Path(args.input_dir)
            if not root.is_dir():raise ValueError('Input directory does not exist')
            targets=[]
            for path in sorted(root.rglob('*.jsonl')):
                for method,_ in infer_jobs(path,args.method,args.score_key):
                    if method=='pmi' and not args.reduce:continue
                    name=path.stem.replace('cot_uq_four',method,1)+'.csv'
                    targets.append((method,path,Path(args.output_dir)/path.relative_to(root).parent/name))
        if not targets:raise ValueError('No supported JSONL inputs found')
        if args.dry_run:
            print(json.dumps({'jobs':len(targets),'calibration':args.calibration,'coverage':args.coverage,
                              'plan':[{'method':m,'input':str(i),'output':str(o)}for m,i,o in targets]},ensure_ascii=False,indent=2));return
        if len({str(t.resolve()) for _,_,t in targets})!=len(targets):raise ValueError('Duplicate output paths')
        if args.output_dir and not args.overwrite:
            for name in ['evaluation_batch.json','summary.csv']:
                if (Path(args.output_dir)/name).exists():raise ValueError(f'Batch output exists: {name}; use --overwrite or another output directory')
        results=[];failures=[]
        for method,path,target in targets:
            try:
                rows,report=process(path,target,method,args)
                results.append({'input_file':str(path),'output_file':str(target),'method':method,
                    'valid_samples':report['counts']['valid'],'accuracy':report['accuracy'],
                    **{row['metric']:row['result']for row in rows}})
                print(f'{method}: {report["counts"]["valid"]} valid rows -> {target}',flush=True)
            except (ValueError,OSError) as exc:
                failures.append({'input_file':str(path),'method':method,'error':str(exc)})
                print(f'FAILED {path.name}: {exc}',file=sys.stderr)
        if args.output_dir:
            root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
            
            report=root/'evaluation_batch.json'
            if report.exists() and not args.overwrite:raise ValueError(f'Batch report exists: {report}')
            report.write_text(json.dumps({'requested_jobs':len(targets),'completed':len(results),'failures':failures,
                'calibration':args.calibration,'coverage':args.coverage},ensure_ascii=False,indent=2)+'\n')
            if results:write_csv_atomic(root/'summary.csv',results,list(results[0]))
        if failures:p.exit(2,f'{len(failures)} input(s) failed; {len(results)} completed.\n')
    except (ValueError,OSError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
