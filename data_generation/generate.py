#!/usr/bin/env python3

from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import importlib.metadata

if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_generation.schema import read_examples,normalize_example,build_messages,finalize_generation,reasoning_response
from data_generation.backends import HFBackend,DeepSeekBackend
from data_generation.evaluation import Evaluator


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',required=True,choices=['qwen','llama','deepseek'])
    p.add_argument('--model-path',required=True,help='Local checkpoint/HuggingFace id, or actual API model id')
    p.add_argument('--dataset',required=True,choices=['gsm8k','triviaqa','humaneval'])
    p.add_argument('--setting',choices=['answer_only','reasoning_augmented'],default='answer_only')
    p.add_argument('--input-file',required=True)
    p.add_argument('--output-file',required=True)
    p.add_argument('--num-samples',type=int,default=10,help='Additional stochastic generations per question, besides greedy')
    p.add_argument('--temperature',type=float,default=0.7)
    p.add_argument('--top-p',type=float,default=0.8)
    p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--top-logprobs',type=int,default=10)
    p.add_argument('--max-new-tokens',type=int,default=512,help='Increase for reasoning/coding tasks')
    p.add_argument('--max-input-tokens',type=int,default=8192)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cuda')
    p.add_argument('--dtype',choices=['auto','fp32','fp16','bf16'],default='auto')
    p.add_argument('--chat-template',choices=['auto','on','off'],default='auto')
    p.add_argument('--trust-remote-code',action='store_true')
    p.add_argument('--save-full-distributions',action='store_true',help='Store raw full-vocabulary token logprobs in NPZ sidecars (large)')
    p.add_argument('--human-eval',choices=['docker','skip'],default='docker')
    p.add_argument('--docker-image',default='python:3.11-slim')
    p.add_argument('--test-timeout',type=float,default=10)
    p.add_argument('--base-url',default='https://api.deepseek.com')
    p.add_argument('--api-key-env',default='DEEPSEEK_API_KEY')
    p.add_argument('--api-logprobs',choices=['on','off'],default='on')
    p.add_argument('--api-timeout',type=float,default=120)
    p.add_argument('--api-retries',type=int,default=3)
    p.add_argument('--limit',type=int,help='Limit input questions; resume counts this limit against the full input')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--resume',action='store_true')
    mode.add_argument('--overwrite',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def stable_seed(seed,qid,index):
    return int.from_bytes(hashlib.sha256(f'{seed}:{qid}:{index}'.encode()).digest()[:4],'big')


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path,obj):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def summary_and_labels(output,dataset,config):
    stats={'questions':0,'greedy_labeled':0,'greedy_correct':0,'sampled_responses':0,'sampled_labeled':0,
           'sampled_correct':0,'generations_with_logprobs':0,'total_generations':0,'truncated_generations':0,
           'missing_final_answer_markers':0}
    pass_values={};labels=output.with_suffix('.correct.jsonl');tmp=labels.with_name(labels.name+'.tmp')
    with output.open() as stream,tmp.open('w') as labels_stream:
        for line in stream:
            row=json.loads(line);stats['questions']+=1
            greedy=row['greedy'];samples=row['samples']
            if greedy['correct'] is not None:
                stats['greedy_labeled']+=1;stats['greedy_correct']+=int(greedy['correct'])
                labels_stream.write(json.dumps({'question_id':row['id'],'correct':greedy['correct'],
                    'response_index':0,'target':'greedy','sample_correct':[s['correct'] for s in samples]})+'\n')
            for generation in [greedy]+samples:
                stats['total_generations']+=1
                stats['generations_with_logprobs']+=int(generation['probabilities_available'])
                stats['truncated_generations']+=int(generation['finish_reason']=='length')
                stats['missing_final_answer_markers']+=int(not generation['final_answer_marker_found'])
            stats['sampled_responses']+=len(samples)
            valid=[s['correct'] for s in samples if s['correct'] is not None]
            stats['sampled_labeled']+=len(valid);stats['sampled_correct']+=sum(valid)
            if dataset=='humaneval' and len(valid)==len(samples):
                n=len(samples);c=sum(valid)
                for k in [1,5,10]:
                    if k<=n:
                        estimate=1.0 if n-c<k else 1-math.comb(n-c,k)/math.comb(n,k)
                        pass_values.setdefault(k,[]).append(estimate)
    os.replace(tmp,labels)
    stats['greedy_accuracy']=stats['greedy_correct']/stats['greedy_labeled'] if stats['greedy_labeled'] else None
    stats['sample_accuracy']=stats['sampled_correct']/stats['sampled_labeled'] if stats['sampled_labeled'] else None
    stats['greedy_evaluation_coverage']=stats['greedy_labeled']/stats['questions'] if stats['questions'] else 0
    stats['pass_at_k']={str(k):{'mean':sum(v)/len(v),'evaluated_questions':len(v)} for k,v in pass_values.items()}
    stats['config']=config
    write_json(output.with_suffix('.summary.json'),stats)
    return stats


def run(args,backend=None,evaluator=None):
    if args.num_samples<1 or not 0<args.temperature or not 0<args.top_p<=1 or args.top_k<0:
        raise ValueError('Require num-samples>=1, temperature>0, 0<top-p<=1, top-k>=0')
    if args.max_new_tokens<1 or args.max_input_tokens<1 or args.top_logprobs<1 or (args.limit is not None and args.limit<1):
        raise ValueError('Token limits, top-logprobs and limit must be positive')
    if args.family=='deepseek' and args.setting=='reasoning_augmented':
        raise ValueError('Paper configuration covers DeepSeek answer_only only')
    if args.family=='deepseek' and args.save_full_distributions:
        raise ValueError('Full distributions require a local white-box model')
    output=Path(args.output_file).resolve();input_path=Path(args.input_file).resolve()
    if output.suffix != '.jsonl':
        raise ValueError('--output-file must end in .jsonl')
    artifacts=[output,output.with_suffix('.meta.json'),output.with_suffix('.summary.json'),output.with_suffix('.correct.jsonl')]
    if input_path in artifacts:
        raise ValueError('Output or sidecar would overwrite input')
    examples=iter(read_examples(input_path))
    try:first=normalize_example(next(examples),args.dataset,0)
    except StopIteration:raise ValueError('Input dataset is empty')
    if output.exists() and not (args.overwrite or args.resume):
        raise ValueError('Output exists; use --resume or --overwrite')
    if args.resume and not output.exists():
        raise ValueError('Cannot resume a missing output')
    if args.dry_run:
        print(json.dumps({'family':args.family,'model_path':args.model_path,'dataset':args.dataset,
            'setting':args.setting,'first_id':first['id'],'generations_per_question':1+args.num_samples,
            'output_files':[str(p) for p in artifacts], 'model_loading':False},ensure_ascii=False,indent=2))
        return
    config={k:v for k,v in vars(args).items() if k not in {'resume','overwrite','dry_run','output_file'}}
    config['input_file']=str(input_path);config['input_sha256']=file_hash(input_path)
    config['schema_version']='uq-stage1-v1'
    config['python_version']=sys.version.split()[0]
    config['package_versions']={}
    for package in ['torch','transformers','openai','math-verify']:
        try:config['package_versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:config['package_versions'][package]=None
    meta=output.with_suffix('.meta.json');done=set()
    if args.resume:
        if not meta.exists() or json.loads(meta.read_text())['config']!=config:
            raise ValueError('Resume configuration/input differs; use the same parameters and source dataset')
        
        with output.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['id'] in done:raise ValueError('Duplicate IDs in resume output')
                done.add(row['id'])
    evaluator=evaluator or Evaluator(args.dataset,args.human_eval,args.docker_image,args.test_timeout)
    backend=backend or (DeepSeekBackend(args) if args.family=='deepseek' else HFBackend(args))
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(meta,{'config':config,'status':'running'})
    seen=set()
    rows=itertools.chain([first],(normalize_example(row,args.dataset,i) for i,row in enumerate(examples,1)))
    if args.limit:rows=itertools.islice(rows,args.limit)
    try:
        with output.open('a' if args.resume else 'w') as stream:
            for example in rows:
                qid=example['id']
                if qid in seen:raise ValueError(f'Duplicate input ID: {qid}')
                seen.add(qid)
                if qid in done:continue
                messages=build_messages(example,args.dataset,args.setting)
                generations=[]
                for index in range(1+args.num_samples):
                    distro=None
                    if args.save_full_distributions:
                        stem=hashlib.sha256(qid.encode()).hexdigest()[:20]
                        distro=output.with_suffix('.distributions')/(stem+f'_{index}.npz')
                    gen=backend.generate(messages,index>0,stable_seed(args.seed,qid,index),distro)
                    if gen.get('full_distribution_file'):
                        gen['full_distribution_file']=os.path.relpath(gen['full_distribution_file'],output.parent)
                    generations.append(finalize_generation(gen,example,args.dataset,args.setting,evaluator))
                greedy,*samples=generations
                row={**example,'question_id':qid,'prompt':example['question'],'ground_truth':example['gt'],
                     'correct answer':example['gt']['value'] if args.dataset=='gsm8k' else example['gt'],
                     'greedy':greedy,'greedy_answer':greedy['text'],'samples':samples,
                     'sampled_answers':[g['text'] for g in samples], 'correct':greedy['correct'],
                     'sample_accuracy':sum(g['correct'] for g in samples)/len(samples) if all(g['correct'] is not None for g in samples) else None,
                     'generation_meta':{'family':args.family,'model_path':args.model_path,'dataset':args.dataset,
                                        'setting':args.setting,'num_samples':args.num_samples,'config_file':meta.name}}
                if args.setting=='reasoning_augmented':
                    row['responses']=[reasoning_response(g,args.dataset) for g in generations]
                    row['target_response_index']=0
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                print(f'{qid}: correct={greedy["correct"]}, samples={len(samples)}',flush=True)
        summary=summary_and_labels(output,args.dataset,config)
        write_json(meta,{'config':config,'status':'complete'})
        print(json.dumps({k:summary[k] for k in ['questions','greedy_accuracy','sample_accuracy','generations_with_logprobs']},ensure_ascii=False))
    except BaseException:
        write_json(meta,{'config':config,'status':'interrupted'})
        raise


def main():
    p=parser();args=p.parse_args()
    try:run(args)
    except (ValueError,RuntimeError,OSError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
