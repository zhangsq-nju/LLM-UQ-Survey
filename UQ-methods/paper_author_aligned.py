#!/usr/bin/env python3
# Original paper/author repository: https://github.com/Ybakman/LLM_Uncertainty
# Original paper/author repository: https://github.com/ZBox1005/CoT-UQ
# Original paper/author repository: https://github.com/Zhiyuan-GG/Conformal-Uncertainty-Criterion
# Original paper/author repository: https://github.com/potsawee/selfcheckgpt

from __future__ import annotations
import argparse
import ast
import hashlib
import gzip
import shutil
import string
import textwrap
import subprocess
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import numpy as np

METHOD = 'selfcheckgpt'
FAMILY = 'qwen'
DATASET = 'gsm8k'
SETTING = 'answer_only'


def finite(values, name='values'):
    result=np.asarray(values,dtype=float)
    if not np.isfinite(result).all():raise ValueError(f'Non-finite {name}')
    return result


def conu_paper(similarities, threshold=.7, coefficient=.5):
    
    s=finite(similarities,'similarity matrix')
    if s.ndim!=2 or s.shape[0]!=s.shape[1] or not len(s):raise ValueError('Square, nonempty similarity matrix required')
    if not np.allclose(s,s.T):raise ValueError('ConU matrix must be symmetric')
    clusters=[]
    for i in range(len(s)):
        cluster=next((c for c in clusters if s[i,c[0]]>=threshold),None)
        if cluster is None:clusters.append([i])
        else:cluster.append(i)
    frequencies=np.asarray([len(c)/len(s) for c in clusters])
    reps=[c[0] for c in clusters]
    scores=1-coefficient*frequencies-(1-coefficient)*(s[np.ix_(reps,reps)]@frequencies)/len(clusters)
    largest=int(np.argmax(frequencies))
    return float(scores[largest]),{'clusters':clusters,'frequencies':frequencies.tolist(),
        'candidate_cluster_uncertainties':scores.tolist(),'reference_index':reps[largest],
        'formula':'ConU paper Eq.(2)','lambda':coefficient,'threshold':threshold}


def mars_score(token_nll, phrase_token_indices, importance, mode='mean'):
    
    nll=finite(token_nll,'token NLL');weights=finite(importance,'importance')
    if len(nll)==0 or len(weights)!=len(phrase_token_indices):raise ValueError('Missing or mismatched MARS statistics')
    flat=[i for group in phrase_token_indices for i in group]
    if sorted(flat)!=list(range(len(nll))) or len(set(flat))!=len(flat):raise ValueError('Phrases must cover every scored token exactly once')
    if (weights<0).any() or weights.sum()<=0:raise ValueError('Invalid learned importance')
    weights=weights/weights.sum()
    reduce={'mean':np.mean,'max':np.max,'min':np.min}[mode]
    phrase_nll=np.asarray([reduce(nll[g]) for g in phrase_token_indices])
    return float(.5*np.dot(weights,phrase_nll)+.5*np.mean(nll))


def cot_aggregate(keywords, method):
    
    scores=[];weights=[]
    for word in keywords:
        p=finite(word['probabilities'],'keyword probabilities')
        if len(p)==0 or ((p<0)|(p>1)).any():raise ValueError('Invalid keyword probabilities')
        weight=float(word['importance'])
        if not math.isfinite(weight) or not 1<=weight<=10:raise ValueError('Keyword importance must be in [1,10]')
        if method=='cot_uq_me':value=float(p.mean())
        elif method=='cot_uq_mi':value=float(p.min())
        elif method=='cot_uq_sar':
            relevance=finite(word['relevance'],'semantic relevance')
            if relevance.shape!=p.shape or relevance.sum()<=0:raise ValueError('Invalid semantic relevance alignment/sum')
            
            value=float(np.dot(relevance/relevance.sum(),p))
        else:raise ValueError(method)
        scores.append(value);weights.append(weight)
    if not weights:raise ValueError('No scored keywords')
    confidence=float(np.dot(weights,scores)/sum(weights))
    return 1-confidence,confidence


def keyword_step(reason):
    matches=list(re.finditer(r'(?im)^\s*Step\s+(\d+)\s*:\s*',reason))
    if not matches:raise ValueError('CoT-UQ needs explicit Step i: reasoning, not invented steps')
    result=[]
    for i,match in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else len(reason)
        text=reason[match.end():end]
        final=re.search(r'(?im)^\s*Final\s+Answer\s*:',text)
        if final:end=match.end()+final.start()
        result.append({'id':int(match.group(1)),'text':reason[match.end():end].strip(),
                       'start':match.end(),'end':end})
    if len({x['id'] for x in result})!=len(result):raise ValueError('Duplicate step IDs')
    return result


def parse_keywords(text, steps):
    matches=list(re.finditer(r'(?im)^\s*Step\s+(\d+)\s*:\s*',text))
    entries=[];seen=set();step_map={s['id']:s for s in steps}
    for i,m in enumerate(matches):
        sid=int(m.group(1))
        if sid not in step_map or sid in seen:raise ValueError('Unknown/duplicate extracted step')
        seen.add(sid);body=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
        if body=='NO ANSWER':continue
        parts=re.findall(r'([^;\n]+?)\s*\(\s*/\s*(\d+(?:\.\d+)?)\s*/\s*\)',body)
        if not parts:raise ValueError('Cannot parse keyword/contribution output')
        for word,score in parts:
            word=word.strip();score=float(score)
            if not word or word not in step_map[sid]['text'] or not 1<=score<=10:raise ValueError('Hallucinated keyword or invalid contribution score')
            entries.append({'step':sid,'text':word,'importance':score})
    if seen!=set(step_map):raise ValueError('Keyword extraction omitted reasoning steps')
    if not entries:raise ValueError('No meaningful keywords extracted')
    return entries


def parse_probe(text):
    match=re.search(r'Confidence\s*:\s*(\d+(?:\.\d+)?)\s*%',text,re.I)
    if match is None:match=re.search(r'(\d+(?:\.\d+)?)\s*%',text)
    if match is None:raise ValueError('Self-probing must return an explicit percentage')
    score=float(match.group(1))/100
    if not 0<=score<=1:raise ValueError('Confidence outside [0,100]%')
    return score


class HFScorer:
    def __init__(self,path,device,family):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.torch=torch;self.family=family;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(path,use_fast=True)
        if not self.tokenizer.is_fast:raise ValueError('Fast tokenizer required for exact text/token alignment')
        self.model=AutoModelForCausalLM.from_pretrained(path,torch_dtype='auto').to(device).eval()
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
    def prompt(self,question):
        kwargs={'enable_thinking':False} if self.family=='qwen' else {}
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True,**kwargs)
        return question+'\n'
    def generate(self,prompt,max_tokens):
        batch=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(self.device)
        with self.torch.inference_mode():
            out=self.model.generate(**batch,max_new_tokens=max_tokens,do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
    def score(self,prompt,text):
        encoded=self.tokenizer(prompt+text,return_offsets_mapping=True,add_special_tokens=False)
        boundary=len(prompt);offsets=encoded.pop('offset_mapping')
        if any(a<boundary<b for a,b in offsets):raise ValueError('Prompt/response boundary crosses a token; use cached exact generation statistics')
        selected=[i for i,(a,b) in enumerate(offsets) if a>=boundary and b>a]
        if not selected or min(selected)==0:raise ValueError('No scoreable continuation tokens')
        ids=self.torch.tensor([encoded['input_ids']],device=self.device)
        with self.torch.inference_mode():
            logits=self.model(ids).logits[0].float()
            logp=logits.log_softmax(-1)
            values=[float(logp[i-1,ids[0,i]]) for i in selected]
        return {'logprobs':values,'offsets':[(offsets[i][0]-boundary,offsets[i][1]-boundary) for i in selected],
                'ids':[encoded['input_ids'][i] for i in selected]}


class LearnedImportance:
    def __init__(self,path,device,allow_pickle=False):
        import torch
        from transformers import AutoModelForTokenClassification,BertTokenizerFast
        if not path:raise ValueError('MARS requires the authors trained importance weights via --importance-model-path')
        self.torch=torch;self.device=device
        self.tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased')
        if Path(path).is_file():
            if not allow_pickle:raise ValueError('Original .pth is a pickled model; explicitly allow a trusted author checkpoint, or use a converted HF directory')
            self.model=torch.load(path,map_location=device,weights_only=False)
        else:self.model=AutoModelForTokenClassification.from_pretrained(path)
        self.model.to(device).eval()
        if self.model.config.num_labels!=3:raise ValueError('MARS checkpoint must have boundary(2)+importance(1) logits')
    def __call__(self,question,answer):
        words=list(re.finditer(r'\w+|[^\w\s]',answer))
        if not words:raise ValueError('Empty MARS response')
        encoded=self.tokenizer([question],[w.group() for w in words],is_split_into_words=True,
                              return_token_type_ids=True,truncation=False)
        if len(encoded['input_ids'])>self.model.config.max_position_embeddings:raise ValueError('MARS importance input exceeds checkpoint context; no silent truncation')
        ids=encoded.word_ids();types_=encoded['token_type_ids']
        batch={k:self.torch.tensor([v],device=self.device) for k,v in encoded.items()}
        with self.torch.inference_mode():logits=self.model(**batch).logits[0]
        starts=[];scores=[];seen=set()
        for i,(wid,typ) in enumerate(zip(ids,types_)):
            if typ!=1 or wid is None or wid in seen:continue
            seen.add(wid)
            if not starts or int(logits[i,:2].argmax())==0:
                starts.append(wid);scores.append(float(logits[i,2].sigmoid()))
        if seen!=set(range(len(words))):raise ValueError('Importance tokenizer did not cover full response')
        spans=[]
        for i,wid in enumerate(starts):
            finish=starts[i+1] if i+1<len(starts) else len(words)
            spans.append((words[wid].start(),words[finish-1].end()))
        return spans,scores


def align_phrases(offsets,spans,importance):
    
    spans=list(spans)
    if spans and offsets:spans[-1]=(spans[-1][0],max(spans[-1][1],offsets[-1][1]))
    groups=[];weights=[];cursor=0;i=0
    while i<len(spans):
        end=spans[i][1];j=i+1
        while True:
            crossing=next(((a,b) for a,b in offsets if a<end<b),None)
            if crossing is None or j==len(spans):break
            end=spans[j][1];j+=1
        selected=[]
        while cursor<len(offsets) and offsets[cursor][0]<end:
            selected.append(cursor);cursor+=1
        if not selected:raise ValueError('Empty MARS phrase token group')
        groups.append(selected);weights.append(float(np.mean(importance[i:j])));i=j
    if cursor!=len(offsets):raise ValueError('Trailing tokens not covered by learned phrases')
    return groups,weights


class Models:
    def __init__(self,args):self.args=args;self.lm=None;self.cross=None;self.importance=None;self.nli=None;self.nlp=None
    def scorer(self):
        if self.lm is None:
            if not self.args.model_path:raise ValueError('--model-path required to rescore/extract keywords')
            self.lm=HFScorer(self.args.model_path,self.args.device,FAMILY)
        return self.lm
    def similarity(self,pairs):
        if self.cross is None:
            from sentence_transformers import CrossEncoder
            self.cross=CrossEncoder(self.args.similarity_model,num_labels=1,device=self.args.device)
        return finite(self.cross.predict(pairs),'CrossEncoder predictions')
    def learned(self):
        if self.importance is None:self.importance=LearnedImportance(self.args.importance_model_path,self.args.device,self.args.allow_pickled_importance_model)
        return self.importance
    def selfcheck(self,target,samples):
        
        if self.nli is None:
            from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
            self.nli=SelfCheckNLI(nli_model=self.args.nli_model,device=self.args.device)
            if self.nli.model.config.num_labels!=2:raise ValueError('SelfCheckNLI requires the authors binary NLI checkpoint, not a neutral-inclusive 3-class replacement')
        if self.nlp is None:
            import spacy
            self.nlp=spacy.load(self.args.sentence_model)
        sentences=[str(s).strip() for s in self.nlp(target).sents if str(s).strip()]
        if not sentences:raise ValueError('No target sentences')
        scores=finite(self.nli.predict(sentences=sentences,sampled_passages=samples))
        return float(scores.mean()),{'sentences':sentences,'sentence_uncertainties':scores.tolist()}


def text_of(response):
    if isinstance(response,str):return response
    if response.get('raw_text') is not None:return str(response['raw_text'])
    if response.get('llm reason'):
        reason=str(response['llm reason']);answer=str(response.get('llm answer',''))
        match=re.search(r'(?im)^\s*Final\s+Answer\s*:\s*',reason)
        if match:return reason if reason[match.end():].strip() else reason+answer
        return reason+'\nFinal Answer: '+answer
    return str(response.get('text') or response.get('llm answer') or response.get('answer') or '')


def answer_of(response):
    if isinstance(response,str):return response
    return str(response.get('llm answer') or response.get('text') or response.get('answer') or text_of(response))


def select_response(item,index):
    responses=item.get('responses')
    if isinstance(responses,list):
        if index>=len(responses):raise ValueError('response_index out of range')
        response=responses[index]
        prediction=str(response.get('llm answer') or response.get('answer info',{}).get('text') or text_of(response))
        other=[answer_of(r) if METHOD=='conu' else text_of(r) for i,r in enumerate(responses) if i!=index]
    else:
        if index!=0:raise ValueError('answer_only has one greedy target')
        response=item.get('greedy') or item.get('greedy_answer')
        if response is None:raise ValueError('Input needs greedy or responses')
        prediction=str(response.get('text',text_of(response))) if isinstance(response,dict) else response
        other=[answer_of(r) if METHOD=='conu' else text_of(r) for r in item.get('samples',item.get('sampled_answers',[]))]
    if not prediction.strip() or not text_of(response).strip():raise ValueError('Empty target response')
    return response,prediction,other


def token_stats(response,models):
    gen=response.get('generation',response) if isinstance(response,dict) else {}
    text=text_of(response)
    steps=gen.get('steps')
    if steps and gen.get('raw_text')==text and all(s.get('chosen_logprob') is not None and s.get('char_start') is not None and s.get('char_end') is not None for s in steps):
        valid=[s for s in steps if s['char_end']>s['char_start']]
        result={'logprobs':[s['chosen_logprob'] for s in valid],
                'offsets':[(s['char_start'],s['char_end']) for s in valid],
                'ids':[s.get('token_id') for s in valid]}
        if (finite(result['logprobs'],'cached log probabilities')>1e-7).any():
            raise ValueError('Cached log probabilities must not be positive')
        offsets=result['offsets']
        if any(a<0 or b>len(text) or a>=b for a,b in offsets) or any(offsets[i][0]<offsets[i-1][0] for i in range(1,len(offsets))):
            raise ValueError('Invalid/nonmonotone generation token offsets')
        return result
    prompt=gen.get('prompt')
    if not prompt:raise ValueError('Missing exact generation prompt and aligned cached probabilities; regenerate stage1, do not invent a scoring context')
    return models.scorer().score(prompt,text)


def prepare_keywords(item,response,models):
    raw=text_of(response);steps=keyword_step(raw)
    supplied=response.get('cot_uq_keywords') if isinstance(response,dict) else None
    if supplied is not None:
        
        if not isinstance(supplied,list) or any(not isinstance(w,dict) or w.get('source')!='llm_keyword_importance' for w in supplied):
            raise ValueError('Cached keywords must declare source=llm_keyword_importance')
        words=[dict(w) for w in supplied]
    else:
        lm=models.scorer()
        template=KEYWORD_PROMPT_MATH if DATASET=='gsm8k' else KEYWORD_PROMPT_GENERIC
        prompt=template.replace('<QUESTION>',get_question(item)).replace('<RESPONSE>',raw)
        words=None;last=None
        for _ in range(models.args.retries+1):
            try:
                raw_extraction=lm.generate(lm.prompt(prompt),models.args.extraction_max_tokens)
                words=parse_keywords(raw_extraction,steps)
                for word in words:word.update(source='llm_keyword_importance',extraction_output=raw_extraction)
                break
            except ValueError as exc:last=exc
        if words is None:raise ValueError('Keyword extraction failed; no approximate fallback') from last
    smap={s['id']:s for s in steps}
    for word in words:
        if word.get('step') not in smap or not 1<=float(word['importance'])<=10:raise ValueError('Bad cached step/contribution')
        if word['text'] not in smap[word['step']]['text']:raise ValueError('Keyword not in its declared step')
    return raw,steps,words


def keyword_probabilities(words,steps,stats):
    smap={s['id']:s for s in steps}
    probs=np.exp(finite(stats['logprobs'],'keyword log probabilities'))
    for word in words:
        step=smap[word['step']];locations=[m.start() for m in re.finditer(re.escape(word['text']),step['text'])]
        occurrence=int(word.get('occurrence',0))
        if len(locations)>1 and 'occurrence' not in word:raise ValueError('Repeated keyword needs explicit zero-based occurrence; refuse ambiguous token alignment')
        if occurrence<0 or occurrence>=len(locations):raise ValueError('Keyword occurrence out of range')
        start=step['start']+locations[occurrence];end=start+len(word['text'])
        indices=[i for i,(a,b) in enumerate(stats['offsets']) if a<end and b>start]
        if not indices:raise ValueError('No tokens for keyword')
        word['probabilities']=probs[indices].tolist()
        word['token_indices']=indices
        word['token_pieces']=[word['text'][max(0,stats['offsets'][i][0]-start):min(end-start,stats['offsets'][i][1]-start)] for i in indices]


def cot_score(item,response,models,method):
    raw,steps,words=prepare_keywords(item,response,models)
    if method=='cot_uq_sp':
        mode=models.args.probe_context or ('key_step' if DATASET=='gsm8k' else 'key_keywords')
        if mode=='key_step':
            means=[np.mean([w['importance'] for w in words if w['step']==s['id']]) if any(w['step']==s['id'] for w in words) else 0 for s in steps]
            best=len(means)-1-int(np.argmax(means[::-1]));context=steps[best]['text']
            context_label='The most critical step in reasoning to the possible answer'
        else:
            selected=list(dict.fromkeys(w['text'] for w in words if float(w['importance'])>=models.args.keyword_threshold))
            if len(selected)<3:selected=list(dict.fromkeys(w['text'] for w in sorted(words,key=lambda w:-float(w['importance']))))[:3]
            context='; '.join(selected);context_label='Keywords during reasoning to the possible answer'
        answer=str(response.get('llm answer',response.get('text','')))
        prompt=(f'Question: {get_question(item)}\nPossible Answer: {answer}\n{context_label}: {context}\n'
                'Q: Considering this reasoning information as additional information, how likely is the above answer to be correct? '
                'Please first show your reasoning concisely and then answer with the following format:\nConfidence: [numerical probability]%')
        lm=models.scorer();failure=None
        for _ in range(models.args.retries+1):
            probe=lm.generate(lm.prompt(prompt),models.args.probe_max_tokens)
            try:
                confidence=parse_probe(probe)
                return 1-confidence,{'keywords':words,'probe_context_mode':mode,'probe_context':context,'probe_output':probe}
            except ValueError as exc:failure=exc
        raise ValueError('Self-probing failed after retries; no 0.5 fill') from failure
    stats=token_stats(response,models);keyword_probabilities(words,steps,stats)
    if method=='cot_uq_sar':
        q=get_question(item)
        for word in words:
            pieces=word['token_pieces']
            if any(not p for p in pieces):raise ValueError('Empty aligned keyword token')
            pairs=[(q+word['text'],q+word['text'].replace(piece,'')) for piece in pieces]
            word['relevance']=(1-models.similarity(pairs)).tolist()
    score,confidence=cot_aggregate(words,method)
    return score,{'keywords':words,'confidence':confidence,'aggregation':'paper Eq.(5)-(6)'}


def get_question(item):
    value=next((item[k] for k in ['question','prompt','input','problem'] if isinstance(item.get(k),str)),None)
    if not value:raise ValueError('Missing original question')
    return value


def score_item(item,response,samples,models):
    if METHOD=='conu':
        if not samples:raise ValueError('ConU needs independently sampled responses')
        n=len(samples);s=np.eye(n);pairs=[(get_question(item)+' '+samples[i],get_question(item)+' '+samples[j]) for i in range(n) for j in range(i+1,n)]
        values=iter(models.similarity(pairs)) if pairs else iter([])
        for i in range(n):
            for j in range(i+1,n):s[i,j]=s[j,i]=float(next(values))
        return conu_paper(s,models.args.semantic_threshold,models.args.conu_lambda)
    if METHOD=='selfcheckgpt':
        if not samples:raise ValueError('SelfCheck-NLI needs sampled evidence')
        return models.selfcheck(text_of(response),samples)
    if METHOD=='mars':
        text=text_of(response);stats=token_stats(response,models)
        spans,importance=models.learned()(get_question(item),text)
        groups,weights=align_phrases(stats['offsets'],spans,importance)
        score=mars_score(-finite(stats['logprobs']),groups,weights,models.args.mars_phrase_mode)
        return score,{'phrase_spans':spans,'importance':importance,'merged_importance':weights,
                      'phrase_token_indices':groups,'phrase_mode':models.args.mars_phrase_mode}
    return cot_score(item,response,models,METHOD)


def load_labels(path):
    labels={}
    if not path:return labels
    for i,row in enumerate(read_examples(path)):
        qid=next((row[k] for k in ['id','question_id','task_id'] if row.get(k) is not None),None)
        index=row.get('response_index',row.get('target_response_index',0));v=row.get('correct')
        if qid is None or v is None or isinstance(v,str) or v not in (0,1) or not isinstance(index,int) or index<0:raise ValueError('Invalid external unit-test label')
        key=(str(qid),index)
        if key in labels:raise ValueError('Duplicate external label')
        labels[key]=int(v)
    return labels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file',required=True);parser.add_argument('--output-file',required=True)
    parser.add_argument('--model-path');parser.add_argument('--model-family',default=FAMILY)
    parser.add_argument('--correct-file');parser.add_argument('--device',default='cpu')
    parser.add_argument('--nli-model',default='potsawee/deberta-v3-large-mnli')
    parser.add_argument('--similarity-model',default='cross-encoder/stsb-distilroberta-base' if METHOD=='conu' else 'cross-encoder/stsb-roberta-large')
    parser.add_argument('--sentence-model',default='en_core_web_sm')
    parser.add_argument('--importance-model-path');parser.add_argument('--allow-pickled-importance-model',action='store_true')
    parser.add_argument('--mars-phrase-mode',choices=['mean','max','min'],default='mean')
    parser.add_argument('--semantic-threshold',type=float,default=.7)
    parser.add_argument('--conu-lambda',type=float,default=.5)
    parser.add_argument('--response-index',type=int,default=0)
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--probe-context',choices=['key_step','key_keywords'])
    parser.add_argument('--keyword-threshold',type=float,default=4)
    parser.add_argument('--probe-max-tokens',type=int,default=256)
    parser.add_argument('--extraction-max-tokens',type=int,default=1024)
    parser.add_argument('--retries',type=int,default=2)
    parser.add_argument('--limit',type=int);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--overwrite',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    inp=Path(args.input_file).resolve();out=Path(args.output_file).resolve()
    if inp==out or not inp.is_file():parser.error('Invalid input/output path')
    if out.exists() and not args.overwrite:parser.error('Output exists; use --overwrite')
    if args.model_family!=FAMILY:parser.error('This file fixes model family '+str(FAMILY))
    if args.response_index<0 or args.retries<0 or (args.limit is not None and args.limit<1) or (args.max_samples is not None and args.max_samples<1):parser.error('Invalid index/limits')
    if not 0<=args.semantic_threshold<=1 or not 0<=args.conu_lambda<=1:parser.error('Invalid ConU parameters')
    if args.probe_max_tokens<1 or args.extraction_max_tokens<1:parser.error('Token limits must be positive')
    if DATASET=='humaneval' and not args.correct_file:parser.error('HumanEval requires --correct-file')
    if METHOD=='mars' and not args.importance_model_path:parser.error('MARS requires author-trained --importance-model-path')
    if args.dry_run:
        print(json.dumps({'method':METHOD,'family':FAMILY,'dataset':DATASET,'setting':SETTING,'implementation':'paper_author_aligned'}));return
    import torch
    import random
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    models=Models(args);labels=load_labels(args.correct_file)
    evaluator=Evaluator(DATASET,human_mode='skip') if DATASET!='humaneval' else None
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+out.name+'.',dir=out.parent)
    seen=set()
    try:
        with os.fdopen(fd,'w') as stream:
            for i,item in enumerate(read_examples(inp)):
                if args.limit is not None and i>=args.limit:break
                qid=str(next((item[k] for k in ['id','question_id','task_id','qid'] if item.get(k) is not None),i))
                if qid in seen:raise ValueError('Duplicate input ID: '+qid)
                seen.add(qid)
                response,prediction,samples=select_response(item,args.response_index)
                if args.max_samples is not None:samples=samples[:args.max_samples]
                score,details=score_item(item,response,samples,models)
                if not math.isfinite(score):raise ValueError('Non-finite method score')
                if DATASET=='humaneval':
                    key=(qid,args.response_index)
                    if key not in labels:raise ValueError('Missing unit-test label: '+str(key))
                    correct=labels[key]
                else:
                    example=normalize_example(item,DATASET,i)
                    verdict=evaluator(example,prediction)
                    if verdict['correct'] is None:raise ValueError('Correctness evaluation failed: '+str(verdict))
                    correct=verdict['correct']
                field={'mars':'mars_uncertainty','conu':'conu_uncertainty','selfcheckgpt':'selfcheck_uncertainty',
                       'cot_uq_me':'cotuq_probas_mean_uncertainty','cot_uq_mi':'cotuq_probas_min_uncertainty',
                       'cot_uq_sar':'cotuq_token_sar_uncertainty','cot_uq_sp':'cotuq_self_probing_uncertainty'}[METHOD]
                row={'id':qid,'question':get_question(item),'prediction':prediction,'correct':correct,
                     'response_index':args.response_index,'target_response_index':args.response_index,
                     'method':METHOD,'implementation':'paper_author_aligned','dataset':DATASET,'family':FAMILY,
                     'setting':SETTING,field:score,'score':score,'score_type':'uncertainty','details':details,
                     'run':vars(args),'sources':SOURCES}
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if not seen:raise ValueError('No input records')
        os.replace(tmp,out)
    finally:Path(tmp).unlink(missing_ok=True)




def read_examples(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        if path.suffix == '.json':
            rows = json.load(stream)
            if isinstance(rows, dict):
                rows = rows.get('data', rows.get('examples'))
            if not isinstance(rows, list):
                raise ValueError('JSON input must be an array or contain data/examples array')
            yield from rows
        else:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f'Input line {number} is not an object')
                    yield row

def normalize_example(row, dataset, index):
    qid = next((row[k] for k in ['id', 'question_id', 'task_id', 'qid'] if row.get(k) is not None), str(index))
    question = next((row[k] for k in ['question', 'prompt', 'problem', 'input', 'query'] if isinstance(row.get(k), str)), None)
    if not question:
        raise ValueError(f'{qid}: missing question/prompt')
    gold = next((row[k] for k in ['gt', 'ground_truth', 'correct answer', 'correct_answer', 'answer', 'answers', 'target'] if row.get(k) is not None), None)
    if dataset == 'humaneval':
        gt = dict(gold) if isinstance(gold, dict) else {}
        for key in ['test', 'entry_point', 'canonical_solution']:
            if key in row:
                gt[key] = row[key]
        if not gt.get('test') or not gt.get('entry_point'):
            raise ValueError(f'{qid}: HumanEval requires test and entry_point')
    else:
        if isinstance(gold, dict):
            value = gold.get('value', gold.get('normalized_value', gold.get('text')))
            aliases = gold.get('aliases', gold.get('normalized_aliases', [])) or []
        elif isinstance(gold, list):
            value = gold[0] if gold else None
            aliases = gold
        else:
            value, aliases = (gold, row.get('aliases', []) or [])
        if value is None or str(value).strip() == '':
            raise ValueError(f'{qid}: missing gold answer')
        value = str(value)
        if dataset == 'gsm8k' and '####' in value:
            value = value.rsplit('####', 1)[1].strip()
        gt = {'value': value, 'aliases': list(dict.fromkeys([value] + [str(x) for x in aliases]))}
    return {'id': str(qid), 'question': question, 'gt': gt}

def normalize_trivia(text):
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub('\\b(a|an|the)\\b', ' ', text).split())

def assemble_code(prompt, answer, entry_point):
    try:
        tree = ast.parse(answer)
        if any((isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point for n in tree.body)):
            imports = '\n'.join((ast.unparse(n) for n in ast.parse(prompt).body if isinstance(n, (ast.Import, ast.ImportFrom))))
            return imports + '\n' + answer
    except SyntaxError:
        pass
    return prompt.rstrip() + '\n' + textwrap.indent(textwrap.dedent(answer).strip(), '    ') + '\n'

class Evaluator:

    def __init__(self, dataset, human_mode='docker', image='python:3.11-slim', timeout=10):
        self.dataset, self.human_mode, self.image, self.timeout = (dataset, human_mode, image, timeout)
        if dataset == 'gsm8k':
            try:
                from math_verify import parse, verify
            except ImportError as exc:
                raise RuntimeError('GSM8K correctness requires pip install math-verify') from exc
            self.parse, self.verify = (parse, verify)
        if dataset == 'humaneval' and human_mode == 'docker':
            if not shutil.which('docker'):
                raise RuntimeError('HumanEval testing requires Docker; use --human-eval skip only to generate unevaluated data')
            check = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, timeout=20)
            if check.returncode:
                raise RuntimeError(f'Docker image unavailable: {image}; pull it before generation')

    def __call__(self, example, answer):
        if self.dataset == 'triviaqa':
            pred = normalize_trivia(answer)
            correct = bool(pred) and any((pred == normalize_trivia(x) for x in example['gt']['aliases']))
            return {'correct': int(correct), 'status': 'normalized_alias_match'}
        if self.dataset == 'gsm8k':
            try:
                pred = self.parse(answer)
                gold = self.parse(example['gt']['value'])
                correct = bool(pred) and bool(gold) and bool(self.verify(gold, pred))
                return {'correct': int(correct), 'status': 'math_verify'}
            except Exception as exc:
                return {'correct': None, 'status': 'evaluation_error', 'evaluation_error': str(exc)}
        if self.human_mode == 'skip':
            return {'correct': None, 'status': 'not_evaluated'}
        code = assemble_code(example['gt'].get('prompt', example['question']), answer, example['gt']['entry_point'])
        harness = code + '\n' + example['gt']['test'] + '\ncheck(' + example['gt']['entry_point'] + ')\n'
        name = 'uq-eval-' + __import__('uuid').uuid4().hex
        command = ['docker', 'run', '--rm', '-i', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '256m', '--cpus', '1', '--user', '65534:65534', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m', self.image, 'python', '-I', '-B', '-']
        try:
            result = subprocess.run(command, input=harness, text=True, capture_output=True, timeout=self.timeout)
            if result.returncode in {125, 126, 127}:
                raise RuntimeError('Docker infrastructure failure: ' + result.stderr[-1000:])
            return {'correct': int(result.returncode == 0), 'status': 'unit_test_pass' if result.returncode == 0 else 'unit_test_fail', 'test_stderr': result.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {'correct': 0, 'status': 'unit_test_timeout'}
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

KEYWORD_PROMPT_GENERIC = ' \nYou will be provided with a question and a multi-step response containing reasoning steps. \nFor each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.\nYou also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.\nIf you find more than one keyword in a specific step, separate them with “;”.\nIf a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:\n\nQuestion:\n<QUESTION>\nMulti-Step Response:\n<RESPONSE>\nKeywords for Each Reasoning Step:\n'


KEYWORD_PROMPT_MATH = ' \nYou will be provided with a question and a multi-step response containing reasoning steps. \nFor each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.\nYou also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.\nIf you find more than one keyword in a specific step, separate them with “;”.\nIf a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:\nQ: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nA: Step 1: Identify the amount of blue fiber needed. The robe requires 2 bolts of blue fiber.  \nStep 2: Determine the amount of white fiber needed. It is half the amount of blue fiber, which is 2 ÷ 2 = 1 bolt.  \nStep 3: Compute the total number of bolts. Add the bolts of blue fiber (2) and white fiber (1) to get 3 bolts.\nFinal Answer: 3\nKeywords for Each Reasoning Step: Step 1: 2 bolts (/3/)\nStep 2: 1 bolt (/10/)\nStep 3: 3 bolts (/7/)\n\nThe following is your task:\nQ: <QUESTION>\nA: <RESPONSE>\nKeywords for Each Reasoning Step:\n'


SOURCES = {'mars': {'repository': 'https://github.com/Ybakman/LLM_Uncertainty', 'revision': '264a349fb7c723937048f771a2f3af15c50719cd', 'files': {'get_likelihoods.py': '1697a026c0b51184e58952ee8777af0e8f9546a318397eb5b0385bf7e7b19590', 'get_semantic_similarities.py': '3d0bce2ae540109b8137492775458671101f61027df334dc2490780d3ab9d9de'}}, 'cot': {'repository': 'https://github.com/ZBox1005/CoT-UQ', 'revision': '62d95752c83713fba1d1a056563a546d06737704', 'files': {'stepuq.py': '994c58ef246a3fb8b3ca55f5a635780898177fe6f9270adca08fa4f0de38317d', 'inference_refining.py': '689cccb1bbe5135b479a131b40133e136c680c8086d49854dce8e4129ad635f5', 'utils.py': '0cce6bd006cd04555265ecf2e79bf71e302764d73d40959c1c0d806344e74aa9', 'src/format/get_step_exact_tokens.py': '58a5057af5f9fbce7f778f5001cdfc16f443114e0d1815b17ed76c086006fdc3'}}, 'conu': {'repository': 'https://github.com/Zhiyuan-GG/Conformal-Uncertainty-Criterion', 'revision': 'c0c450e8dbef489cfed28f585f1a4ddf0e66078b', 'files': {'conformal_uncertainty_criterion.py': '943f6e1e289d89765fbfdbc72b4ba3428f1c0fb62aaa633f5c1f8230aed23d09', 'open_domain_qa_similarity.py': '0cff56708e781c151f4cf9b1b715ad8e932fbe4dad7c3d87105eb2af2a05913e'}}, 'selfcheck': {'repository': 'https://github.com/potsawee/selfcheckgpt', 'revision': '19b492a2a380931bf1ed0ca94a9565c9aa7b03e1', 'files': {'selfcheckgpt/modeling_selfcheck.py': '813175327153e4d44fac690e02236d28aecb4e5b2049fae4de25fd8a509be2d0'}}}


if __name__ == '__main__':
    main()
