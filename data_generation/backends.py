#!/usr/bin/env python3

from __future__ import annotations
import math
from pathlib import Path


def shared_prefix_length(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:
            return i
    return min(len(a),len(b))


def aligned_offsets(decoded_prefixes, text):
    previous=0
    spans=[]
    for prefix in decoded_prefixes:
        end=max(previous, shared_prefix_length(prefix,text))
        spans.append((previous,end))
        previous=end
    if spans and previous != len(text):
        raise ValueError('Could not align generated token boundaries to decoded output')
    return spans


class HFBackend:
    def __init__(self,args,model=None,tokenizer=None):
        import torch
        self.torch,self.args=torch,args
        if model is None:
            from transformers import AutoModelForCausalLM,AutoTokenizer
            dtype={'auto':'auto','fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[args.dtype]
            self.tokenizer=AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=args.trust_remote_code)
            kwargs=dict(torch_dtype=dtype,trust_remote_code=args.trust_remote_code)
            if args.device=='cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA requested but unavailable')
                kwargs['device_map']='auto'
            self.model=AutoModelForCausalLM.from_pretrained(args.model_path,**kwargs)
            if args.device!='cuda':
                self.model.to(args.device)
            self.model.eval()
        else:
            self.model,self.tokenizer=model,tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id=self.tokenizer.eos_token_id

    def prompt(self,messages):
        mode=self.args.chat_template
        has=bool(getattr(self.tokenizer,'chat_template',None))
        if mode=='on' and not has:
            raise ValueError('--chat-template on requires a tokenizer chat template')
        if mode=='on' or (mode=='auto' and has):
            kwargs=dict(tokenize=False,add_generation_prompt=True)
            if self.args.family=='qwen':
                kwargs['enable_thinking']=False
            return self.tokenizer.apply_chat_template(messages,**kwargs)
        return '\n\n'.join(m['role'].capitalize()+': '+m['content'] for m in messages)+'\n\nAssistant:'

    def generate(self,messages,do_sample,seed,distribution_path=None):
        torch=self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prompt=self.prompt(messages)
        inputs=self.tokenizer(prompt,return_tensors='pt')
        if inputs['input_ids'].shape[-1]>self.args.max_input_tokens:
            raise ValueError('Prompt exceeds --max-input-tokens; refusing silent truncation')
        device=self.model.get_input_embeddings().weight.device
        inputs={k:v.to(device) for k,v in inputs.items()}
        input_len=inputs['input_ids'].shape[-1]
        kwargs=dict(max_new_tokens=self.args.max_new_tokens,do_sample=do_sample,
                    return_dict_in_generate=True,output_scores=True,output_logits=True,
                    pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id)
        
        kwargs.update(temperature=self.args.temperature if do_sample else 1.0,
                      top_p=self.args.top_p if do_sample else 1.0,
                      top_k=self.args.top_k if do_sample else 0,
                      num_beams=1,repetition_penalty=1.0)
        with torch.inference_mode():
            result=self.model.generate(**inputs,**kwargs)
        ids=result.sequences[0,input_len:].tolist()
        if getattr(result,'logits',None) is None or len(result.logits)!=len(ids):
            raise RuntimeError('Model did not return raw logits; cannot label processed scores as raw probabilities')
        text=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        prefixes=[self.tokenizer.decode(ids[:i+1],skip_special_tokens=True,clean_up_tokenization_spaces=False) for i in range(len(ids))]
        offsets=aligned_offsets(prefixes,text)
        raw_steps=[];full=[]
        special=set(self.tokenizer.all_special_ids)
        def top_entries(logp):
            values,indices=torch.topk(logp,min(self.args.top_logprobs,logp.shape[-1]))
            return [{'token_id':int(t),'token_text':self.tokenizer.decode([int(t)],clean_up_tokenization_spaces=False),
                     'logprob':float(v),'prob':math.exp(float(v))} for v,t in zip(values.tolist(),indices.tolist()) if math.isfinite(v)]
        for i,token in enumerate(ids):
            raw=torch.log_softmax(result.logits[i][0].float(),dim=-1)
            sampling=torch.log_softmax(result.scores[i][0].float(),dim=-1)
            probs=raw.exp()
            entropy=float(-(torch.where(probs>0,probs*raw,torch.zeros_like(raw))).sum().item())
            lp=float(raw[token].item());slp=float(sampling[token].item())
            start,end=offsets[i]
            raw_steps.append({'token_id':token,'token_text':self.tokenizer.decode([token],clean_up_tokenization_spaces=False),
                              'text_piece':text[start:end],'char_start':start,'char_end':end,
                              'is_special':token in special,'chosen_logprob':lp,'chosen_prob':math.exp(lp),
                              'sampling_logprob':slp,'sampling_prob':math.exp(slp),'max_logprob':float(raw.max().item()),
                              'entropy':entropy,'top_logprobs':top_entries(raw),
                              'sampling_top_logprobs':top_entries(sampling)})
            if distribution_path:
                full.append(raw.cpu().numpy())
        if distribution_path:
            import numpy as np
            path=Path(distribution_path);path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,token_ids=np.array(ids),logprobs=np.stack(full) if full else np.empty((0,0)))
        return {'messages':messages,'prompt':prompt,'raw_text':text,
                'steps':[s for s in raw_steps if not s['is_special']], 'raw_steps':raw_steps,
                'token_ids':ids,'seed':seed,'do_sample':do_sample,
                'finish_reason':'length' if len(ids)>=self.args.max_new_tokens and (not ids or ids[-1] not in special) else 'stop',
                'probability_source':'raw_model_logits','entropy_source':'full_vocabulary',
                'full_distribution_file':str(distribution_path) if distribution_path else None}


class DeepSeekBackend:
    def __init__(self,args,client=None):
        import os
        self.args=args
        if client is None:
            from openai import OpenAI
            key=os.environ.get(args.api_key_env)
            if not key:
                raise RuntimeError(f'Set {args.api_key_env} before generating via API')
            self.client=OpenAI(api_key=key,base_url=args.base_url,timeout=args.api_timeout,max_retries=args.api_retries)
        else:
            self.client=client

    def generate(self,messages,do_sample,seed,distribution_path=None):
        if distribution_path:
            raise ValueError('DeepSeek API cannot provide full-vocabulary distributions')
        kwargs=dict(model=self.args.model_path,messages=messages,max_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature if do_sample else 0,
                    top_p=self.args.top_p if do_sample else 1.0,
                    extra_body={'thinking':{'type':'disabled'}})
        if self.args.api_logprobs=='on':
            kwargs.update(logprobs=True,top_logprobs=min(self.args.top_logprobs,20))
        result=self.client.chat.completions.create(**kwargs)
        choice=result.choices[0];text=choice.message.content or ''
        values=getattr(getattr(choice,'logprobs',None),'content',None) or []
        prefixes=[];raw=[];buffer=b''
        for token in values:
            buffer+=bytes(token.bytes) if getattr(token,'bytes',None) is not None else token.token.encode('utf-8')
            prefixes.append(buffer.decode('utf-8',errors='replace'))
        spans=aligned_offsets(prefixes,text) if values else []
        for token,(start,end) in zip(values,spans):
            lp=float(token.logprob)
            lp=lp if math.isfinite(lp) and lp>-1000 else None
            top=[{'token_text':t.token,'token_id':None,'logprob':float(t.logprob),'prob':math.exp(float(t.logprob))}
                 for t in (getattr(token,'top_logprobs',None) or []) if math.isfinite(t.logprob) and t.logprob>-1000]
            raw.append({'token_id':None,'token_text':token.token,'text_piece':text[start:end],
                        'char_start':start,'char_end':end,'is_special':False,
                        'chosen_logprob':lp,'chosen_prob':math.exp(lp) if lp is not None else None,
                        'sampling_logprob':None,'sampling_prob':None,
                        'max_logprob':max((t['logprob'] for t in top),default=None),
                        'entropy':None,'top_logprobs':top,'sampling_top_logprobs':[]})
        return {'messages':messages,'prompt':None,'raw_text':text,'steps':raw,'raw_steps':raw,
                'token_ids':None,'seed':None,'requested_seed':seed,'do_sample':do_sample,
                'finish_reason':choice.finish_reason,'probability_source':'api_reported_logprobs' if values else 'unavailable',
                'entropy_source':'unavailable','full_distribution_file':None,
                'provider_model':getattr(result,'model',None),'system_fingerprint':getattr(result,'system_fingerprint',None),
                'usage':result.usage.model_dump() if getattr(result,'usage',None) else None}
