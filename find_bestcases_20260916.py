import json, re, hashlib
from pathlib import Path
root=Path('./external/Mem-T')
out=root/'BestCases_20260916'
out.mkdir(exist_ok=True)
raw=json.loads((root/'Mem-T-main/data/locomo/locomo10.json').read_text())
samples={s['sample_id']:s for s in raw}
def norm(x):
    return re.sub(r'\s+',' ',re.sub(r'\b(a|an|the)\b',' ',re.sub(r'[^\w\s]',' ',str(x).lower()))).strip()
paths=list((root/'Mem-T-main/traj').glob('*/qa_trajectories.jsonl'))
paths+=list((root/'Code/Exp-A').glob('traj/**/qa_trajectories.jsonl'))
paths+=list((root/'Code/Exp-A_Result').glob('*/sources/**/qa_trajectories.jsonl'))
found=[]; errors=[]; total=0
for p in paths:
    with p.open() as f:
        for line,s in enumerate(f,1):
            try:r=json.loads(s)
            except Exception:errors.append([str(p),line]);continue
            total+=1
            if r.get('sample_id') not in samples:continue
            gold=r.get('gold',''); pred=r.get('pred','')
            if not norm(pred) or norm(pred) not in [norm(x) for x in (gold if isinstance(gold,list) else [gold])]:continue
            traces=r.get('traces',[])
            ev=[]; conv=samples[r['sample_id']]['conversation']
            for eid in r.get('evidence',[]):
                short=eid.split('_')[-1]
                for key,turns in conv.items():
                    if isinstance(turns,list):
                        for t in turns:
                            if t.get('dia_id')==short:ev.append(dict(id=eid,date=conv.get(key+'_date_time'),turn=t))
            if not ev:continue
            found.append(dict(source=str(p),line=line,record=r,evidence_source=ev))
found.sort(key=lambda x:(('Exp-A_Result' in x['source'] or '/Code/Exp-A/' in x['source']),min(len(x['record'].get('traces',[])),6),len(x['evidence_source'])),reverse=True)
unique=[]; seen=set()
for x in found:
    k=x['record']['question']
    if k in seen:continue
    seen.add(k);unique.append(x)
(out/'candidates.json').write_text(json.dumps(unique,ensure_ascii=False,indent=2))
(out/'search_manifest.json').write_text(json.dumps(dict(files=[str(p) for p in paths],records=total,matched_records=len(found),unique_questions=len(unique),parse_errors=errors),indent=2))
for i,x in enumerate(unique[:30]):
    r=x['record'];print(json.dumps(dict(i=i,q=r['question'],a=r['pred'],steps=len(r.get('traces',[])),evidence=len(x['evidence_source']),source=x['source'],line=x['line']),ensure_ascii=False))
