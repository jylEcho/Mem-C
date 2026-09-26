import json, hashlib
from pathlib import Path
p=Path('./BestCases_20260916')
data=json.loads((p/'candidates.json').read_text())
questions=[('Cross-session attribute aggregation','What instruments does Tim play?'),('Resolving a relative date','When did Maria donate her car?'),('Aggregating travel locations','What states has Maria vacationed at?')]
selected=[next(x for x in data if x['record']['question']==q) for _,q in questions]
for x in selected:
    x['source_sha256_at_export']=hashlib.sha256(Path(x['source']).read_bytes()).hexdigest()
(p/'selected_cases.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
def esc(s):
    return ''.join({'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','\\':r'\textbackslash{}'}.get(c,c) for c in str(s))
tex=[r'% Requires \usepackage[most]{tcolorbox}',r'\section{Qualitative Case Studies}',r'\label{app:qualitative_cases}',r'These selectively chosen successful examples illustrate memory-agent behavior, not aggregate superiority or a causal effect of credit allocation. Evidence below is verified against the source dataset; it is not a verbatim retrieval response. The saved QA traces contain tool calls but do not preserve the full retrieved passages. Model outputs and reference answers are reproduced without correction.']
notes=['The answer combines piano and violin mentions from sessions 8 and 21. Three factual-memory searches precede an explicit finish action. This illustrates cross-session aggregation, not proof that additional searches were necessary.', 'The source session is dated 22 December 2022 and describes the donation as yesterday, consistent with the returned date of 21 December 2022. One factual-memory search precedes an explicit finish action.', 'The source evidence names Oregon and Florida in separate sessions. Two factual-memory searches precede an explicit finish action. This illustrates set-valued answer aggregation, not measured search efficiency.']
md=['# Verified qualitative candidates','', 'The uploaded reference is a prompt/tool-definition appendix, not a best-case result. These examples instead show recorded successful QA behavior. They do not establish Q/H/U scaling, Agentic superiority, or SOTA.','']
for idx,((title,_),x,note) in enumerate(zip(questions,selected,notes),1):
    r=x['record']
    tex += [r'\begin{tcolorbox}[title={Case '+str(idx)+': '+title+r'},colback=white,colframe=black!60,breakable]',r'\small',r'\textbf{Question.} '+esc(r['question'])+r'\par',r'\textbf{Source evidence (dataset excerpts).}\par']
    for e in x['evidence_source']:
        tex += [r'\textit{'+esc(e['turn']['dia_id']+'; '+str(e['date']))+r'}: '+esc(e['turn']['text'])+r'\par']
    tex += [r'\textbf{Recorded tool sequence.}',r'\begin{enumerate}']
    for t in r['traces']:
        call=t.get('tool_call',{});args=call.get('args',{})
        tex += [r'\item \texttt{'+esc(call.get('name',''))+r'}: '+esc(args.get('query',args.get('answer','')))]
    tex += [r'\end{enumerate}',r'\textbf{Model answer.} '+esc(r['pred'])+r'\par',r'\textbf{Reference answer.} '+esc(r['gold'])+r'\par',r'\textbf{Interpretation.} '+esc(note)+r'\par',r'\textbf{Record ID.} \texttt{'+esc(r['qa_id'])+r'}.',r'\end{tcolorbox}']
    md += [f'## {idx}. {title}',f'- Question: {r["question"]}',f'- Prediction: {r["pred"]}',f'- Gold: {r["gold"]}',f'- QA ID: {r["qa_id"]}',f'- Source: `{x["source"]}`',f'- JSONL line: {x["line"]}',f'- SHA256: `{x["source_sha256_at_export"]}`',f'- Interpretation: {note}','']
md += ['## Excluded despite a matching final answer','- Jolene/Susie/Seraphim: evidence references include a purchase in Paris and a music-related utterance, which do not establish the claimed pet adoption chronology. Do not use without resolving source annotation/context.','- Maria/Coco/Shadow: the trajectory contains contradictory and incorrect intermediate date calculations before a fallback answer matches the reference. Not a clean reasoning-success case.','- Dave/bands: correct final answer, but six searches and max-step fallback; suitable for a qualified recovery case, not an efficiency best case.','','## Limits','All selected examples are from Exp-A run directories. Directory names are provenance, not independent verification of checkpoint loading. These QA logs lack complete returned observation text; dataset evidence is explicitly labeled, not represented as an observed retrieval response. No successful memory update or training-branch causal effect is inferred from these examples.']
(p/'case_studies.tex').write_text('\n\n'.join(tex))
(p/'README.md').write_text('\n'.join(md))
manifest=json.loads((p/'search_manifest.json').read_text())
print({k:v for k,v in manifest.items() if k!='files'})
print('Selected',len(selected),'cases; files saved to',p)
