"""Read-only inventory; writes only the audit report under Mem-T."""
import os,json,re,sqlite3
from pathlib import Path
from datetime import datetime
root=Path('./external')
out=root/'Mem-T'/('hotpot_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir()
errors=[];logs=[];dbs=[];qa=[];names=[];links=[];total=0
for directory,dirs,files in os.walk(root,onerror=lambda e:errors.append(str(e)),followlinks=False):
    for d in dirs:
        p=Path(directory)/d
        if p.is_symlink():links.append(str(p))
    for name in files:
        total+=1;p=Path(directory)/name
        if out in p.parents:continue
        hot='hotpot' in str(p).lower()
        if 'hotpot' in name.lower() or 'ideav5' in name.lower():names.append(str(p))
        try:
            if name=='chroma.sqlite3':
                c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,timeout=2)
                cols=[x[0] for x in c.execute('select name from collections')]
                dbs.append(dict(path=str(p),collections=len(cols),names=cols,
                    embeddings=c.execute('select count(*) from embeddings').fetchone()[0]))
                c.close()
            if name=='qa_trajectories.jsonl' and hot:
                n=0;ids=set();bad=0
                with p.open() as f:
                    for line in f:
                        try:
                            d=json.loads(line);n+=1;ids.add(str(d.get('sample_id')))
                        except Exception:bad+=1
                qa.append(dict(path=str(p),rows=n,ids=sorted(ids),bad=bad))
            if p.suffix in ('.log','.out','.txt','.md','.sh') and p.stat().st_size<200_000_000:
                # Inspect headers of generically named logs too, not only Hotpot filenames.
                with p.open(errors='replace') as f:head=f.read(32768)
                if not hot and not re.search(r'hotpotqa|idea.?v5',head,re.I):continue
                t=p.read_text(errors='replace')
                if 'dataset=hotpotqa' in t or 'db_path=' in t or 'hotpot' in name.lower():
                    done=re.findall(r'sample_done\s+(\d+)\s+(\S+)',t)
                    progress=re.findall(r'Sample (\S+):[^\r\n]*',t)
                    logs.append(dict(path=str(p),headers=re.findall(r'(?:dataset=|dataset_path=|db_path=|traj_dir=)[^\r\n]*',t)[:8],
                        done_count=len(done),done_ids=sorted(set(x[1] for x in done)),
                        elapsed=re.findall(r'elapsed_sec=[^\r\n]*',t)[-1:],
                        last_build=re.findall(r'building \d+ \S+',t)[-1:]))
        except Exception as e:errors.append(str(p)+': '+str(e))
r=dict(root=str(root),files_enumerated=total,errors=errors,directory_symlinks_not_followed=links,
       matching_names=names,databases=dbs,logs=logs,qa=qa)
(out/'inventory.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print('REPORT',out,'FILES',total,'ERRORS',len(errors),'DBS',len(dbs),'LOGS',len(logs),'QA',len(qa))
for d in dbs:
    if 'hotpot' in d['path'].lower():print('DB',d['path'],d['collections'],d['embeddings'])
for q in qa:print('QA',q['path'],q['rows'],len(q['ids']))
for l in logs:
    if l['done_count'] or l['elapsed']:print('LOG',l['path'],l['done_count'],l['elapsed'])
print('ERRORS',errors[:10])
