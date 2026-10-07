"""Compare numerical fields with reference values in data/paper_tables.json."""
from pathlib import Path
import re,json,csv
import numpy as np
PACKAGE=Path(__file__).resolve().parents[1]

def num(s):return float(re.search(r'\d+(?:\.\d+)?',str(s)).group())

def main():
    tables=json.loads((PACKAGE/'data/paper_tables.json').read_text());records=[]
    for idx,tag,field in [(2,'q1','T_C'),(3,'q1','C'),(4,'q2','T_C'),(5,'q2','C'),(6,'q3','C'),(8,'q4_uniform','C')]:
        a=np.load(PACKAGE/'results/raw'/f'{tag}.npz')
        for row in tables[idx]['rows'][1:]:
            seconds=a['history'][-1,0] if '结束' in row[0] else num(row[0])*(1 if idx in [2,3] else 3600)
            pos=int(np.argmin(abs(a['history'][:,0]-seconds)))
            r=a['r_m'][pos]
            targets=[0,.005,.01,r[-1]] if idx==8 else [0,.005,.01,.015,.02]
            values=np.interp(targets,r,a[field][pos]).tolist()
            if idx==8:values.append(r[-1]*100)
            for col,(p,v) in enumerate(zip(row[1:],values),1):
                expected=num(p)
                records.append({'section':tables[idx]['section'],'time':row[0].replace('*',''),'position':tables[idx]['rows'][0][col],
                    'paper':expected,'computed':v,'difference':v-expected,'within_rounding':bool(abs(v-expected)<=.000050001)})
    with (PACKAGE/'results/consistency_check.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    status={'all_values_match':all(r['within_rounding'] for r in records),
            'compared_values':len(records),'matching_values':sum(r['within_rounding'] for r in records),
            'note':'Numerical results are compared with the reference table dataset; per-value differences are recorded in consistency_check.csv.'}
    (PACKAGE/'results/consistency_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
