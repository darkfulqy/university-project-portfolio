"""T0: observed-set coordinate normalization drift; see T0_PREREG.md."""
import hashlib
import json
from pathlib import Path
import numpy as np
from common import ROOT, SEED, layouts, jsonable

SIGMA = .02

def normalise(x):
    return (x-x.min(axis=0))/(np.ptp(x, axis=0)+1e-8)

def main():
    rng = np.random.default_rng(SEED)
    null_rng = np.random.default_rng(SEED+91)
    nulls = {}
    def null(n):
        if n not in nulls:
            e = null_rng.normal(0, SIGMA, (2048,n,3))
            nulls[n] = dict(linf_mean=float(abs(e).max(axis=(1,2)).mean()),
                            linf_q95=float(np.quantile(abs(e).max(axis=(1,2)),.95)),
                            mae_mean=float(abs(e).mean()),
                            mae_q95=float(np.quantile(abs(e).mean(axis=(1,2)),.95)))
        return nulls[n]
    def evaluate(x, keep):
        d = normalise(x[keep])-normalise(x)[keep]
        ref = null(len(keep))
        return dict(n_kept=len(keep), linf=float(abs(d).max()), mae=float(abs(d).mean()),
                    rms=float(np.sqrt(np.mean(d*d))), mean_l2=float(np.linalg.norm(d,axis=1).mean()),
                    linf_in_sigma=float(abs(d).max()/SIGMA), mae_in_sigma=float(abs(d).mean()/SIGMA),
                    exceeds_matched_jitter_linf_q95=bool(abs(d).max()>ref['linf_q95']),
                    fixed_frame_linf=0.)
    results = {}
    for name, item in layouts().items():
        if name=='ten_ten': continue
        x, names = item['xyz'], item['names']
        n = len(x)
        deletes = dict(frontal=np.flatnonzero(x[:,1]>=np.quantile(x[:,1],.75)),
                       temporal=np.flatnonzero((abs(x[:,0])>=np.quantile(abs(x[:,0]),.75)) & (x[:,2]<=np.median(x[:,2]))),
                       extrema=np.unique(np.r_[x.argmin(axis=0),x.argmax(axis=0)]))
        cases={}
        for label,deleted in deletes.items():
            keep=np.setdiff1d(np.arange(n),deleted)
            cases[label]=dict(deleted=[names[i] for i in deleted], **evaluate(x,keep))
        runs=[]
        for _ in range(256):
            keep=np.sort(rng.choice(n,(n+1)//2,replace=False))
            runs.append(dict(kept_indices=keep.tolist(),**evaluate(x,keep)))
        cases['random50'] = dict(n_trials=len(runs),
            summary={metric:{'mean':float(np.mean([r[metric] for r in runs])),
                             'median':float(np.median([r[metric] for r in runs])),
                             'q95':float(np.quantile([r[metric] for r in runs],.95))}
                     for metric in ('linf','mae','rms','mean_l2','linf_in_sigma','mae_in_sigma')},
            exceeds_jitter_fraction=float(np.mean([r['exceeds_matched_jitter_linf_q95'] for r in runs])),runs=runs)
        results[name] = cases
    medians={label:float(np.median([item[label]['linf'] for item in results.values()]))
             for label in ('frontal','temporal','extrema')}
    supports = sum(v>3*SIGMA for v in medians.values())>=2
    out=dict(seed=SEED, prereg_sha256=hashlib.sha256((ROOT/'T0_PREREG.md').read_bytes()).hexdigest(),
             sigma=SIGMA, expected_abs_single_jitter=float(SIGMA*np.sqrt(2/np.pi)),
             expected_abs_two_view_jitter_difference=float(SIGMA*2/np.sqrt(np.pi)),
             numerical_epsilon=1e-8, layouts=results, jitter_nulls=nulls,
             aggregate=dict(median_linf=medians,threshold=3*SIGMA,verdict='supports' if supports else 'inconclusive'),
             scope='Coordinate drift only, not neural embeddings or downstream accuracy.')
    (ROOT/'t0_results.json').write_text(json.dumps(jsonable(out),indent=2)+'\n')
    lines=['# T0：逐样本坐标归一化漂移','',
           '预注册见 T0_PREREG.md；全通道列表及真实模板坐标见 layouts.json。下表为 L∞ / mean absolute coordinate drift（均为归一化坐标单位）。随机列取256次中位数。', '',
           '|布局|额区|颞区|极值|随机50%|','|---|---:|---:|---:|---:|']
    for name,item in results.items():
        row=[f"{item[k]['linf']:.5f} / {item[k]['mae']:.5f}" for k in ('frontal','temporal','extrema')]
        r=item['random50']['summary']; row.append(f"{r['linf']['median']:.5f} / {r['mae']['median']:.5f}")
        lines.append('|'+name+'|'+'|'.join(row)+'|')
    lines.extend(['',f"预注册 median L∞：{medians}；门槛=.06，判定 **{out['aggregate']['verdict']}**。",
                  '',f"单视图抖动σ=.02，元素绝对值期望={out['expected_abs_single_jitter']:.5f}；两个独立抖动视图差的绝对值期望={out['expected_abs_two_view_jitter_difference']:.5f}。匹配残存电极数量的抖动L∞分布及95%阈值已逐项写入JSON。",
                  '', '固定全布局变换对相同未删坐标的漂移严格为0。本检验仅支持固定 head frame 消融的必要性；没有测任务收益，不能据此支持求积权重或代表已完成模型修复。',
                  '', '额/颞区域为预注册几何代理，TCP中点区域并不等同于端点电极删除。TCP19特指reader切片；19/20/22共享大量坐标，不是六个独立实验样本。'])
    (ROOT/'T0.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(out['aggregate'],indent=2))

if __name__=='__main__': main()
