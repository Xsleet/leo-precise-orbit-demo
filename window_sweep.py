"""2019 八个固定窗口：无预热，比较动力学拟合长度与 TimesFM 输入长度。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
from main import ROOT, MODEL_ID, REVISION, load_evaluator
from short_2019 import DATA_PATH
from no_warmup import FIT_HOURS, CONTEXT_HOURS
from no_warmup import history, future_reference, prepare, correct, prefit, save_arc, fingerprint
from plotting import write_csv
from compare_windows import summarize


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--model',choices=['physics','timesfm'],default='timesfm')
    parser.add_argument('--checkpoint',default=MODEL_ID)
    parser.add_argument('--fit-hours',type=int,nargs='+',choices=FIT_HOURS,default=list(FIT_HOURS))
    parser.add_argument('--context-hours',type=int,nargs='+',choices=CONTEXT_HOURS,default=list(CONTEXT_HOURS))
    parser.add_argument('--workers',type=int,choices=range(1,17),default=1)
    parser.add_argument('--threads',type=int,default=4,help='TimesFM 使用的 CPU 线程数')
    parser.add_argument('--refit',action='store_true')
    parser.add_argument('--output',type=Path,help='自定义输出目录')
    args = parser.parse_args()
    args.fit_hours = sorted(set(args.fit_hours))
    args.context_hours = sorted(set(args.context_hours))
    if args.model=='physics' and args.device!='cpu':
        parser.error('动力学使用 CPU；GPU 用于 TimesFM')
    if args.threads < 1:
        parser.error('线程数必须为正')
    contexts_by_fit = {
        hours: [context for context in args.context_hours if context <= hours]
        for hours in args.fit_hours
    }
    if args.model == 'timesfm' and not all(contexts_by_fit.values()):
        parser.error('每个拟合长度至少需要一种不超过它的 TimesFM 输入长度')
    folder = args.output or ROOT/'outputs'/f'window-sweep-{args.model}-{args.device}'
    folder.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    timings = []
    if args.workers>1:
        jobs=[(n,h,args.refit) for h in sorted(args.fit_hours,reverse=True) for n in range(1,9)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(prefit,job):job for job in jobs}
            for task in as_completed(futures):
                result=task.result(); timings.append(result)
                print('动力学完成',json.dumps(result),flush=True)
                (folder/'prefit.json').write_text(json.dumps(timings,indent=2)+'\n',encoding='utf8')
    data = np.loadtxt(DATA_PATH,delimiter=',',skiprows=1)
    if args.model=='timesfm':
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    started_model = time.perf_counter()
    evaluator = load_evaluator(args.device,args.checkpoint) if args.model=='timesfm' else None
    load_seconds = time.perf_counter()-started_model
    all_errors, by_window, predictions, baselines = {}, [], [], []
    for hours in args.fit_hours:
        for number in range(1,9):
            print(f'拟合 {hours} h，窗口 {number}/8',flush=True)
            past = history(data,number,hours)
            arc = prepare(past,recompute=args.refit and args.workers==1)
            save_arc(folder/f'fit-{hours:02d}h-window-{number:02d}',arc)
            forecasts = {0:arc['baseline']}
            if evaluator is not None:
                for context in contexts_by_fit[hours]:
                    forecasts[context]=correct(arc,evaluator,context)
            # 所有预测已完成，才读取未来参考并评分。
            reference = future_reference(data,number)
            for context,xyz in forecasts.items():
                key=f'fit{hours:02d}_context{context:02d}'
                error=np.linalg.norm(xyz-reference[:,1:4],axis=1)
                all_errors.setdefault(key,[]).append(error)
                by_window.append(dict(fit_hours=hours,context_hours=context,window=number,
                    rms_3d_m=float(np.sqrt(np.mean(error**2))),endpoint_3d_m=float(error[-1])))
                target=baselines if context==0 else predictions
                for lead in range(60):
                    row=dict(fit_hours=hours,context_hours=context,window=number,lead_minutes=lead+1,
                             gps_seconds=reference[lead,0],error_3d_m=error[lead])
                    row.update({f'reference_{c}_m':reference[lead,i+1] for i,c in enumerate('xyz')})
                    row.update({f'predicted_{c}_m':xyz[lead,i] for i,c in enumerate('xyz')})
                    target.append(row)
            print('  '+', '.join(f"输入 {r['context_hours']} h: {r['endpoint_3d_m']:.4f} m"
                  for r in by_window[-len(forecasts):]),flush=True)
    summary,leads=summarize(all_errors)
    for row in summary:
        h,c=map(int,[row['model'][3:5],row['model'][-2:]])
        row.update(fit_hours=h,context_hours=c)
    write_csv(folder/'summary.csv',summary)
    write_csv(folder/'by_window.csv',by_window)
    write_csv(folder/'by_lead.csv',leads)
    write_csv(folder/'baseline.csv',baselines)
    if predictions:write_csv(folder/'prediction.csv',predictions)
    sources=['window_sweep.py','no_warmup.py','main.py','short_2019.py','precise.py','dynamics_cache.py',
             'rdop.py','orbit.py','gravity.py','compare_windows.py','plotting.py']
    metadata=dict(device=args.device,model=args.model,fit_hours=args.fit_hours,context_hours=args.context_hours,
        contexts_by_fit=contexts_by_fit if evaluator is not None else {},
        warmup_minutes=0,horizon_minutes=60,windows=8,fit_step_seconds=10,model_step_seconds=60,
        selection='Fixed 2019-01-02 00:00 through 07:00 GPS; all eight windows retained',
        source_sha256=fingerprint(sources),data_sha256=fingerprint(['data/orbit-2019.csv']),
        force_manifest_sha256=fingerprint(['data/forces/manifest.json']),
        python=platform.python_version(),platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','orekit-jpype','JPype1','jdk4py']},
        elapsed_seconds=time.perf_counter()-started,model_load_seconds=load_seconds,
        model_id=MODEL_ID,revision=REVISION,fine_tuned=False,per_core_batch_size=1)
    if evaluator is not None:
        import torch
        metadata.update(torch=torch.__version__,torch_threads=torch.get_num_threads(),
                        torch_interop_threads=torch.get_num_interop_threads(),
                        gpu=torch.cuda.get_device_name(0) if args.device=='gpu' else None,
                        timesfm=importlib.metadata.version('timesfm'))
    (folder/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)
    print('结果：',folder,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8');sys.stderr.reconfigure(encoding='utf8')
    run()
