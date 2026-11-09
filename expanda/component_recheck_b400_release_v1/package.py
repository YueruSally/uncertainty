"""Build a minimal, self-contained release from the completed 900-run experiment."""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent
EXPANDA=HERE.parent
ENGINE=EXPANDA/'formal_icm_s0_v1'
sys.path.insert(0,str(EXPANDA/'hv_instance_union_v1'))
import recompute as hv


def dependency_files(src):
    """Keep only the local import closure of the frozen execution modules."""
    modules={}
    for root in (src,src/'support'):
        for path in root.rglob('*.py'):
            if root==src and 'support' in path.relative_to(root).parts:continue
            parts=list(path.relative_to(root).with_suffix('').parts)
            if parts[-1]=='__init__':parts.pop()
            modules['.'.join(parts)]=path
    pending=['experiment'];seen=set()
    while pending:
        name=pending.pop()
        if name not in modules or name in seen:continue
        seen.add(name);path=modules[name]
        package=name if path.name=='__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=node.module or ''
                if node.level:
                    parent=package.split('.') if package else []
                    parent=parent[:len(parent)-node.level+1]
                    prefix='.'.join(parent+([prefix] if prefix else []))
                names=[prefix]+[prefix+'.'+a.name for a in node.names if a.name!='*']
            for target in names:
                bits=target.split('.')
                pending.extend('.'.join(bits[:i]) for i in range(1,len(bits)+1))
    return sorted({modules[n] for n in seen})


README='''B400 扩大验证：九组方法完整精简复现包

先看 results/RESULT_TABLE.csv；运行只用 run.py。
10 个 S0 实例 × 10 个共同种子 × 9 种方法 = 900 次搜索。
pop20，B400，100 个优化情景，q90，K8。三个历史模型冻结，不做 PPO。
七组单组件/Base，另含 ICM 启发式和 ICM Transformer。
ICM 启发式：Directed + Simple + Balanced。
ICM Transformer：冻结 I residual + C residual + M Transformer。

目录
  src/      I、C、M、组合、网络、基线与实际运行依赖；support 为依赖，不是附加实验。
  assets/   九个权重、十个实际实例、训练期归一化文件、运输数据、配置和校验记录。
  results/  九组总表、逐实例表、900 次逐运行 HV、共同界限及原始最终解集。
            raw_runs.jsonl.gz 将 2700 份运行/配置/前沿文件无损合并为一个文件。
  run.py    核验、重算 HV、重跑搜索的唯一入口。

环境（原服务器 Linux CPU）
  conda env create -f environment.yml
  conda activate icm-b400-release

核验和使用
  python run.py --verify
    验证文件哈希；从原始目标重算全部旧 HV 与新 HV；核对三张结果表；不启动搜索。
  python run.py --recompute
    重算现有 900 次结果，保存到 recomputed_hv；不覆盖打包结果。
  python -u run.py --run
    单进程重新执行完整 900 次搜索，输出 new_runs；全部完成后自动计算 new_runs_hv。
    不需要 Git、服务器原路径或其他仓库；模型路径在包内自动重定位。
    只有明确使用 --run 才运行搜索。再次执行可复用完成运行；保留中断目录，需另选输出后重跑。
  python run.py --run --instance recheck-b400-S0-001 --seed 961001 --method I_C_M_K8_HEURISTIC --out smoke
    只复现指定一组；不会当作完整九组报告。

HV 口径
每实例合并九种方法和十个种子的各自最终可行非支配前沿，不再全局过滤。
分别取原始 cost/emission/makespan 的 min/max，所有方法共用该实例界限。
常数维映射为零，参考点为 (1.1,1.1,1.1)，每次运行仍只计算自己的前沿。
相对 Base 及对应规则的提升逐实例、逐种子配对；实例等权；分层 bootstrap。
事后 HV 归一化与搜索使用的训练期归一化分开保存，不改变已冻结的模型和搜索。
百分比列不是平均 HV 的比值；wins/ties/losses 列相对 Base。

完整性范围
本包完整支持本轮冻结模型的搜索复现、最终解核验和 HV 统计重算。
不包含历史失败实验、训练数据与从零训练流程、tests、逐候选日志、Git 历史。
原始代码保留算法逻辑，打包只整理入口、路径和结果存储；已有原结果不被覆盖。
原环境版本记录在 assets/design.json。跨环境仅可用 --allow-environment-change 作移植检查，不能承诺数值逐位一致。
本轮是 S0/B400 开发验证，不代表更大预算或独立测试结论。
'''


def payload(out,source,design):
    old=Path(hv.read_json(source/'PLAN.json')['source_root'])
    code_hashes={}
    for path in dependency_files(ENGINE/'src'):
        target=out/'src'/path.relative_to(ENGINE/'src');target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);code_hashes[str(path)]=hv.digest(path)
    for path in (ENGINE/'src/data').glob('*.csv'):
        target=out/'src/data'/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(EXPANDA/'component_recheck_b400_icm_v1/controller.py',out/'src/combined_rule.py')
    shutil.copy2(HERE/'portable_run.py',out/'run.py')
    report=(EXPANDA/'hv_instance_union_v1/recompute.py').read_text()
    report=report.replace("SUPPORT = HERE.parent/'formal_icm_s0_v1/src/support'","SUPPORT = HERE/'support'")
    (out/'src/hv_report.py').write_text(report)
    for c in 'ICM':
        target=out/'assets/models'/c;target.mkdir(parents=True)
        bundle=hv.read_json(old/'models'/f'{c}_BUNDLE.json')
        hv.write_json(target/'BUNDLE.json',bundle)
        spec=bundle['models']['200'] if c=='M' else bundle
        if len(spec['checkpoints'])!=3:raise ValueError('Expected three checkpoints per component')
        for filename in spec['checkpoints']:
            path=Path(filename)
            if hv.digest(path)!=spec['checkpoint_sha256'][filename]:raise ValueError('Weight hash mismatch')
            shutil.copy2(path,target/path.name)
    for name in design['instances']:
        target=out/'assets/instances'/f'{name}.json';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(old/'instances'/f'{name}.json',target)
    shutil.copy2(old/'instances/catalog.json',out/'assets/instances/catalog.json')
    for path in (ENGINE/'assets/data').iterdir():
        if path.suffix in ('.xlsx','.csv'):
            target=out/'assets/data'/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(ENGINE/'assets/reward_normalization.json',out/'assets/reward_normalization.json')
    hv.write_json(out/'assets/design.json',design)
    (out/'README_CN.txt').write_text(README,encoding='utf-8')
    env=(ENGINE/'environment.yml').read_text().replace('name: icm-components','name: icm-b400-release')
    env=env.replace('python=3.10','python='+design['environment']['python'])
    (out/'environment.yml').write_text(env)
    return code_hashes


def build(source,destination):
    source=source.resolve();destination=destination.resolve()
    if destination.exists():raise FileExistsError(f'Preserve existing package: {destination}')
    plan,rows,inputs=hv.load_runs(source)
    if len(rows)!=900 or len(plan['methods'])!=9:raise ValueError('Requires completed nine-arm 900-run experiment')
    frozen=hv.read_json(source/'PLAN.json')
    for file,sha in frozen['runtime_files'].items():
        if hv.digest(Path(file))!=sha:raise ValueError(f'Frozen code/input changed: {file}')
    reported=source.parent/(source.name+'_hv_union_v1')
    manifest=hv.read_json(reported/'INPUT_MANIFEST.json')
    for file,sha in manifest['files'].items():
        if hv.digest(Path(file))!=sha:raise ValueError(f'Report input changed: {file}')
    old=Path(frozen['source_root']);old_plan=hv.read_json(old/'PLAN.json')
    design={k:plan[k] for k in ('instances','algorithm_seeds','methods','protocol')}
    design.update(environment=old_plan['environment'],settings=old_plan['config'],
                  code_version='422227e frozen nine-arm execution; portable packaging only',
                  normalization='per-instance union of nine methods final fronts; ref=(1.1,1.1,1.1)')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='icm-b400-package-') as tmp:
        root=Path(tmp)/'ICM_B400_900_Compact';root.mkdir()
        code_hashes=payload(root,source,design)
        result=root/'results';result.mkdir()
        for name in ('RESULT_TABLE.csv','PER_INSTANCE.csv','PER_RUN.csv','INSTANCE_BOUNDS.json','QUICK_RESULT.json'):
            shutil.copy2(reported/name,result/name)
        with gzip.open(result/'raw_runs.jsonl.gz','wt',encoding='utf-8') as f:
            for n in plan['instances']:
                for s in plan['algorithm_seeds']:
                    for m in plan['methods']:
                        base=old if m in plan['reused_methods'] else source
                        folder=base/'runs'/n/f'seed_{s}'/m
                        record={k:hv.read_json(folder/file) for k,file in (
                            ('run','RUN_COMPLETE.json'),('configuration','configuration.json'),('front','final_feasible_nondominated.json'))}
                        f.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
        with gzip.open(root/'assets/provenance.json.gz','wt',encoding='utf-8') as f:
            json.dump(dict(original_plan=old_plan,extension_plan=frozen,hv_inputs=manifest,
                           source_code_sha256=code_hashes,packager_sha256=hv.digest(Path(__file__))),f,separators=(',',':'),allow_nan=False)
        files={str(p.relative_to(root)):hv.digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
        hv.write_json(root/'assets/manifest.json',dict(files=files,run_count=900,
            scope='Frozen inference, all nine arms, complete final fronts and HV replay; no training dataset or old diagnostics.'))
        checked=subprocess.run([sys.executable,str(root/'run.py'),'--verify'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if checked.returncode:raise RuntimeError('Portable verification failed:\n'+checked.stdout[-8000:])
        print(checked.stdout.splitlines()[-1],flush=True)
        # runtime_inputs only contains regenerated paths; do not ship temporary paths.
        shutil.rmtree(root/'runtime_inputs',ignore_errors=True)
        stage=destination.with_name(destination.name+'.building')
        with zipfile.ZipFile(stage,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(root.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(root.parent)))
        stage.replace(destination)
    print(f'[PACKAGE] COMPLETE {destination} ({destination.stat().st_size/1024**2:.2f} MiB); no search or training.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=EXPANDA/'transformer_ppo_runs/diagnostics/ccp_component_recheck_b400_icm_v1')
    p.add_argument('--out',type=Path,default=Path.home()/'ICM_B400_900_Compact.zip')
    a=p.parse_args();build(a.source,a.out)
