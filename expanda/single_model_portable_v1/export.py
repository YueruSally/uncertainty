#!/usr/bin/env python3
"""Build a relocatable, frozen single-member runtime from the completed server run."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
VERSION = '05b66cac5b52830b83e6f23ea27e71d626be8cea'
PACK = 'ICM_SINGLE_B5000_RUNNABLE'


def read(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    value = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    diag = EXP/'transformer_ppo_runs/diagnostics'
    parser.add_argument('--root', type=Path, default=diag/'ccp_icm_single_member_panel_b5000_v1')
    parser.add_argument('--shared-root', type=Path, default=diag/'ccp_icm_shared_single_991001_b5000_v1')
    parser.add_argument('--out', type=Path, default=diag/(PACK+'.zip'))
    parser.add_argument('--with-wheels', action='store_true', help='Also download all Python dependency wheels for this Linux/Python platform')
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise SystemExit('Run this exporter on server 41 with its experiment Python environment.')
    if args.out.exists():
        raise SystemExit('Output exists; use a new --out filename: '+str(args.out))
    plan = read(args.root/'PLAN.json')
    if not read(args.root/'STATUS.json').get('completed'):
        raise ValueError('The single-member panel must have completed')
    settings = plan['settings']
    if (settings['population'],settings['budget'],settings['scenario_count'],settings['pool_size']) != (250,5000,100,8):
        raise ValueError('Wrong experiment settings')
    if plan['seed'] != 991001 or plan['ppo_update'] != 1:
        raise ValueError('Wrong member/update')
    if (settings['screen_seeds'] != [993001,993002,993003]
            or settings['milestones'] != [400,1000,5000]
            or sorted(e['configuration'] for e in plan['entries']) != ['S'+str(i) for i in range(9)]):
        raise ValueError('Wrong development panel')
    repo = EXP.parent
    def git(*cmd):
        return subprocess.check_output(['git','-C',str(repo),*cmd])
    def blob(name):
        return git('show', VERSION+':'+name)
    # Validate source entrypoints and use their historical dependency version.
    for name in ('icm_single_member_b5000.py','icm_single_member_panel_b5000.py'):
        expected = plan['source_code'][str(EXP/name)]
        if hashlib.sha256(blob('expanda/'+name)).hexdigest() != expected:
            raise ValueError('Historical single-member entrypoint differs: '+name)
    print('[EXPORT] frozen single-member plan verified', flush=True)
    original = read(diag/'ccp_component_joint_b5000_v1/PLAN.json')
    if original['sha256'] != plan['source_plan']:
        raise ValueError('Original plan mismatch')
    with tempfile.TemporaryDirectory(prefix='single-package-') as temp:
        pkg = Path(temp)/PACK
        pkg.mkdir()
        dirs = tuple('expanda/'+x+'/' for x in (
            'component_joint_b5000_v1','component_screen_b20_v1','joint_icm_multiscale_v1',
            'joint_icm_multiscale_terminal_v1','ppo_joint_b5000_pilot_v2','ppo_joint_b5000_screen27_v1',
            'ppo_joint_b5000_base_report_v1','hv_joint_b5000_union_v1'))
        for name in git('ls-tree','-r','--name-only',VERSION,'expanda').decode().splitlines():
            p = Path(name)
            if any('test' in bit.lower() or bit == '__pycache__' for bit in p.parts) or p.name == 'checks.py':
                continue
            selected = (name in ('expanda/icm_single_member_b5000.py','expanda/icm_single_member_panel_b5000.py')
                        or name.startswith(dirs) and p.suffix in ('.py','.json')
                        or name.startswith('expanda/formal_icm_s0_v1/src/') and p.suffix in ('.py','.csv'))
            if selected:
                content = blob(name)
                frozen = original['files'].get(str(repo/name))
                if frozen is not None and hashlib.sha256(content).hexdigest() != frozen:
                    raise ValueError('Historical runtime differs from frozen run: '+name)
                if p.suffix == '.py': ast.parse(content.decode('utf-8-sig'), filename=name)
                dest = pkg/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(content)
        data_rel = 'formal_icm_s0_v1/assets/data/data_expanded.xlsx'
        workbook = EXP/data_rel
        if sha(workbook) != original['files'][str(workbook)]:
            raise ValueError('Transport workbook changed')
        dst = pkg/'expanda'/data_rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(workbook,dst)
        shared = args.shared_root/'one_member'
        shared_plan = read(args.shared_root/'PLAN.json')
        if shared_plan['training_seed'] != 991001 or shared_plan['original_plan'] != plan['source_plan']:
            raise ValueError('Shared single-member provenance differs')
        sources = dict(shared=shared, independent=Path(plan['models']['independent']), ppo=Path(plan['models']['ppo']))
        expected = dict(shared=shared_plan['single_model'], **plan['hashes'])
        for kind, folder in sources.items():
            b = read(folder/'BUNDLE.json')
            if sha(folder/'BUNDLE.json') != expected[kind] or len(b['checkpoints']) != 1:
                raise ValueError('Wrong single-member bundle: '+kind)
            if b['normalization'] != plan['normalization']:
                raise ValueError('Model normalization differs: '+kind)
            name = b['checkpoints'][0]
            if Path(name).name != name or sha(folder/name) != b['checkpoint_sha256'][name]:
                raise ValueError('Wrong model weight: '+kind)
            dst = pkg/'models'/kind
            dst.mkdir(parents=True)
            for filename in ('BUNDLE.json',name): shutil.copyfile(folder/filename,dst/filename)
            print('[EXPORT]',kind,name,'SHA256 verified',flush=True)
        for entry in plan['entries']:
            saved = diag/'ccp_component_joint_b5000_v1/instances'/(entry['instance_id']+'.json')
            if read(saved) != entry: raise ValueError('Instance changed: '+str(saved))
            target = pkg/'assets/instances'/saved.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(saved,target)
        portable = dict(settings=settings, entries=plan['entries'], scenarios=plan['scenarios'],
                        normalization=plan['normalization'], training_seed=991001, ppo_update=1,
                        original_single_plan_sha=plan['sha256'], source_commit=VERSION)
        dump(pkg/'assets/PLAN.json',portable)
        shutil.copyfile(HERE/'run.py',pkg/'run.py')
        subprocess.run([sys.executable,'-c',
            'import run,sys,json; s=run.runtime(); p=json.load(open(sys.argv[1])); '
            'assert s.canonical({k:v for k,v in p.items() if k!="sha256"})==p["sha256"], "Frozen plan digest differs"; '
            'q=json.load(open(sys.argv[2])); assert s.canonical(q)==p["shared_single_plan"], "Shared plan digest differs"',
            str(args.root/'PLAN.json'), str(args.shared_root/'PLAN.json')],cwd=pkg,check=True)
        (pkg/'results').mkdir()
        shutil.copyfile(HERE/'provided_results.csv',pkg/'results/单个模型平均提升.csv')
        packages = ('numpy','pandas','torch','matplotlib','openpyxl')
        versions = {name:importlib.metadata.version(name) for name in packages}
        (pkg/'requirements.txt').write_text('\n'.join(k+'=='+v for k,v in versions.items())+'\n')
        dump(pkg/'ENVIRONMENT.json',dict(python=sys.version,platform=platform.platform(),packages=versions))
        if args.with_wheels:
            wheelhouse = pkg/'wheelhouse'
            wheelhouse.mkdir()
            cmd = [sys.executable,'-m','pip','download','--only-binary=:all:','--dest',str(wheelhouse),
                   '-r',str(pkg/'requirements.txt')]
            if '+cpu' in versions['torch']: cmd += ['--extra-index-url','https://download.pytorch.org/whl/cpu']
            print('[EXPORT] downloading dependency wheels; no installation or experiment',flush=True)
            subprocess.run(cmd,check=True)
        install = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
if [ -d wheelhouse ]; then
  .venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.txt
else
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python run.py --check
'''
        (pkg/'install.sh').write_text(install)
        (pkg/'run.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -u run.py "$@"
else
  exec python3 -u run.py "$@"
fi
''')
        (pkg/'README_CN.md').write_text('''# 单模型可运行包

这是已训练权重的单成员部署包，不重新训练，不读取 41 上的历史目录。
含 I/C/M 独立网络的一份成员、共享监督的一份成员、PPO u1 的对应成员；三个部署类别各一个 checkpoint。
独立 ICM 内部仍有三个任务网络，不等于三任务共享网络，也不使用三个随机种子成员做集成。

## 运行

Linux CPU，Python 小版本应与 ENVIRONMENT.json 相同。已有同版本依赖可直接：
```bash
bash run.sh --check
bash run.sh --workers 2
```
否则先 `bash install.sh`；附有 wheelhouse 时离线安装，否则联网安装锁定版本。
`--check` 只校验文件和加载模型，不执行搜索。
`bash run.sh --quick --workers 2` 只跑首个实例的首个种子和全部 11 种方法，不能作为完整 27 对的结果。
默认完整运行：S0–S8 各一个实例 × 三个算法种子 × 11 种方法 = 297 次搜索。
pop250，固定 CCP100，预算上限 5000，在 400/1000/5000 保存前沿；不设固定 200 代终止。

## 结果

results/单个模型平均提升.csv 是用户本次提供的表格，原样保存，未由本包重新计算或核验。
该表部分值与此前逐次记录不同，不能将其视为本程序重跑结果或复现保证。
新运行统计另存 computed_full/COMPUTED_RESULTS.csv；只在 5000 节点导出胜平负，不导出400/1000胜负明细表。
程序保留新搜索的原始前沿记录以便复核，不覆盖提供的表格。统计以同实例、共同种子配对；所有方法/种子/节点联合确定 min/max，参考点(1.1,1.1,1.1)。
未捆绑历史运行明细或测试文件。标准第三方依赖以原始 wheel 分发，未修改其内部内容。

## 来源

算法依赖来自历史单成员面板提交 05b66ca。run.py 是可移植运行入口：改变路径、调度及报告形式，不改变候选选择、目标评价和算法预算。
模型、数据与实例导出时核对原冻结摘要。MANIFEST.json 记录随包文件摘要。
训练源代码作为模型来源参考保留；本包只复现单成员推理搜索，未附训练标签或训练轨迹。
''',encoding='utf-8')
        for name in ('install.sh','run.sh'): (pkg/name).chmod(0o755)
        files = {str(p.relative_to(pkg)):sha(p) for p in pkg.rglob('*') if p.is_file()}
        dump(pkg/'MANIFEST.json',dict(files=files,source_commit=VERSION,single_member=True,
                                     training_updates=0,provided_summary_verified=False))
        # Load the exact frozen weights with the bundled runtime before releasing an archive.
        subprocess.run([sys.executable,str(pkg/'run.py'),'--check'],cwd=pkg,check=True)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        tmp = args.out.with_suffix('.zip.tmp')
        try:
            with ZipFile(tmp,'w',ZIP_DEFLATED,compresslevel=6) as z:
                for p in sorted(pkg.rglob('*')):
                    if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts:
                        z.write(p,PACK+'/'+str(p.relative_to(pkg)))
            with ZipFile(tmp) as z:
                if z.testzip() is not None: raise ValueError('Corrupt archive')
            tmp.replace(args.out)
        finally:
            if tmp.exists(): tmp.unlink()
    print('[EXPORT COMPLETE]',args.out,flush=True)
    print('No training/searches; source summary preserved; early W/T/L not included.',flush=True)


if __name__ == '__main__':
    main()
