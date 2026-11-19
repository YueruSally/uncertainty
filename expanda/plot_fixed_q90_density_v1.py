#!/usr/bin/env python3
"""Paper-style KDE figures from frozen q90 audit samples; no new evaluations."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np

EXP = Path(__file__).resolve().parent
DEFAULT_SOURCE = EXP/'transformer_ppo_runs/diagnostics/ccp_fixed_q90_s0_s11_v1'
CONFIGS = [f'S{i}' for i in range(12)]
OBJECTIVES = ('cost', 'emission', 'makespan')
XLABELS = ('Total cost (USD)', r'Total emissions (g CO$_2$)', 'Completion time (h)')
PANEL_LABELS = ('(a) Total cost', '(b) Total emissions', '(c) Completion time')
BLUE, ORANGE = '#4166ff', '#ff9800'
SETTINGS = dict(version='fixed-q90-density-v1', representative='smallest exact decision SHA256',
    fresh100_replicate=1, reference_size=1000, fresh_size=100,
    kernel='Gaussian', bandwidth='reference1000 sample SD (ddof=1) * 1000**(-1/5); same absolute bandwidth for both samples',
    grid_points=1024, extent='full observed range of both samples plus four bandwidths on either side',
    colors=[BLUE, ORANGE], fill_alpha=.23, line_width=1.4,
    new_searches=0, new_objective_evaluations=0, training_updates=0,
    source='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    tmp = path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def pin(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Plot inputs/settings changed; choose a new output directory: '+str(path))
    if not path.exists():
        write(path, value)


def verify(files):
    for name, sha in files.items():
        if digest(Path(name)) != sha:
            raise ValueError('Source file changed: '+name)


def load_samples(root, cfg):
    folder = root/cfg
    audit_plan = read(root/'PLAN.json')
    inputs_path, result_path = folder/'INPUTS.json', folder/'RESULT.json'
    inputs, result = read(inputs_path), read(result_path)
    if (inputs['sha256'] != canonical({k:v for k,v in inputs.items() if k!='sha256'})
            or not result.get('completed') or result['inputs_sha256'] != inputs['sha256']
            or inputs['configuration'] != cfg):
        raise ValueError('Incomplete or inconsistent source audit: '+cfg)
    decisions = inputs['decisions']
    if not decisions or decisions[0]['exact_decision_id'] != min(d['exact_decision_id'] for d in decisions):
        raise ValueError('Predeclared representative decision was changed')
    files = {str(p):digest(p) for p in (root/'PLAN.json', inputs_path, result_path)}
    samples, seeds = {}, {}
    for tag, n in [('reference_1000',1000), ('replicate_01_100',100)]:
        meta_path, array_path = folder/'packs'/(tag+'.json'), folder/'packs'/(tag+'.npz')
        meta = read(meta_path)
        files.update({str(p):digest(p) for p in (meta_path, array_path)})
        if (not meta.get('completed') or meta['identity']['tag'] != tag
                or meta['identity']['inputs_sha256'] != inputs['sha256']
                or meta['identity']['audit_plan_sha256'] != audit_plan['sha256']
                or meta['identity']['realizations'] != n or meta['data_sha256'] != files[str(array_path)]):
            raise ValueError('Completed sample pack identity failed: '+str(array_path))
        with np.load(array_path, allow_pickle=False) as pack:
            all_samples = pack['samples']
            if all_samples.shape != (len(decisions),3,n):
                raise ValueError('Unexpected array shape: '+str(array_path))
            value = all_samples[0].copy()
            quantiles = pack['quantiles'][0].copy()
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite original sample; no observations will be dropped')
        if not np.allclose(np.sort(value,axis=1)[:,math.ceil(.9*n)-1],quantiles,rtol=1e-10,atol=1e-8):
            raise ValueError('Empirical q90 identity failed')
        samples[tag] = value
        seeds[tag] = meta['identity']['scenario_seed']
    if seeds['reference_1000'] != inputs['new_seeds'][0] or seeds['replicate_01_100'] != inputs['new_seeds'][1]:
        raise ValueError('Wrong reference or first replicate selected')
    verify(files)
    return samples['reference_1000'], samples['replicate_01_100'], dict(
        configuration=cfg, instance=inputs['instance']['instance_id'],
        exact_decision_id=decisions[0]['exact_decision_id'], source_files=files,
        inputs_sha256=inputs['sha256'], scenario_seeds=seeds)


def gaussian_density(samples, grid, width):
    if width <= 0 or not np.isfinite(width):
        raise ValueError('Positive finite bandwidth required')
    z = (grid[:,None]-samples[None,:])/width
    return np.exp(-.5*z*z).mean(axis=1)/(width*math.sqrt(2*math.pi))


def curves(reference, small):
    width = float(np.std(reference,ddof=1)*len(reference)**(-.2))
    # A constant series has a point mass, not a continuous density. Never add
    # random noise or a visual bandwidth floor to manufacture a bell curve.
    if width <= 0 or not np.isfinite(width):
        return dict(bandwidth=None, degenerate=True, reference=reference, small=small)
    low, high = min(reference.min(),small.min()), max(reference.max(),small.max())
    grid = np.linspace(low-4*width,high+4*width,SETTINGS['grid_points'])
    return dict(bandwidth=width, degenerate=False, x=grid,
        reference_density=gaussian_density(reference,grid,width),
        small_density=gaussian_density(small,grid,width))


def draw_panel(ax, data, xlabel, legend=True):
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    if data['degenerate']:
        ax.plot(data['reference'],np.zeros(len(data['reference'])),'|',color=BLUE,label='Sample = 1000')
        ax.plot(data['small'],np.zeros(len(data['small'])),'|',color=ORANGE,label='Sample = 100')
        ax.text(.5,.5,'Constant reference sample\nKDE is not defined',ha='center',va='center',transform=ax.transAxes,fontsize=9)
        ax.set_ylim(-.1,1)
    else:
        x = data['x']
        for key,color,label in [('reference_density',BLUE,'Sample = 1000'),('small_density',ORANGE,'Sample = 100')]:
            ax.fill_between(x,data[key],color=color,alpha=SETTINGS['fill_alpha'],linewidth=0,zorder=1)
            ax.plot(x,data[key],color=color,lw=SETTINGS['line_width'],label=label,zorder=2)
        ax.set_xlim(x[0],x[-1]); ax.set_ylim(0,1.08*max(data['reference_density'].max(),data['small_density'].max()))
    ax.set(xlabel=xlabel,ylabel='Density')
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    for axis in (ax.xaxis,ax.yaxis):
        formatter = ScalarFormatter(useOffset=False, useMathText=False)
        formatter.set_powerlimits((-3,4)); axis.set_major_formatter(formatter)
    ax.tick_params(direction='out',length=3,width=.7,labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(.7); spine.set_color('#333333')
    if legend:
        ax.legend(loc='upper right',fontsize=8,frameon=True,facecolor='white',edgecolor='#cccccc',framealpha=.95)


def save_figure(fig, path):
    for suffix in ('png','pdf','svg'):
        fig.savefig(path.with_suffix('.'+suffix),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.08)


def configuration_figure(cfg, series, destination):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12.1,3.85))
    fig.subplots_adjust(left=.065,right=.99,bottom=.27,top=.86,wspace=.34)
    for j, ax in enumerate(axes):
        draw_panel(ax,series[j],XLABELS[j])
        ax.text(.5,-.34,PANEL_LABELS[j],transform=ax.transAxes,ha='center',va='top',fontsize=13,fontfamily='DejaVu Serif')
    fig.suptitle(cfg+' | Fixed transport plan',y=.98,fontsize=12)
    save_figure(fig,destination/(cfg+'_DENSITY_KDE')); plt.close(fig)


def overview_figure(series, destination):
    import matplotlib.pyplot as plt
    cols = min(3,len(series)); rows = math.ceil(len(series)/cols)
    fig, axes = plt.subplots(rows,cols,figsize=(4.05*cols,3.0*rows),squeeze=False)
    fig.subplots_adjust(left=.08,right=.985,bottom=.07,top=.95,hspace=.57,wspace=.36)
    for i,(cfg,data) in enumerate(series.items()):
        ax = axes.flat[i]
        draw_panel(ax,data,'Completion time (h)',legend=i==0)
        ax.set_title(f'({chr(97+i)}) {cfg}',fontsize=11,fontfamily='DejaVu Serif',pad=8)
    for ax in list(axes.flat)[len(series):]:
        ax.set_visible(False)
    fig.suptitle('Completion-time distributions: independent 1000 vs fresh 100',fontsize=13,y=.995)
    save_figure(fig,destination/'TIME_CONFIGURATIONS_KDE'); plt.close(fig)


def execute(args):
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root.with_name(root.name+'_density_figures_v1')
    if out==root or root in out.parents or out in root.parents:
        raise ValueError('Use a separate figure output directory; source audit remains unchanged')
    plan = read(root/'PLAN.json')
    if (plan.get('protocol',{}).get('version') != 'fixed-q90-s0-s11-v1'
            or canonical({k:v for k,v in plan.items() if k!='sha256'}) != plan['sha256']):
        raise ValueError('Expected the fixed q90 audit source')
    if len(args.configurations) != len(set(args.configurations)):
        raise ValueError('Duplicate configurations requested')
    out.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out/'.matplotlib')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
        'axes.grid':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    with (out/'.plot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pin(out/'PLOT_PLAN.json',dict(settings=SETTINGS,source_root=str(root),
            script_sha256=digest(Path(__file__)),source_plan_sha256=plan['sha256']))
        time_series = {}
        for cfg in args.configurations:
            big,small,meta = load_samples(root,cfg)
            series = [curves(big[j],small[j]) for j in range(3)]
            meta.update(settings=SETTINGS,bandwidth_by_objective={o:series[j]['bandwidth'] for j,o in enumerate(OBJECTIVES)},
                empirical_q90_reference={o:float(np.sort(big[j])[899]) for j,o in enumerate(OBJECTIVES)},
                empirical_q90_fresh100={o:float(np.sort(small[j])[89]) for j,o in enumerate(OBJECTIVES)})
            pin(out/(cfg+'_PROVENANCE.json'),meta)
            configuration_figure(cfg,series,out)
            tmp = out/(cfg+'_PLOTTED_DENSITIES.csv.writing')
            with tmp.open('w',encoding='utf-8-sig',newline='') as f:
                writer = csv.writer(f); writer.writerow(['objective','x_original_units','density_1000','density_100','common_bandwidth'])
                for j,obj in enumerate(OBJECTIVES):
                    d=series[j]
                    if not d['degenerate']:
                        writer.writerows((obj,float(x),float(y),float(z),d['bandwidth']) for x,y,z in zip(d['x'],d['reference_density'],d['small_density']))
            tmp.replace(out/(cfg+'_PLOTTED_DENSITIES.csv'))
            verify(meta['source_files']); time_series[cfg] = series[2]
            caption = (f'{cfg}. Distributions of total cost, total emissions and completion time for one fixed transport plan. '
                'The plan is the predeclared smallest exact decision hash from the original audit. '
                'Independent sets of 1,000 realizations and the first fresh set of 100 realizations are shown. '
                'Gaussian kernel density estimates use the same absolute bandwidth for both sample sets within each objective; '
                'bandwidth equals the sample standard deviation of the 1,000-realization reference multiplied by 1000^(-1/5). '
                'Shading denotes estimated probability density, not a confidence interval. '
                'Empirical q90 values and accuracy/ranking statistics are not recalculated from smoothed curves. '
                'This illustrative fixed-plan distribution comparison is not by itself evidence that 100 realizations suffice.\n')
            (out/(cfg+'_CAPTION.txt')).write_text(caption,encoding='utf-8')
            (out/(cfg+'_CAPTION_CN.txt')).write_text(
                f'{cfg}：同一固定运输方案在独立1000次实现与第1组新100次实现下的成本、排放和完成时间分布。\n'
                '方案沿用原精度核对预先指定的最小决策哈希；不按曲线重合程度挑选方案或100次实现组。\n'
                '每个目标的两条曲线采用相同绝对带宽的高斯核密度估计；阴影是密度面积，不是置信区间。\n'
                '图中的100和1000是随机实现数，不是搜索评价预算；原始样本、经验P90和误差统计均未修改。\n'
                '单方案分布图只作说明，不能单独证明100次实现已经足够。\n',encoding='utf-8')
            print('[Q90-KDE] saved',out/(cfg+'_DENSITY_KDE.png'),flush=True)
        overview_figure(time_series,out)
        write(out/'STATUS.json',dict(completed=True,configurations=args.configurations,new_searches=0,
            new_objective_evaluations=0,training_updates=0,output_directory=str(out)))
        print('[Q90-KDE] COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--configurations',nargs='+',choices=CONFIGS,default=CONFIGS)
    execute(parser.parse_args())
