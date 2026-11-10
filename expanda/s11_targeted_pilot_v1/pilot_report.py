"""Direction-only pilot reports; no promotion or automatic expansion."""
from common import *
from report import report as full_report,comparison,csv_write
import numpy as np

def report(raw_rows,settings,out):
    q=full_report(raw_rows,settings,out)
    rows=read_json(out/'PAIRED_RUNS.json');direction={};table=[]
    for component in 'ICM':
        by_instance=[]
        for name in sorted({r['instance'] for r in rows}):
            panel=[r for r in rows if r['instance']==name]
            rule=comparison(panel,component+'_RULE','BASE')
            learn=comparison(panel,component+'_LEARN',component+'_RULE')
            metric='absolute' if rule['reference_zero_pairs'] or learn['reference_zero_pairs'] else 'percent'
            by_instance.append({'instance':name,'metric':metric,'rule_vs_base':rule[metric]['mean'],
                'learn_vs_rule':learn[metric]['mean'],
                'both_positive':rule[metric]['mean']>0 and learn[metric]['mean']>0})
        direction[component]={'mean_direction_positive':q['acceptance'][component]['both_point_estimates_positive'],
            'instances_both_positive':sum(x['both_positive'] for x in by_instance),'per_instance':by_instance}
        rule=q['methods'][component+'_RULE'];learn=q['methods'][component+'_LEARN']
        table.append({'组件':component,'规则相对Base(%)':rule['vs_base']['percent']['mean'],
            'Transformer相对Base(%)':learn['vs_base']['percent']['mean'],
            'Transformer相对规则(%)':learn['vs_rule']['percent']['mean'],
            '规则平均秒':rule['mean_seconds'],'Transformer平均秒':learn['mean_seconds'],
            '两项均改善的实例数':direction[component]['instances_both_positive'],
            '筛查实例数':len(by_instance)})
    q['exploratory_interval_condition']=q.pop('all_components_met_reliable_gain')
    q.update(pilot_only=True,pilot_direction=direction,automatic_expansion_started=False,
        qualification='Directional S11 pilot, only two held-out development instances and one algorithm seed each. Bootstrap intervals are exploratory and cannot establish stable superiority. Limited training data also prevents a negative pilot from ruling out the approach.',
        next_action='Stop here. Review the fixed pilot results before deciding whether to expand; no new searches or retraining start automatically.')
    csv_write(out/'00_小试方向及耗时.csv',table);write_json(out/'QUICK_RESULT.json',q)
    return q
