"""Check aggregate units, denominators, missing evidence and source preservation."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from step_gain import summarize,analyze


def fixture(count=2):
    return [{'pool_source':'validation','component':c,'instance':str(i),
             'candidate_pools':10 if i==0 else 30,'different_choices':2 if i==0 else 3,
             'recorded_one_step_mean_hv_difference':.02 if i==0 else -.01} for c in 'ICM' for i in range(count)]


class Checks(unittest.TestCase):
    def test_instance_weighting_changed_only_denominator_and_training_exclusion(self):
        rows=fixture();rows+=[{'pool_source':'training','component':'I','recorded_one_step_mean_hv_difference':None}]
        summary,detail=summarize(rows,{c:2 for c in 'ICM'},bootstrap=1000)
        for r in summary:
            self.assertAlmostEqual(r['实例等权平均单步HV差'],.005)
            self.assertAlmostEqual(r['状态等权平均单步HV差'],-.0025)
            self.assertAlmostEqual(r['仅改选状态的平均单步HV差'],-.02)
            self.assertEqual((r['实例均值为正'],r['实例均值为负']),(1,1))
            self.assertGreaterEqual(r['实例重采样95区间下限'],-.01)
        self.assertEqual(len(detail),6)

    def test_zero_changes_and_tiny_real_signs(self):
        rows=fixture()
        for r in rows:r.update(different_choices=0,recorded_one_step_mean_hv_difference=0.)
        summary,_=summarize(rows,{c:2 for c in 'ICM'},bootstrap=100)
        self.assertTrue(all(r['仅改选状态的平均单步HV差'] is None for r in summary))
        rows[0].update(different_choices=1,recorded_one_step_mean_hv_difference=-1e-12)
        summary,_=summarize(rows,{c:2 for c in 'ICM'},bootstrap=100)
        self.assertEqual(summary[0]['实例均值为负'],1)

    def test_missing_duplicates_and_impossible_aggregates_fail(self):
        for edit in ('missing','duplicate','nolabel','nan','impossible'):
            rows=fixture()
            if edit=='missing':rows.pop()
            if edit=='duplicate':rows.append(copy.deepcopy(rows[0]))
            if edit=='nolabel':rows[0]['recorded_one_step_mean_hv_difference']=''
            if edit=='nan':rows[0]['recorded_one_step_mean_hv_difference']='nan'
            if edit=='impossible':rows[0]['different_choices']=0
            with self.subTest(edit=edit),self.assertRaises(ValueError):summarize(rows,{c:2 for c in 'ICM'},bootstrap=10)

    def test_completed_audit_coverage_and_read_only_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=fixture(20)
            with (root/'SAME_POOL_PER_TRAJECTORY.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            quick={'completed':True,'choice_replay':{'validation_pool_instances_per_component':{c:20 for c in 'ICM'},
                   'same_pool_summary':[{'候选来源':'validation','组件':c,'候选池数':580,'不同选择次数':59} for c in 'ICM']}}
            q=root/'QUICK_RESULT.json';q.write_text(json.dumps(quick))
            before={p.name:p.read_bytes() for p in root.iterdir()}
            summary,detail,metadata=analyze(root)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            self.assertEqual(len(detail),60);self.assertEqual(metadata['new_model_inference_calls'],0)
            quick['choice_replay']['same_pool_summary'][0]['不同选择次数']=60;q.write_text(json.dumps(quick))
            with self.assertRaisesRegex(ValueError,'coverage'):analyze(root)


if __name__=='__main__':unittest.main()
