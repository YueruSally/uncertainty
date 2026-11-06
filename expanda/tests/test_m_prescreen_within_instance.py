import unittest

from check_m_prescreen_within_instance import temporal_split


class WithinInstanceScreenTests(unittest.TestCase):
    def test_each_trajectory_is_split_in_time_without_crossing_instances(self):
        rows=[]
        for instance in ('a','b'):
            for seed in (1,2):
                for event in range(10):
                    rows.append({'instance':instance,'trajectory_seed':seed,
                        'generation':event//2,'evaluation':event,
                        'event':event})
        early,late,details=temporal_split(rows,'a')
        self.assertEqual(len(early),10)
        self.assertEqual(len(late),10)
        self.assertEqual({row['instance'] for row in early+late},{'a'})
        self.assertEqual(len(details),2)
        for seed in (1,2):
            self.assertLess(max(row['event'] for row in early
                                if row['trajectory_seed']==seed),
                            min(row['event'] for row in late
                                if row['trajectory_seed']==seed))

    def test_short_trajectory_is_rejected(self):
        rows=[{'instance':'a','trajectory_seed':1,'generation':0,
               'evaluation':event,'event':event} for event in range(6)]
        with self.assertRaisesRegex(ValueError,'too short'):
            temporal_split(rows,'a')


if __name__=='__main__':
    unittest.main()
