"""Regression checks on real ablation DAGs; no training or datasets required."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ablation_experiments.common.planning import build_plan, execution_slice
from ablation_experiments.common.registry import TASKS, select


class MatchedReferencePlanningTests(unittest.TestCase):
    seeds = (42, 43, 44)

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def plan(self, names, unit='CANDIDATE'):
        return build_plan(select(names), self.root / 'output', self.root / 'data',
                          seeds=self.seeds, unit=unit)

    def views(self, plan, variant):
        return {(v['pretrain_seed'], v['task']): v for v in plan['views']
                if v['variant_id'] == variant}

    def test_matched_references_exclude_dm_and_keep_pd(self):
        plan = self.plan(['C03', 'C04', 'C05'])
        jobs = {job['id']: job for job in plan['jobs']}
        for variant in ('C03.default', 'C04.default', 'C05.zip', 'C05.cartesian',
                        'M0.paired', 'M0.matched_pairs'):
            views = self.views(plan, variant)
            for seed in self.seeds:
                with self.subTest(variant=variant, seed=seed):
                    dm = views[seed, 'dm_binary']
                    self.assertEqual(dm['status'], 'not_applicable')
                    self.assertFalse(dm.get('members'))
                    for task in TASKS[:-1]:
                        pd = views[seed, task]
                        self.assertEqual(pd['status'], 'planned')
                        for member in pd['members']:
                            cfg = jobs[member['job_id']]['config']
                            option = ('matched_pair_budget' if variant in
                                      ('C05.zip', 'C05.cartesian', 'M0.matched_pairs')
                                      else 'paired_cohort')
                            self.assertIs(cfg['extensions'][option], True)
        self.assertFalse(any(j['stage'] == 'stage2' and
                             j['config']['task'] == 'dm_binary'
                             for j in plan['jobs']))

    def test_shared_paired_views_are_identical_and_order_independent(self):
        separate = [self.views(self.plan([name]), 'M0.paired')
                    for name in ('C03', 'C04')]
        self.assertEqual(separate[0], separate[1])
        for names in (['C03', 'C04'], ['C04', 'C03']):
            plan = self.plan(names)
            self.assertEqual(self.views(plan, 'M0.paired'), separate[0])
            self.assertEqual(sum(v['variant_id'] == 'M0.paired'
                                 for v in plan['views']), len(self.seeds) * len(TASKS))

    def test_ordinary_reference_does_not_inherit_candidate_restrictions(self):
        # D02 is inactive where fusion is already frozen, but M0 is still needed.
        plan = self.plan(['D02', 'C03', 'C04'])
        baseline = self.plan(['M0'])
        self.assertTrue(any(v['status'] == 'not_applicable'
                            for v in self.views(plan, 'D02.default').values()))
        self.assertEqual(self.views(plan, 'M0.default'),
                         self.views(baseline, 'M0.default'))
        self.assertEqual(len(self.views(plan, 'M0.default')),
                         len(self.seeds) * len(TASKS))
        for job in baseline['jobs']:
            self.assertIn(job, plan['jobs'])

    def test_execution_slice_does_not_restore_invalid_dm_jobs(self):
        plan = execution_slice(self.plan(['M0', 'C03', 'C04']), ['C03'])
        self.assertEqual({v['variant_id'] for v in plan['views']},
                         {'C03.default', 'M0.paired'})
        self.assertFalse(any(j['stage'] == 'stage2' and
                             j['config']['task'] == 'dm_binary'
                             for j in plan['jobs']))
        ids = {j['id'] for j in plan['jobs']}
        for job in plan['jobs']:
            self.assertTrue(set(job['dependencies']) <= ids)

    def test_final_keeps_inactive_votes_without_cohort_intervention(self):
        # FINAL intentionally retains unchanged candidates as ensemble members.
        plan = self.plan(['M0', 'C03', 'C04'], unit='FINAL')
        jobs = {job['id']: job for job in plan['jobs']}
        baseline = self.views(plan, 'M0.default')
        for variant in ('M0.paired', 'C03.default', 'C04.default'):
            for seed in self.seeds:
                view = self.views(plan, variant)[seed, 'dm_binary']
                expected = baseline[seed, 'dm_binary']['members']
                self.assertEqual([m['job_id'] for m in view['members']],
                                 [m['job_id'] for m in expected])
                for member in view['members']:
                    self.assertFalse(member['intervened'])
                    self.assertFalse(jobs[member['job_id']]['config']
                                     .get('extensions', {}).get('paired_cohort'))


if __name__ == '__main__':
    unittest.main()
