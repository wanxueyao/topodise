"""Keep the public paper presets aligned with the reported comparisons."""
from contextlib import redirect_stdout
from io import StringIO
import unittest

from ablation_experiments.common.registry import select
from ablation_experiments.paper_targets import main as target_plan
from topodise.training.runtime.batch_experiments import main as main_plan


class PaperPresets(unittest.TestCase):
    def test_target_grid_uses_full_pd_protocol(self):
        with redirect_stdout(StringIO()):
            grid = target_plan(['--dry-run'])
            main = main_plan(['--dry-run'])
        self.assertEqual([job['variant'] for job in main['stage2']], ['gt_hgrad_dual'])
        tasks = ['pd_binary', 'pd_stage', 'pd_subtype_tdrest']
        self.assertEqual([r['task'] for r in main['stage2'][0]['config']['recipes']], tasks + ['dm_binary'])
        expected = {(s0, s1) for s0 in ('gt', 'hgrad_dual', 'frangi') for s1 in ('frangi', 'hgrad_dual')}
        self.assertEqual({job['config']['topo_target'] for job in grid['stage0']}, {pair[0] for pair in expected})
        self.assertEqual(len(grid['stage1']), len(expected))
        for job in grid['stage1']:
            cfg = job['config']
            s0 = cfg['run_name'].removesuffix('_' + cfg['topo_target'])
            self.assertIn((s0, cfg['topo_target']), expected)
            self.assertIn('/stage0/' + s0 + '/', cfg['parent_ckpt'])
        self.assertEqual({job['variant'] for job in grid['stage2']}, {f'{a}_{b}' for a, b in expected})
        for job in grid['stage2']:
            recipes = job['config']['recipes']
            self.assertEqual([r['task'] for r in recipes], tasks)
            self.assertEqual([len(r['candidates']) for r in recipes], [1, 3, 5])

    def test_candidate_preset_is_limited_to_reported_interventions(self):
        ids = [v['variant_id'] for v in select(['paper'])]
        expected = [f'{i}.default' for i in
                    'M0 A01 A02 A03 A04 A05 C03 C04 D01 D05 D07 D10 D15 D16 E01'.split()]
        self.assertCountEqual(ids, expected + ['A06.gt_on', 'A06.gt_off'])


if __name__ == '__main__':
    unittest.main()
