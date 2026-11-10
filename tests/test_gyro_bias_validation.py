"""Selection and audit guards, independent of calibration candidate results."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bias_validation", ROOT / "scripts/validate_gyro_bias.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(rotations, ate=1., position=2.):
    return {"summary": {"mean_episode_rotation_rmse_rad": sum(rotations) / len(rotations),
                        "mean_episode_sim3_ate_m": ate, "mean_episode_unaligned_position_rmse_m": position},
            "episodes": [{"episode_id": i, "metrics": {"rotation_rmse": x}} for i, x in enumerate(rotations)]}


class GyroBiasValidationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "configs/gyro-bias-v4.json").read_text())
        self.variants = {v["name"]: candidate([1., 1., 1., 1.]) for v in self.protocol["variants"]}

    def test_selection_requires_material_rotation_both_position_guards_and_wins(self):
        self.variants['gyro-only'] = candidate([.7] * 4, ate=1.01)
        self.variants['calibrated-residual'] = candidate([.7] * 4, position=2.01)
        self.variants['calibrated-gyro'] = candidate([.3, .3, 1.1, 1.1], ate=.7, position=1.)
        result = MODULE.select_variant(self.variants, self.protocol)
        self.assertEqual(result['selected_variant'], 'current')
        self.assertFalse(result['candidate_eligibility']['calibrated-gyro']['conditions']['rotation_win_count'])
        self.variants['calibrated-gyro'] = candidate([.81] * 4, ate=.7, position=1.)
        self.assertEqual(MODULE.select_variant(self.variants, self.protocol)['selected_variant'], 'current')

    def test_selection_minimizes_ate_then_rotation_then_protocol_order(self):
        self.variants['gyro-only'] = candidate([.6] * 4, ate=.9)
        self.variants['calibrated-residual'] = candidate([.7] * 4, ate=.8)
        self.variants['calibrated-gyro'] = candidate([.75] * 4, ate=.8)
        self.assertEqual(MODULE.select_variant(self.variants, self.protocol)['selected_variant'], 'calibrated-residual')
        self.variants['calibrated-gyro'] = candidate([.7] * 4, ate=.8)
        self.assertEqual(MODULE.select_variant(self.variants, self.protocol)['selected_variant'], 'calibrated-residual')

    def test_recording_predictor_retains_raw_inputs_without_mutation(self):
        class Fake:
            meta, size = {}, (4, 3)
            def predict(self, pair, state): return np.arange(6, dtype=np.float32)
        recorder = MODULE.RecordingPredictor(Fake())
        pair, state = np.ones((2, 3, 4), np.float32), np.arange(14, dtype=np.float32)
        recorder.predict(pair, state)
        np.testing.assert_array_equal(state, recorder.states[0])
        self.assertEqual(recorder.pair_hashes[0], MODULE.array_digest(pair))
        state[0] = 100
        self.assertEqual(recorder.states[0][0], 0)

    def test_repeat_and_raw_prediction_guards_reject_changes(self):
        record = {'states': np.zeros((2, 14)), 'predictions': np.zeros((2, 6)),
                  'image_pair_sha256': np.array(['a', 'b']), 'local_translation': np.zeros((2, 3)),
                  'outputs': np.zeros((3, 13)), 'diagnostics': [{'status': 'disabled'}]}
        MODULE.invariant_checks(record, copy.deepcopy(record), repeat=True)
        for key in ['states', 'predictions', 'local_translation', 'outputs']:
            changed = copy.deepcopy(record); changed[key].flat[0] = 1
            with self.assertRaises(ValueError): MODULE.invariant_checks(record, changed, repeat=True)

    def test_causal_prefix_includes_deadline_output_and_rejection_is_exact(self):
        base = {'outputs': np.zeros((4, 13))}
        calibrated = {'outputs': base['outputs'].copy(), 'diagnostics': [{'status': 'accepted', 'activation_after_t': 1.0}]}
        calibrated['outputs'][3, 0] = 1
        result = MODULE.verify_causal_prefix(calibrated, base, np.array([.9, .95, 1., 1.05]))
        self.assertEqual(result['first_permitted_corrected_frame'], 3)
        calibrated['outputs'][2, 0] = 1
        with self.assertRaises(ValueError): MODULE.verify_causal_prefix(calibrated, base, np.array([.9, .95, 1., 1.05]))
        calibrated['diagnostics'] = [{'status': 'rejected'}]
        with self.assertRaises(ValueError): MODULE.verify_causal_prefix(calibrated, base, np.array([.9, .95, 1., 1.05]))


if __name__ == '__main__':
    unittest.main()
