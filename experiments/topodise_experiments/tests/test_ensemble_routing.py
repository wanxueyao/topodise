import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.topodise_experiments.batch_experiments import (
    _aggregate_patient_logit_ensembles,
    _fold_calibrated_entropy_choice,
)


def _candidate(name, pred, probs):
    return {'candidate': name, 'y_pred': pred, 'probs': probs}


def test_fold_calibration_removes_candidate_confidence_offset():
    candidates = [
        _candidate('U0', 0, [0.80, 0.10, 0.10]),
        _candidate('U1', 1, [0.40, 0.50, 0.10]),
    ]

    chosen, centered = _fold_calibrated_entropy_choice(
        candidates,
        {'U0': 0.10, 'U1': 1.00},
    )

    assert chosen['candidate'] == 'U1'
    assert centered['U1'] < centered['U0']


def test_stage_aggregation_only_emits_fold_calibrated_entropy():
    children = []
    for name, probs in [('U0', [0.80, 0.10, 0.10]),
                        ('U1', [0.10, 0.80, 0.10])]:
        children.append({
            '_group': name,
            'stage2_results': {
                'pd_stage': {
                    'fold_results': [{
                        'fold': 1,
                        'best_patient_acc': 1.0,
                        'best_patient_predictions': [{
                            'patient_id': 'P1',
                            'y_true': 0,
                            'y_pred': 0 if name == 'U0' else 1,
                            'patient_probs': probs,
                        }],
                    }],
                },
            },
        })

    result = _aggregate_patient_logit_ensembles(children, 'pd_stage')

    assert set(result['predictions_by_method']) == {
        'fold_calibrated_entropy_routing'
    }
    assert [row['method'] for row in result['methods']] == [
        'fold_calibrated_entropy_routing'
    ]
