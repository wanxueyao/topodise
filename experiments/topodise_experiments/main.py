"""Main entry and fixed patient-level entropy routing and majority voting."""
from __future__ import annotations
import math


def fixed_patient_ensemble(indices, names, method):
    """Combine patient predictions by minimum-entropy routing or majority vote."""
    if method not in ('min_entropy_routing', 'majority_vote_tie_avg_prob'):
        raise ValueError(f'Unsupported fixed report method: {method}')
    if len(names) != len(set(names)) or len(indices) != len(names):
        raise ValueError('Unique aligned candidate names are required')
    rows = []
    for fold, patient in sorted(indices[0]):
        values = sorted([(name, index[(fold, patient)]) for name, index in zip(names, indices)])
        if method == 'min_entropy_routing':
            # Select the candidate with the smallest predictive entropy.
            entropy = lambda p: -sum(float(x)*math.log(max(float(x),1e-12)) for x in p)
            name, selected = min(values, key=lambda v: (entropy(v[1]['patient_probs']),v[0]))
            probs = selected['patient_probs']
            pred = selected['y_pred']
            extra = {'selected_candidate': name, 'patient_entropy': entropy(probs),
                     'uses_labels_for_routing': False}
        else:
            probs = [0.0] * len(values[0][1]['patient_probs'])
            for _, value in values:
                for c, probability in enumerate(value['patient_probs']):
                    probs[c] += float(probability)
            probs = [p / float(len(values)) for p in probs]
            votes = {}
            # Each candidate contributes one predicted class label.
            for _,v in values:
                label = int(v['y_pred']); votes[label] = votes.get(label,0)+1
            best = max(votes.values()); tied = [label for label in votes if votes[label]==best]
            pred = max(tied,key=lambda label:(probs[label],-label))
            extra = {'votes': votes}
        rows.append({'fold': fold,'patient_id': patient,'y_true': int(values[0][1]['y_true']),
                     'y_pred': int(pred),'ensemble_probs': list(probs),'method': method,**extra})
    return rows


def main(argv=None):
    from topodise.training.runtime.batch_experiments import main as run
    return run(argv)

if __name__ == '__main__':
    main()
