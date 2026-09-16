"""Dispatch artifact-based experiments; each command validates its own inputs."""
import argparse,importlib

ENTRIES={
 'D17':'D01_D18_downstream_components.selection','H06':'H01_H06_structure_diagnostics.region_probe','I09':'I01_I09_sensitivity.costs',
 'D04.tuned':'D01_D18_downstream_components.tuning',
 'D11':'D01_D18_downstream_components.replay','D12':'D01_D18_downstream_components.replay','D13':'D01_D18_downstream_components.replay','E06':'D01_D18_downstream_components.replay',
 'F01':'F01_F03_ensemble_strategies.analyze','F02':'F01_F03_ensemble_strategies.analyze','F03':'F01_F03_ensemble_strategies.analyze',
 'G01':'G01_G05_encoder_comparison.run','G02':'G01_G05_encoder_comparison.linear_probe','G03':'G01_G05_encoder_comparison.extract','G04':'G01_G05_encoder_comparison.linear_probe','G05':'G01_G05_encoder_comparison.linear_probe',
 'H01':'H01_H06_structure_diagnostics.visualize','H02':'H01_H06_structure_diagnostics.probes','H03':'H01_H06_structure_diagnostics.probes','H04':'H01_H06_structure_diagnostics.probes','H05':'H01_H06_structure_diagnostics.compare',
}
DEFAULTS={'D11':['--mode','classifier'],'D12':['--mode','mean'],'D13':['--mode','support_topk'],'E06':['--mode','no_generated_support'],'G03':['--encoder','random_topodise'],'G04':['--mlp'],'H01':['--kind','targets'],'H02':['--kind','grid'],'H03':['--kind','descriptors'],'H04':['--kind','retrieval']}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,add_help=False);p.add_argument('--experiment',choices=ENTRIES,required=True);a,rest=p.parse_known_args(argv)
    return importlib.import_module('ablation_experiments.'+ENTRIES[a.experiment]).main(DEFAULTS.get(a.experiment,[])+rest)

if __name__=='__main__': main()
