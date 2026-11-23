"""Numbered ablation entry: plan by default; --execute explicitly starts training."""
from pathlib import Path
import argparse
from collections import Counter
from .common.registry import HOME,TASKS,select
from .common.planning import build_plan, execution_slice
from topodise.training.runtime.protocol import write,lock


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',default=['all'],
                        help='paper for the published component/probe training plan, all for the full registry, or individual IDs')
    parser.add_argument('--include-supplementary',action='store_true')
    parser.add_argument('--only',nargs='+',help='Execute only these groups/IDs/variants and their references/dependencies within the unchanged master plan')
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    parser.add_argument('--unit',choices=['CANDIDATE','FINAL'],default='CANDIDATE')
    parser.add_argument('--pretrain-seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--dataset-root',type=Path,default=HOME.parent/'datasets')
    parser.add_argument('--output',type=Path,default=HOME/'results/ablation_v2')
    parser.add_argument('--devices',nargs='+',default=['auto'])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--write-plan',type=Path)
    args=parser.parse_args(argv)
    if args.dry_run and args.execute: parser.error('--dry-run and --execute conflict')
    if len(set(args.devices))!=len(args.devices) or ('auto' in args.devices and len(args.devices)>1): parser.error('Use unique explicit devices or one auto device')
    specs=select(args.experiments,args.include_supplementary)
    plan=build_plan(specs,args.output,args.dataset_root,args.pretrain_seeds,args.tasks,args.repeats,args.unit)
    plan['output']=str(args.output.resolve())
    try:
        selected=execution_slice(plan,args.only)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Variants: {len(specs)}; jobs: {dict(Counter(j["stage"] for j in plan["jobs"]))}; views: {len(plan["views"])}')
    if args.only: print(f'Selected jobs (including references/parents): {dict(Counter(j["stage"] for j in selected["jobs"]))}')
    for entry in selected['analyses']: print(f'  {entry["spec"]["variant_id"]}: {entry["kind"]} ({entry["status"]})')
    if args.write_plan: write(args.write_plan,plan)
    if not args.execute: return plan
    from .common.execution import execute,source_hashes
    plan['source_sha256']=source_hashes()
    lock(args.output/'protocol.json',plan)
    for job in plan['jobs']: lock(args.output/'configs'/f'{job["id"]}.json',job)
    execute(selected,args.devices)
    from .common.reporting import collect
    collect(args.output)
    return plan

if __name__=='__main__': main()
