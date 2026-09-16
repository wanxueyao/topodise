"""All research-specific hooks are owned by the ablation package."""
from pathlib import Path
from topodise.training.hooks import TrainingHooks


class AblationHooks(TrainingHooks):
    def __init__(self):
        self.full_data_len=None
        self.label_budget=None
        self.feature_stats={}

    def build_external_encoder(self,cfg,device):
        from ..G01_G05_encoder_comparison.encoders.stage2_bridge import build_stage2_encoder
        return build_stage2_encoder(cfg,device)

    def build_encoder(self,config,metadata):
        from ..C01_C05_encoder_architecture.architecture import build
        return build(config,metadata)

    def target(self,function,cfg,**kwargs):
        from ..B01_B06_topology_targets.targets import target
        return target(function,cfg,**kwargs)

    def topology_loss(self,logits,target,cfg):
        from ..B01_B06_topology_targets.targets import loss
        return loss(logits,target,cfg)

    def configure_pretrain(self,encoder,unet,head,cfg,stage):
        options=cfg.extensions
        if options.get('initial_checkpoint'):
            import torch
            saved=torch.load(options['initial_checkpoint'],map_location='cpu',weights_only=False)
            for model,key in ((encoder,'encoder'),(unet,'unet'),(head,'topo_head')): model.load_state_dict(saved[key],strict=True)
        if stage==1:
            if options.get('reset_topology_head'):
                for module in head.modules():
                    if hasattr(module,'reset_parameters'): module.reset_parameters()
            if options.get('unfreeze_encoder'):
                encoder.requires_grad_(True)
            head.requires_grad_(bool(options.get('train_topology_head',False)) and cfg.lambda_topo>0)
        if stage==0 and cfg.lambda_simple==0: unet.requires_grad_(False)
        if cfg.lambda_topo==0: head.requires_grad_(False)

    def dataset(self,ds,cfg):
        if hasattr(ds,'pairs'):
            self.full_data_len=len(ds)
            fraction=cfg.extensions.get('unlabeled_fraction',1.)
            if fraction<1:
                from ..I01_I09_sensitivity.subsets import unlabeled
                ds=unlabeled(ds,fraction,cfg.seed)
                if sum(r['is_paired'] for r in ds.pairs)<cfg.batch_size and cfg.lambda_region>0:
                    raise ValueError('I07 subset has too few valid pairs for the declared region objective')
            return ds
        from ..C01_C05_encoder_architecture.architecture import dataset
        return dataset(ds,cfg)

    def feature_diagnostics(self, raw, reference, gated, labels, cfg, scope):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            cosine=F.cosine_similarity(raw.detach(),reference.detach(),dim=-1)
            threshold=cfg.fdiff_gate_threshold
            if cfg.fdiff_gate_mode=='none': trust=torch.ones_like(cosine)
            elif cfg.fdiff_gate_mode=='soft_cos': trust=torch.sigmoid((cosine-threshold)/max(cfg.fdiff_gate_temperature,1e-6))
            else: trust=(cosine>threshold).float()
            before=(raw.detach()-reference.detach()).norm(dim=-1)
            after=(gated.detach()-reference.detach()).norm(dim=-1)
            for label in labels.unique():
                mask=labels==label;key=f'{scope}.class{int(label)}';entry=self.feature_stats.setdefault(key,dict(count=0,cosine_sum=0.,trust_sum=0.,displacement_before_sum=0.,displacement_after_sum=0.))
                entry['count']+=int(mask.sum());entry['cosine_sum']+=float(cosine[mask].sum());entry['trust_sum']+=float(trust[mask].sum());entry['displacement_before_sum']+=float(before[mask].sum());entry['displacement_after_sum']+=float(after[mask].sum())

    def feature_head(self,head,cfg):
        from ..E01_E09_feature_diffusion.generators import build
        return build(head,cfg)

    def sampling_budget(self,bags,cfg,default):
        from ..D01_D18_downstream_components.policies import balanced_slots
        if cfg.extensions.get('matched_label_budget') and self.label_budget is not None: return self.label_budget
        return balanced_slots(bags) if cfg.extensions.get('matched_sampling_budget') else default

    def bag_order(self,bags,cfg,rng,default):
        if cfg.extensions.get('matched_label_budget') and self.label_budget is not None:
            pool=default if cfg.mil_balanced_bags else bags
            return [rng.choice(pool) for _ in range(self.label_budget)]
        if cfg.extensions.get('matched_sampling_budget'):
            return [rng.choice(bags) for _ in range(self.sampling_budget(bags,cfg,len(bags)))]
        return default

    def morphology_sources(self,train_bags,val_bags,cfg,rng,epoch):
        from ..D01_D18_downstream_components.policies import morphology_sources
        return morphology_sources(train_bags,val_bags,cfg,epoch)

    def fold_bags(self,train_bags,val_bags,cfg,fold):
        from ..D01_D18_downstream_components.policies import balanced_slots
        self.label_budget=balanced_slots(train_bags) if cfg.mil_balanced_bags else len(train_bags)
        fraction=cfg.extensions.get('label_fraction',1.)
        if fraction<1:
            from ..I01_I09_sensitivity.subsets import labeled
            train_bags=labeled(train_bags,fraction,cfg.seed+fold)
        return train_bags,val_bags

    def record_epoch(self,cfg,fold,epoch,predictions,directory):
        from topodise.training.runtime.protocol import write
        if self.feature_stats:
            write(Path(directory)/f'fold{fold}_epoch{epoch}_feature_diagnostics.json',self.feature_stats)
            self.feature_stats={}
        if cfg.extensions.get('record_trajectory'):
            from topodise.training.runtime.protocol import write
            write(Path(directory)/'trajectory'/f'fold{fold}_epoch{epoch}.json',dict(fold=fold,epoch=epoch,predictions=predictions))

    def instance_loss(self,z,morph,head,y,cfg,class_weights):
        from ..D01_D18_downstream_components.policies import instance_loss
        return instance_loss(z,morph,head,y,cfg,class_weights)

    def split_indices(self, dataset, cfg):
        return cfg.extensions.get('explicit_splits')

    def batches(self,loader,cfg):
        steps=cfg.extensions.get('steps_per_epoch')
        if cfg.extensions.get('steps_reference_manifest'):
            from topodise.data import RegionPairDataset
            reference=RegionPairDataset(manifest_path=cfg.extensions['steps_reference_manifest'],base_path=cfg.extensions['reference_data_root'])
            steps=len(reference)//cfg.batch_size
        if cfg.extensions.get('data_budget')=='fixed_steps' and self.full_data_len is not None:
            steps=self.full_data_len//cfg.batch_size
        if steps is None: return loader
        if len(loader)==0: raise ValueError('No complete pretraining batch')
        class FixedBatches:
            def __len__(self): return steps
            def __iter__(self):
                iterator=iter(loader)
                for _ in range(steps):
                    try: value=next(iterator)
                    except StopIteration: iterator=iter(loader);value=next(iterator)
                    yield value
        return FixedBatches()
