"""TopoDiSE 实验实现：Stage0/1/2 衔接、恢复、分类评估和集成。

批量配置由 batch_experiments.py 传入；本模块不读取实验清单、不调用批量入口。
"""
from __future__ import annotations
import os
import sys
import json
import time
import copy
import glob
import hashlib
import traceback
import math
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TopoDiSEExperiment:
    """One experiment runtime. All settings are explicitly supplied by the batch controller."""

    def __init__(self, settings):
        for name, value in copy.deepcopy(settings).items():
            setattr(self, name, value)

    def _run_parallel(self, experiments, output_dir, gpus, concurrency_per_gpu=1):
        # 单组运行使用当前进程；批量控制器重载此方法分配 GPU。
        return [self.run_experiment(exp, output_dir) for exp in experiments]

    def _resolve_project_path(self, path: Optional[str]) -> Optional[str]:
        """相对路径按 codes 根目录解析，绝对路径保持不变。"""
        if path is None:
            return None
        p = str(path).strip()
        if not p:
            return p
        if os.path.isabs(p):
            return os.path.normpath(p)
        return os.path.normpath(os.path.join(self.PROJECT_ROOT, p))


    def _pretrain_key(self, stage0_ov: Dict[str, Any], stage1_ov: Dict[str, Any]) -> str:
        """计算 Stage0/Stage1 预训练配方的稳定短哈希，用于去重。"""
        payload = json.dumps({'s0': stage0_ov, 's1': stage1_ov}, sort_keys=True, default=str)
        return hashlib.md5(payload.encode('utf-8')).hexdigest()[:12]


    def _fallback_recipe(self, exp: Dict[str, Any]) -> Optional[tuple]:
        """返回实验需要自动生成的 Stage0/Stage1 配方。

        完整流程或显式指定检查点的实验返回 None。触发条件与
        run_experiment 的自动回退一致：只有缺少有效父检查点、未设阶段过滤且仅运行
        Stage2 的实验，才会参与共享预训练去重。
        """
        if self.STAGE_FILTER is not None:
            return None
        stages = list(exp.get('stages', [0, 1, 2]))
        if not (2 in stages and 0 not in stages and 1 not in stages):
            return None
        parent = exp.get('parent_ckpt_override') or self.EXTERNAL_PARENT_CKPT or None
        parent = self._resolve_project_path(parent) if parent else None
        if parent and os.path.exists(parent):
            return None
        s0 = exp.get('stage0_overrides') or dict(self.DEFAULT_STAGE0_OVERRIDES)
        s1 = exp.get('stage1_overrides') or dict(self.DEFAULT_STAGE1_OVERRIDES)
        return s0, s1


    def _apply_smoke_overrides(self, e: Dict[str, Any]):
        """将实验的每个阶段缩减为 1 轮的最小烟雾测试配置。"""
        for k in ('stage0_overrides', 'stage1_overrides', 'stage2_overrides'):
            ov = e.setdefault(k, {})
            ov['epochs'] = 1
            ov['batch_size'] = 2
            ov['image_size'] = 64
            ov['num_workers'] = 0
        e.setdefault('stage0_overrides', {}).update({
            'encoder_base_channels': 16, 'unet_model_channels': 16,
            'z_dim': 64, 'T': 50,
            'encoder_channel_mult': (1, 2, 2, 2),
            'encoder_blocks_per_stage': (1, 1, 1, 1),
            'unet_channel_mult': (1, 2, 2),
        })
        e.setdefault('stage2_overrides', {}).update({'adapter_rank': 8})
        for child in e.get('oracle_experiments') or []:
            self._apply_smoke_overrides(child)


    def _apply_overrides(self, cfg, overrides: Dict[str, Any]):
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"  [warn] unknown override '{k}' for {type(cfg).__name__}")
        return cfg


    def _stage_run_name(self, exp_group: str, stage: int, suffix: str = '') -> str:
        base = f"{exp_group}_stage{stage}"
        return base if not suffix else f"{base}_{suffix}"


    def _resolve_cfg_paths(self, cfg, fields: List[str]):
        for f in fields:
            if hasattr(cfg, f):
                v = getattr(cfg, f)
                if isinstance(v, str) and v.strip():
                    setattr(cfg, f, self._resolve_project_path(v))
        return cfg


    def _validate_paths(self, required: Dict[str, str]) -> Optional[str]:
        """必需路径缺失时返回错误字符串，否则返回 None。"""
        missing = []
        for name, p in required.items():
            if not p or not os.path.exists(p):
                missing.append(f"{name}={p}")
        if missing:
            return "missing required path(s): " + ", ".join(missing)
        return None


    def _record_stage_error(self, summary: Dict[str, Any], stage_key: str, exp_dir: str,
                            error: str, tb: str):
        summary['stage_results'][stage_key] = {'error': error, 'traceback': tb}
        print(f"[error][{summary['_group']}][{stage_key}] {error}")
        print(tb)
        self._save_summary(summary, exp_dir)


    def _load_existing_summary(self, exp_dir: str) -> Optional[Dict[str, Any]]:
        """读取该实验已保存的 summary.json；不存在时返回 None。"""
        p = os.path.join(exp_dir, 'summary.json')
        if not os.path.exists(p):
            return None
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None


    def _stage_ckpt_result(self, exp_dir: str, group: str, stage: int) -> Optional[Dict[str, Any]]:
        ckpt_path = os.path.join(exp_dir, self._stage_run_name(group, stage), 'ckpt', 'ema_best.pt')
        if not os.path.exists(ckpt_path):
            return None
        return {'ckpt_path': ckpt_path, 'recovered_from_ckpt': True}


    def _recover_pretrain_stages_from_ckpt(self, prev: Optional[Dict[str, Any]], exp_dir: str,
                                           group: str, stages: List[int]) -> Optional[Dict[str, Any]]:
        """根据标准检查点路径补全缺失的 Stage0/Stage1 摘要。

        即使 summary.json 未写入或只写入一部分，中断后重启的消融批次也不会
        重复训练已完成的预训练阶段。
        """
        if not any(s in stages for s in (0, 1)):
            return prev
        recovered = copy.deepcopy(prev) if isinstance(prev, dict) else {
            '_group': group, 'stage_results': {}, 'stage2_results': {},
        }
        stage_results = recovered.setdefault('stage_results', {})
        changed = False
        for stage in (0, 1):
            if stage not in stages:
                continue
            key = f'stage{stage}'
            existing = stage_results.get(key)
            if isinstance(existing, dict) and 'error' not in existing and existing.get('ckpt_path'):
                continue
            ckpt_result = self._stage_ckpt_result(exp_dir, group, stage)
            if ckpt_result is not None:
                stage_results[key] = ckpt_result
                changed = True
        if 0 in stages and 1 in stages and not self._stage_done(recovered, 'stage0'):
            # Stage1 只能与其训练时使用的 Stage0 父检查点一起复用。
            # 如果 Stage0 必须重跑，则不能接入过期的 Stage1 检查点。
            old_stage1 = stage_results.get('stage1')
            if isinstance(old_stage1, dict) and 'error' not in old_stage1:
                stage_results.pop('stage1', None)
                changed = True
        return recovered if changed or prev is not None else None


    def _candidate_group_summary_paths(self, group: str, output_dir: str) -> List[str]:
        roots = self.RESUME_SEARCH_DIRS or [self.OUTPUT_BASE_DIR]
        patterns: List[str] = []
        for root in roots:
            if not root:
                continue
            patterns.extend([
                os.path.join(root, group, 'summary.json'),
                os.path.join(root, '*', group, 'summary.json'),
            ])
        current = os.path.normpath(os.path.join(output_dir, group, 'summary.json'))
        seen = set()
        candidates: List[str] = []
        for pat in patterns:
            for path in glob.glob(pat):
                norm = os.path.normpath(path)
                if norm == current or norm in seen:
                    continue
                seen.add(norm)
                if os.path.isfile(norm):
                    candidates.append(norm)
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates


    def _candidate_group_dirs(self, group: str, output_dir: str) -> List[str]:
        roots = self.RESUME_SEARCH_DIRS or [self.OUTPUT_BASE_DIR]
        current = os.path.normpath(os.path.join(output_dir, group))
        patterns: List[str] = []
        for root in roots:
            if not root:
                continue
            patterns.extend([
                os.path.join(root, group),
                os.path.join(root, '*', group),
            ])
        seen = set()
        candidates: List[str] = []
        for pat in patterns:
            for path in glob.glob(pat):
                norm = os.path.normpath(path)
                if norm == current or norm in seen:
                    continue
                seen.add(norm)
                if os.path.isdir(norm):
                    candidates.append(norm)
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates


    def _load_resume_summary_from_search_dirs(self, group: str, output_dir: str) -> Optional[Dict[str, Any]]:
        """在历史批次或归档目录中查找同组摘要。

        同时覆盖 ``<root>/<group>/<group>_stage1/`` 和
        ``<root>/<batch>/<group>/`` 等消融部署结构，以及 summary.json 指向
        历史检查点的 experiment_records 归档目录。
        """
        for path in self._candidate_group_summary_paths(group, output_dir):
            try:
                with open(path, 'r') as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if summary.get('_group') != group:
                continue
            summary = copy.deepcopy(summary)
            summary['resumed_from_summary'] = path
            summary['stage2_results'] = {}
            return summary
        return None


    def _load_resume_summary(self, exp_dir: str, group: str, stages: List[int],
                             output_dir: str) -> Optional[Dict[str, Any]]:
        prev = self._load_existing_summary(exp_dir)
        if prev is None:
            prev = self._load_resume_summary_from_search_dirs(group, output_dir)
        prev = self._recover_pretrain_stages_from_ckpt(prev, exp_dir, group, stages)
        if prev is not None and all(
                s not in stages or self._stage_done(prev, f'stage{s}') for s in (0, 1)):
            return prev
        for old_exp_dir in self._candidate_group_dirs(group, output_dir):
            prev = self._recover_pretrain_stages_from_ckpt(prev, old_exp_dir, group, stages)
            if prev is not None and all(
                    s not in stages or self._stage_done(prev, f'stage{s}') for s in (0, 1)):
                prev['resumed_from_group_dir'] = old_exp_dir
                return prev
        return prev


    def _stage_done(self, prev: Optional[Dict[str, Any]], stage_key: str) -> bool:
        """只有预训练阶段记录了仍存在的 ckpt_path，才视为完成。

        因此中断的阶段不会被误判为完成，而是触发其轮次级断点续训。
        """
        if not prev:
            return False
        res = (prev.get('stage_results') or {}).get(stage_key)
        if not isinstance(res, dict) or 'error' in res:
            return False
        ck = res.get('ckpt_path')
        if not ck:
            return False
        return os.path.exists(self._resolve_project_path(ck))


    def _stage2_task_done(self, prev: Optional[Dict[str, Any]], task: str) -> bool:
        """如果历史运行为某个 Stage2 任务记录了无错误结果，则视为完成。"""
        if not prev:
            return False
        res = (prev.get('stage2_results') or {}).get(task)
        return isinstance(res, dict) and 'error' not in res and res.get('status') != 'running'


    def _stage2_unit_done(self, prev: Optional[Dict[str, Any]], task: str,
                          is_multitask: bool) -> bool:
        """检查一个 Stage2 调用单元是否完成。

        多任务结果保存在元数据键下，单任务结果则以任务名为键。
        """
        if not prev:
            return False
        s2 = prev.get('stage2_results') or {}
        if is_multitask:
            meta = s2.get('_multitask_meta')
            return isinstance(meta, dict) and 'error' not in meta and meta.get('status') != 'running'
        return self._stage2_task_done(prev, task)


    def _oracle_stage2_done(self, prev: Optional[Dict[str, Any]]) -> bool:
        if not prev:
            return False
        s2 = prev.get('stage2_results') or {}
        meta = s2.get('_oracle_meta')
        return isinstance(meta, dict) and 'error' not in meta and meta.get('status') != 'running'


    def _experiment_is_complete(self, prev: Optional[Dict[str, Any]],
                                declared_stages: List[int],
                                exp: Dict[str, Any]) -> bool:
        """仅当每个声明的阶段和 Stage2 任务都在 ``prev`` 中有成功记录时返回 True。"""
        if not prev:
            return False
        is_multitask = exp.get('stage2_overrides', {}).get('mode') == 'multitask'
        is_oracle = bool(exp.get('oracle_experiments'))
        s2 = prev.get('stage2_results') or {}
        for s in declared_stages:
            if s in (0, 1):
                if not self._stage_done(prev, f'stage{s}'):
                    return False
            elif s == 2:
                if is_oracle:
                    if not self._oracle_stage2_done(prev):
                        return False
                    tasks = exp.get('stage2_tasks', [])
                    if not tasks:
                        return False
                    if any(not self._stage2_task_done(prev, t) for t in tasks):
                        return False
                elif is_multitask:
                    meta = s2.get('_multitask_meta')
                    if not isinstance(meta, dict) or 'error' in meta or meta.get('status') == 'running':
                        return False
                else:
                    tasks = exp.get('stage2_tasks', [])
                    if not tasks:
                        return False
                    if any(not self._stage2_task_done(prev, t) for t in tasks):
                        return False
        return True


    def _experiment_done_on_disk(self, exp: Dict[str, Any],
                                 output_dir: str) -> Optional[Dict[str, Any]]:
        """实验已全部完成时返回历史摘要，使调度器可整体跳过；否则返回 None。"""
        if not self.AUTO_RESUME:
            return None
        declared = list(exp.get('stages', [0, 1, 2]))
        if self.STAGE_FILTER is not None:
            declared = [s for s in declared if s in self.STAGE_FILTER]
        exp_dir = os.path.join(output_dir, exp['_group'])
        prev = self._load_resume_summary(exp_dir, exp['_group'], declared, output_dir)
        if prev is not None and self._experiment_is_complete(prev, declared, exp):
            return prev
        return None


    def run_experiment(self, exp: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """依次运行单个实验的各阶段，并返回摘要字典。"""
        from experiments._stage_configs import Stage0Config
        from experiments.topodise_experiments.stage0_corn1_pretrain.main import (
            run_single_experiment as run_stage0,
        )
        from experiments._stage_configs import Stage1Config
        from experiments.topodise_experiments.stage1_rscp_finetune.main import (
            run_single_experiment as run_stage1,
        )
        from experiments._stage_configs import Stage2Config
        from experiments.topodise_experiments.stage2_racf_downstream.main import (
            run_single_experiment as run_stage2,
        )
        exp = copy.deepcopy(exp)  # 防止 auto-fallback 修改影响共享的 EXPERIMENTS 条目
        group = exp['_group']
        desc = exp.get('_desc', '')
        requested_stages: List[int] = list(exp.get('stages', [0, 1, 2]))
        if self.STAGE_FILTER is not None:
            requested_stages = [s for s in requested_stages if s in self.STAGE_FILTER]
        # 实验"声明"的阶段集合（auto-fallback 之前），用于断点续跑的完成度判断。
        declared_stages: List[int] = list(requested_stages)

        parent_ckpt: Optional[str] = (exp.get('parent_ckpt_override')
                                      or self.EXTERNAL_PARENT_CKPT or None)
        parent_ckpt = self._resolve_project_path(parent_ckpt) if parent_ckpt else None

        # ---- Auto-fallback：父 ckpt 不指向 / 为空 / 文件缺失 -> 自动补跑 Stage0/1 ----
        # 说明：正常批跑时，main() 的「共享预训练去重」已为每个唯一 Stage0/1 配方训练一份
        #   共享父 ckpt 并通过 parent_ckpt_override 注入，所以这里通常不会触发。
        #   本分支只是兜底：当某条 stage2-only 实验既无共享 ckpt、也无外部 ckpt 时，
        #   就地补跑 Stage0/1（用 DEFAULT 配方）以免整条实验失败。
        # 仅在「请求了 stage2 但没请求 stage1、且未用 RUN_STAGE_FILTER 限制」时触发，
        # 避免覆盖用户用 STAGE_FILTER=2 显式只跑 stage2 的意图。
        auto_fallback: Optional[Dict[str, Any]] = None
        if (2 in requested_stages and 0 not in requested_stages and 1 not in requested_stages
                and self.STAGE_FILTER is None
                and (not parent_ckpt or not os.path.exists(parent_ckpt))):
            reason = 'empty' if not parent_ckpt else f'missing:{parent_ckpt}'
            regen_stages: List[int] = [0, 1]
            for s in requested_stages:
                if s not in regen_stages:
                    regen_stages.append(s)
            requested_stages = regen_stages
            if not exp.get('stage0_overrides'):
                exp['stage0_overrides'] = dict(self.DEFAULT_STAGE0_OVERRIDES)
            if not exp.get('stage1_overrides'):
                exp['stage1_overrides'] = dict(self.DEFAULT_STAGE1_OVERRIDES)
            parent_ckpt = None  # 强制由新跑出的 Stage0/1 产物提供
            auto_fallback = {'reason': reason, 'regen_stages': [0, 1]}
            print(f"[auto-fallback][{group}] parent ckpt {reason}; "
                  f"prepending Stage0/1 -> stages={requested_stages}")

        exp_dir = os.path.join(output_dir, group)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, '_experiment_def.json'), 'w') as f:
            json.dump({k: v for k, v in exp.items() if k != 'stage2_tasks'} | {
                'stage2_tasks': exp.get('stage2_tasks', []),
                'resolved_stages': requested_stages,
                'auto_fallback': auto_fallback,
                'site': self.RUN_SITE,
            }, f, indent=2)

        summary: Dict[str, Any] = {
            '_group': group, '_desc': desc, 'dataset': exp.get('dataset', 'pd'),
            'stages_run': requested_stages, 'started': datetime.now().isoformat(),
            'auto_fallback': auto_fallback,
            'stage_results': {}, 'stage2_results': {},
        }

        # ---- 断点续跑：检测历史运行 ----
        prev_summary = (self._load_resume_summary(exp_dir, group, requested_stages, output_dir)
                        if self.AUTO_RESUME else None)
        if prev_summary is not None and self._experiment_is_complete(prev_summary, declared_stages, exp):
            print(f"[resume][skip][{group}] 已全部完成，跳过重跑")
            return prev_summary
        if prev_summary is not None:
            # 预填已成功的阶段/任务结果；下面的循环据此跳过它们，仅补跑缺失部分。
            for k, v in (prev_summary.get('stage_results') or {}).items():
                if isinstance(v, dict) and 'error' not in v:
                    summary['stage_results'][k] = v
            for k, v in (prev_summary.get('stage2_results') or {}).items():
                if isinstance(v, dict) and 'error' not in v and v.get('status') != 'running':
                    summary['stage2_results'][k] = v
            summary['resumed_from'] = prev_summary.get('finished') or prev_summary.get('started')
            print(f"[resume][{group}] 检测到未完成的历史运行，接续未完成部分 "
                  f"(已完成 stage={list(summary['stage_results'])} "
                  f"stage2={[k for k in summary['stage2_results'] if not k.startswith('_')]})")

        # 阶段 0
        if 0 in requested_stages:
            if self.AUTO_RESUME and self._stage_done(prev_summary, 'stage0'):
                r0 = prev_summary['stage_results']['stage0']
                parent_ckpt = self._resolve_project_path(r0.get('ckpt_path'))
                summary['stage_results']['stage0'] = r0
                print(f"[resume][{group}][stage0] 已完成，复用 {parent_ckpt}")
            else:
                cfg = Stage0Config()
                cfg = self._apply_overrides(cfg, exp.get('stage0_overrides', {}))
                cfg.num_workers = self.NUM_WORKERS
                ckpt_dir0 = os.path.join(exp_dir, self._stage_run_name(group, 0), 'ckpt')
                last_ckpt0 = os.path.join(ckpt_dir0, 'last.pt')
                best_ckpt0 = os.path.join(ckpt_dir0, 'ema_best.pt')
                if self.AUTO_RESUME:
                    cfg.resume = True
                    if os.path.exists(last_ckpt0):
                        cfg.resume_ckpt = last_ckpt0
                        print(f"[resume][{group}][stage0] {last_ckpt0}")
                    elif os.path.exists(best_ckpt0):
                        cfg.resume_ckpt = best_ckpt0
                        print(f"[resume][{group}][stage0] fallback to {best_ckpt0}")
                cfg = self._resolve_cfg_paths(cfg, ['corn1_root'])
                cfg.output_dir = exp_dir
                cfg.run_name = self._stage_run_name(group, 0)
                path_err = self._validate_paths({'corn1_root': cfg.corn1_root})
                if path_err:
                    self._record_stage_error(summary, 'stage0', exp_dir, path_err, '')
                    return summary
                t0 = time.time()
                try:
                    r = run_stage0(cfg)
                    summary['stage_results']['stage0'] = {**r, 'elapsed_s': time.time() - t0}
                    parent_ckpt = r['ckpt_path']
                    self._save_summary(summary, exp_dir)
                except Exception as e:
                    self._record_stage_error(summary, 'stage0', exp_dir, str(e), traceback.format_exc())
                    return summary

        # 阶段 1
        if 1 in requested_stages:
            if self.AUTO_RESUME and self._stage_done(prev_summary, 'stage1'):
                r1 = prev_summary['stage_results']['stage1']
                parent_ckpt = self._resolve_project_path(r1.get('ckpt_path'))
                summary['stage_results']['stage1'] = r1
                print(f"[resume][{group}][stage1] 已完成，复用 {parent_ckpt}")
            elif parent_ckpt is None:
                summary['stage_results']['stage1'] = {'error': 'missing parent ckpt for stage1'}
                self._save_summary(summary, exp_dir)
                return summary
            else:
                cfg = Stage1Config()
                cfg.parent_ckpt = parent_ckpt
                cfg = self._apply_overrides(cfg, exp.get('stage1_overrides', {}))
                cfg.num_workers = self.NUM_WORKERS
                ckpt_dir1 = os.path.join(exp_dir, self._stage_run_name(group, 1), 'ckpt')
                last_ckpt1 = os.path.join(ckpt_dir1, 'last.pt')
                best_ckpt1 = os.path.join(ckpt_dir1, 'ema_best.pt')
                if self.AUTO_RESUME:
                    cfg.resume = True
                    if os.path.exists(last_ckpt1):
                        cfg.resume_ckpt = last_ckpt1
                        print(f"[resume][{group}][stage1] {last_ckpt1}")
                    elif os.path.exists(best_ckpt1):
                        cfg.resume_ckpt = best_ckpt1
                        print(f"[resume][{group}][stage1] fallback to {best_ckpt1}")
                cfg = self._resolve_cfg_paths(cfg, ['parent_ckpt', 'pd_json', 'base_path', 'region_pair_manifest'])
                cfg.output_dir = exp_dir
                cfg.run_name = self._stage_run_name(group, 1)
                path_err = self._validate_paths({'parent_ckpt': cfg.parent_ckpt, 'pd_json': cfg.pd_json})
                if path_err:
                    self._record_stage_error(summary, 'stage1', exp_dir, path_err, '')
                    return summary
                t0 = time.time()
                try:
                    r = run_stage1(cfg)
                    summary['stage_results']['stage1'] = {**r, 'elapsed_s': time.time() - t0}
                    parent_ckpt = r['ckpt_path']
                    self._save_summary(summary, exp_dir)
                except Exception as e:
                    self._record_stage_error(summary, 'stage1', exp_dir, str(e), traceback.format_exc())
                    return summary

        # 阶段 2（按任务运行，或执行一次多任务训练）
        if 2 in requested_stages:
            if parent_ckpt is None:
                summary['stage_results']['stage2'] = {'error': 'missing parent ckpt for stage2'}
                self._save_summary(summary, exp_dir)
                return summary
            if exp.get('oracle_experiments'):
                if self.AUTO_RESUME and self._oracle_stage2_done(prev_summary):
                    print(f"[resume][{group}][stage2/oracle] 已完成，跳过")
                else:
                    summary = self._run_oracle_stage2(exp, exp_dir, parent_ckpt, summary)
                summary['finished'] = datetime.now().isoformat()
                self._save_summary(summary, exp_dir)
                return summary
            is_multitask = exp.get('stage2_overrides', {}).get('mode') == 'multitask'
            # 多任务模式只调用一次 Stage2，同时产生三个 PD 任务的准确率。
            if is_multitask:
                tasks_iter = ['__multitask__']
            else:
                tasks_iter = exp.get('stage2_tasks', [])
            for task in tasks_iter:
                if self.AUTO_RESUME and self._stage2_unit_done(prev_summary, task, is_multitask):
                    # 该 stage2 单元（单任务或多任务）此前已成功完成；其结果
                    # 已在上面预填进 summary['stage2_results']，这里直接跳过不重跑。
                    label = 'multitask' if is_multitask else task
                    print(f"[resume][{group}][stage2/{label}] 已完成，跳过")
                    continue
                cfg = Stage2Config()
                cfg.parent_ckpt = parent_ckpt
                if not is_multitask:
                    cfg.task = task
                cfg = self._apply_overrides(cfg, exp.get('stage2_overrides', {}))
                cfg.num_workers = self.NUM_WORKERS
                cfg = self._resolve_cfg_paths(cfg, ['parent_ckpt', 'pd_json', 'base_path'])
                cfg.output_dir = exp_dir
                cfg.run_name = self._stage_run_name(group, 2)
                path_err = self._validate_paths({'parent_ckpt': cfg.parent_ckpt, 'json': cfg.pd_json})
                if path_err:
                    summary['stage2_results'][task] = {'error': path_err, 'traceback': ''}
                    print(f"[error][{group}][stage2/{task}] {path_err}")
                    continue
                t0 = time.time()
                try:
                    unit_key = '_multitask_meta' if is_multitask else task
                    summary['stage2_results'][unit_key] = {
                        'status': 'running',
                        'started': datetime.now().isoformat(),
                        'run_name': cfg.run_name,
                        'task': task,
                    }
                    self._save_summary(summary, exp_dir)
                    r = run_stage2(cfg)
                    if is_multitask and isinstance(r, dict) and 'mt_results' in r:
                        elapsed = time.time() - t0
                        for sub_task, sub_r in r['mt_results'].items():
                            summary['stage2_results'][sub_task] = {
                                **sub_r, 'elapsed_s': elapsed, 'mode': 'multitask',
                            }
                        summary['stage2_results']['_multitask_meta'] = {
                            k: v for k, v in r.items() if k != 'mt_results'
                        }
                        summary['stage2_results']['_multitask_meta']['elapsed_s'] = elapsed
                    else:
                        summary['stage2_results'][task] = self._add_patient_auc({
                            **r, 'elapsed_s': time.time() - t0,
                        })
                except Exception as e:
                    tb = traceback.format_exc()
                    summary['stage2_results'][task] = {'error': str(e), 'traceback': tb}
                    print(f"[error][{group}][stage2/{task}] {e}")
                    print(tb)
                # 每完成一个 stage2 单元就增量落盘，便于硬中断后接续。
                self._save_summary(summary, exp_dir)

        summary['finished'] = datetime.now().isoformat()
        self._save_summary(summary, exp_dir)
        return summary


    def _std(self, values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


    def _binary_auc(self, labels: List[int], scores: List[float]) -> Optional[float]:
        if len(labels) != len(scores) or len(labels) < 2:
            return None
        n_pos = sum(1 for y in labels if int(y) == 1)
        n_neg = sum(1 for y in labels if int(y) == 0)
        if n_pos == 0 or n_neg == 0:
            return None
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]))
        ranks = [0.0] * len(scores)
        idx = 0
        while idx < len(ranked):
            j = idx + 1
            while j < len(ranked) and float(ranked[j][1]) == float(ranked[idx][1]):
                j += 1
            avg_rank = (idx + 1 + j) / 2.0
            for k in range(idx, j):
                ranks[ranked[k][0]] = avg_rank
            idx = j
        rank_sum_pos = sum(rank for rank, y in zip(ranks, labels) if int(y) == 1)
        return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


    def _patient_binary_score(self, row: Dict[str, Any]) -> Optional[float]:
        probs = self._safe_float_vec(row.get('patient_probs'))
        if probs is not None and len(probs) >= 2:
            return float(probs[1])
        logits = self._safe_float_vec(row.get('patient_logits'))
        if logits is not None and len(logits) >= 2:
            return float(self._softmax_vec(logits)[1])
        return None


    def _add_patient_auc(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result
        if str(result.get('task', '')).split('_')[-1] not in ('binary',):
            return result
        auc_by_fold: List[Dict[str, Any]] = []
        pooled_labels: List[int] = []
        pooled_scores: List[float] = []
        for fold in result.get('fold_results', []) or []:
            preds = fold.get('best_patient_predictions') or fold.get('patient_predictions') or []
            labels: List[int] = []
            scores: List[float] = []
            for row in preds:
                if not isinstance(row, dict):
                    continue
                try:
                    label = int(row.get('y_true'))
                except (TypeError, ValueError):
                    continue
                score = self._patient_binary_score(row)
                if score is None:
                    continue
                labels.append(label)
                scores.append(score)
            auc = self._binary_auc(labels, scores)
            fold_auc = {
                'fold': fold.get('fold'),
                'auc': auc,
                'n_patients': len(labels),
                'n_pos': sum(1 for y in labels if int(y) == 1),
                'n_neg': sum(1 for y in labels if int(y) == 0),
            }
            auc_by_fold.append(fold_auc)
            pooled_labels.extend(labels)
            pooled_scores.extend(scores)
        valid_fold_aucs = [float(row['auc']) for row in auc_by_fold if row.get('auc') is not None]
        pooled_auc = self._binary_auc(pooled_labels, pooled_scores)
        result['patient_auc_by_fold'] = auc_by_fold
        result['patient_auc'] = (sum(valid_fold_aucs) / len(valid_fold_aucs)) if valid_fold_aucs else None
        result['patient_auc_std'] = self._std(valid_fold_aucs)
        result['patient_auc_pooled'] = pooled_auc
        result['patient_auc_note'] = 'Computed from best_patient_predictions patient_probs/logits; score is P(class=1).'
        return result


    def _safe_float_vec(self, value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or not value:
            return None
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None


    def _softmax_vec(self, logits: List[float], temp: float = 1.0) -> List[float]:
        scale = max(float(temp), 1e-8)
        vals = [float(v) / scale for v in logits]
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        denom = max(sum(exps), 1e-12)
        return [v / denom for v in exps]


    def _mean_vec(self, vectors: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        if not vectors:
            return []
        if weights is None:
            weights = [1.0] * len(vectors)
        total = sum(float(w) for w in weights)
        if total <= 0:
            weights = [1.0] * len(vectors)
            total = float(len(vectors))
        out = [0.0] * len(vectors[0])
        for vec, weight in zip(vectors, weights):
            for idx, val in enumerate(vec):
                out[idx] += float(weight) * float(val)
        return [v / total for v in out]


    def _argmax_vec(self, values: List[float]) -> int:
        return max(range(len(values)), key=lambda idx: (float(values[idx]), -idx))


    def _entropy_weight(self, probs: List[float]) -> float:
        if len(probs) <= 1:
            return 1.0
        ent = -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs)
        return max(1e-6, 1.0 - ent / math.log(len(probs)))


    def _prob_entropy(self, probs: List[float]) -> float:
        return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probs)


    def _fold_calibrated_entropy_choice(self,
        candidates: List[Dict[str, Any]],
        entropy_baselines: Dict[str, float],
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Route by entropy relative to each candidate's fold-level baseline.

        Raw entropies from independently trained candidates are not necessarily
        calibrated to the same scale. Centering each entropy by that candidate's
        mean entropy in the fold preserves entropy routing while removing a fixed
        candidate-specific confidence offset. No labels are used.
        """
        if not candidates:
            raise ValueError('fold-calibrated entropy requires at least one candidate')

        centered_entropies: Dict[str, float] = {}
        for candidate in candidates:
            name = str(candidate.get('candidate', ''))
            entropy = self._prob_entropy(list(candidate['probs']))
            baseline = float(entropy_baselines.get(name, 0.0))
            centered_entropies[name] = entropy - baseline
        chosen = min(
            candidates,
            key=lambda c: (centered_entropies[str(c.get('candidate', ''))],
                           str(c.get('candidate', ''))),
        )
        return chosen, centered_entropies


    def _expected_round_pred(self, probs: List[float]) -> int:
        exp = sum(idx * float(prob) for idx, prob in enumerate(probs))
        return min(len(probs) - 1, max(0, int(round(exp))))


    def _prediction_metrics(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {
                'n_patients_oof': 0, 'accuracy': 0.0, 'balanced_accuracy': 0.0,
                'mae': 0.0, 'per_class_accuracy': {}, 'fold_accuracy_mean': 0.0,
                'fold_accuracy_std': 0.0,
            }
        n = len(rows)
        acc = sum(int(r.get('correct', 0)) for r in rows) / max(n, 1)
        mae = sum(abs(int(r['y_pred']) - int(r['y_true'])) for r in rows) / max(n, 1)
        per_total: Dict[int, int] = {}
        per_correct: Dict[int, int] = {}
        fold_total: Dict[int, int] = {}
        fold_correct: Dict[int, int] = {}
        for row in rows:
            y = int(row['y_true'])
            fold = int(row['fold'])
            per_total[y] = per_total.get(y, 0) + 1
            per_correct[y] = per_correct.get(y, 0) + int(row.get('correct', 0))
            fold_total[fold] = fold_total.get(fold, 0) + 1
            fold_correct[fold] = fold_correct.get(fold, 0) + int(row.get('correct', 0))
        per_acc = {c: per_correct.get(c, 0) / max(per_total[c], 1) for c in sorted(per_total)}
        fold_acc = [fold_correct.get(f, 0) / max(fold_total[f], 1) for f in sorted(fold_total)]
        return {
            'n_patients_oof': n,
            'accuracy': acc,
            'balanced_accuracy': sum(per_acc.values()) / max(len(per_acc), 1),
            'mae': mae,
            'per_class_accuracy': per_acc,
            'fold_accuracy_mean': sum(fold_acc) / max(len(fold_acc), 1),
            'fold_accuracy_std': self._std(fold_acc),
        }


    def _summarize_ensemble_method(self, method: str, rows: List[Dict[str, Any]], practical: bool) -> Dict[str, Any]:
        return {
            'method': method,
            'practical': practical,
            **self._prediction_metrics(rows),
        }


    def _aggregate_patient_logit_ensembles(self, child_summaries: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
        grouped: Dict[tuple, List[Dict[str, Any]]] = {}
        candidates: List[str] = []
        for child in child_summaries:
            if not isinstance(child, dict):
                continue
            candidate = str(child.get('_group') or '')
            result = (child.get('stage2_results') or {}).get(task)
            if not candidate or not isinstance(result, dict) or 'error' in result:
                continue
            candidates.append(candidate)
            for fold in result.get('fold_results', []) or []:
                try:
                    fold_idx = int(fold.get('fold'))
                except (TypeError, ValueError):
                    continue
                ckpt_acc = self._fold_metric(fold, 'best_patient_acc')
                preds = fold.get('best_patient_predictions') or fold.get('patient_predictions') or []
                if not isinstance(preds, list):
                    continue
                for item in preds:
                    if not isinstance(item, dict) or item.get('patient_id') is None:
                        continue
                    logits = self._safe_float_vec(item.get('patient_logits'))
                    probs = self._safe_float_vec(item.get('patient_probs'))
                    if logits is None and probs is None:
                        continue
                    if probs is None and logits is not None:
                        probs = self._softmax_vec(logits)
                    if logits is None and probs is not None:
                        logits = [math.log(max(float(p), 1e-12)) for p in probs]
                    try:
                        row = {
                            'candidate': candidate,
                            'fold': fold_idx,
                            'patient_id': str(item.get('patient_id')),
                            'y_true': int(item.get('y_true')),
                            'y_pred': int(item.get('y_pred')),
                            'logits': logits,
                            'probs': probs,
                            'checkpoint_weight': float(ckpt_acc) if ckpt_acc is not None else 1.0,
                        }
                    except (TypeError, ValueError):
                        continue
                    grouped.setdefault((fold_idx, row['patient_id']), []).append(row)

        candidates = sorted(set(candidates))
        expected = set(candidates)
        complete_groups = {
            key: sorted(vals, key=lambda r: r['candidate'])
            for key, vals in grouped.items()
            if expected and {r['candidate'] for r in vals} == expected
        }
        if not complete_groups:
            return {
                'status': 'skipped',
                'reason': 'no aligned patient_logits across all candidates',
                'candidates': candidates,
            }

        _fold_oracle_candidate: Dict[int, str] = {}
        if task != 'pd_stage':
            # 非分期任务保留诊断上界；分期路径不再计算任何使用标签的路由。
            _fold_correct: Dict[int, Dict[str, int]] = {}
            for (fold_idx, _patient_id), vals in complete_groups.items():
                y_values = {int(v['y_true']) for v in vals}
                if len(y_values) != 1:
                    continue
                y_true = next(iter(y_values))
                bucket = _fold_correct.setdefault(fold_idx, {})
                for value in vals:
                    candidate = value['candidate']
                    bucket[candidate] = bucket.get(candidate, 0) + int(
                        int(value['y_pred']) == y_true
                    )
            for fold_idx, bucket in _fold_correct.items():
                _fold_oracle_candidate[fold_idx] = max(
                    candidates, key=lambda c: (bucket.get(c, 0), c)
                )

        # 候选模型由不同训练轨迹产生，原始熵的尺度可能存在固定偏移。按折、按候选
        # 计算无标签基准熵，供校准熵路由使用。这里只读取预测分布，不读取标签。
        _fold_candidate_entropy_baseline: Dict[tuple, float] = {}
        if task == 'pd_stage':
            _fold_candidate_entropy_values: Dict[tuple, List[float]] = {}
            for (fold_idx, _patient_id), vals in complete_groups.items():
                for value in vals:
                    key = (fold_idx, value['candidate'])
                    _fold_candidate_entropy_values.setdefault(key, []).append(
                        self._prob_entropy(list(value['probs']))
                    )
            _fold_candidate_entropy_baseline = {
                key: sum(values) / len(values)
                for key, values in _fold_candidate_entropy_values.items()
                if values
            }

        def _build_prediction_rows(method: str) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for (fold_idx, patient_id), vals in sorted(complete_groups.items()):
                y_values = {int(v['y_true']) for v in vals}
                if len(y_values) != 1:
                    continue
                y_true = next(iter(y_values))
                probs: List[float]
                if method.startswith('single:'):
                    candidate = method.split(':', 1)[1]
                    chosen = next(v for v in vals if v['candidate'] == candidate)
                    pred = int(chosen['y_pred'])
                    probs = list(chosen['probs'])
                    extra = {'candidate': candidate}
                elif method == 'majority_vote_tie_avg_prob':
                    vote_counts: Dict[int, int] = {}
                    for v in vals:
                        vote_counts[int(v['y_pred'])] = vote_counts.get(int(v['y_pred']), 0) + 1
                    best_count = max(vote_counts.values())
                    tied = [label for label, count in vote_counts.items() if count == best_count]
                    probs = self._mean_vec([list(v['probs']) for v in vals])
                    pred = tied[0] if len(tied) == 1 else max(tied, key=lambda label: (probs[label], -label))
                    extra = {'votes': vote_counts}
                elif method == 'fold_calibrated_entropy_routing':
                    baselines = {
                        value['candidate']: _fold_candidate_entropy_baseline[
                            (fold_idx, value['candidate'])
                        ]
                        for value in vals
                    }
                    chosen, centered_entropies = self._fold_calibrated_entropy_choice(
                        vals, baselines
                    )
                    probs = list(chosen['probs'])
                    pred = int(chosen['y_pred'])
                    extra = {
                        'selected_candidate': chosen['candidate'],
                        'patient_entropy': self._prob_entropy(probs),
                        'entropy_baseline': baselines[chosen['candidate']],
                        'centered_entropy': centered_entropies[chosen['candidate']],
                        'centered_entropies': centered_entropies,
                        'route': 'minimum_fold_centered_entropy',
                    }
                elif method == 'mean_logits':
                    logits = self._mean_vec([list(v['logits']) for v in vals])
                    probs = self._softmax_vec(logits)
                    pred = self._argmax_vec(logits)
                    extra = {}
                elif method == 'mean_probs':
                    probs = self._mean_vec([list(v['probs']) for v in vals])
                    pred = self._argmax_vec(probs)
                    extra = {}
                elif method == 'mean_probs_expected_round':
                    probs = self._mean_vec([list(v['probs']) for v in vals])
                    pred = self._expected_round_pred(probs)
                    extra = {'ordinal_rule': 'round_expected_class'}
                elif method == 'confidence_weighted_probs':
                    weights = [self._entropy_weight(list(v['probs'])) for v in vals]
                    probs = self._mean_vec([list(v['probs']) for v in vals], weights)
                    pred = self._argmax_vec(probs)
                    extra = {'weights': {v['candidate']: w for v, w in zip(vals, weights)}}
                elif method == 'checkpoint_weighted_probs_optimistic':
                    weights = [max(1e-6, float(v['checkpoint_weight'])) for v in vals]
                    probs = self._mean_vec([list(v['probs']) for v in vals], weights)
                    pred = self._argmax_vec(probs)
                    extra = {'weights': {v['candidate']: w for v, w in zip(vals, weights)}}
                elif method == 'checkpoint_weighted_logits_optimistic':
                    weights = [max(1e-6, float(v['checkpoint_weight'])) for v in vals]
                    logits = self._mean_vec([list(v['logits']) for v in vals], weights)
                    probs = self._softmax_vec(logits)
                    pred = self._argmax_vec(logits)
                    extra = {'weights': {v['candidate']: w for v, w in zip(vals, weights)}}
                elif method == 'oracle_pick_correct_candidate_upper_bound':
                    chosen = max(vals, key=lambda v: (int(v['y_pred'] == y_true), max(v['probs'])))
                    pred = int(chosen['y_pred'])
                    probs = list(chosen['probs'])
                    extra = {'selected_candidate': chosen['candidate']}
                elif method == 'fold_wise_oracle_upper_bound':
                    # 理论上限：每折选择该折患者准确率最高的候选项（使用验证标签），
                    # 再将其应用于该折所有患者。此方法不能部署。
                    candidate = _fold_oracle_candidate.get(fold_idx)
                    chosen = next((v for v in vals if v['candidate'] == candidate), vals[0])
                    probs = list(chosen['probs'])
                    pred = self._argmax_vec(probs)
                    extra = {'selected_candidate': chosen['candidate'], 'selection': 'per_fold_best'}
                else:
                    continue
                out.append({
                    'method': method,
                    'fold': fold_idx,
                    'patient_id': patient_id,
                    'y_true': y_true,
                    'y_pred': int(pred),
                    'correct': int(int(pred) == y_true),
                    'ensemble_probs': [float(v) for v in probs],
                    'extra': extra,
                })
            return out

        if task == 'pd_stage':
            # 分期只保留唯一正式算法；不再生成旧熵路由、投票、均值或 oracle 结果。
            practical_methods = ['fold_calibrated_entropy_routing']
            optimistic_methods: List[str] = []
        else:
            # subtype 等任务继续保留其原有的多数投票及诊断集成方法。
            practical_methods = (
                [f'single:{candidate}' for candidate in candidates]
                + ['majority_vote_tie_avg_prob', 'mean_logits', 'mean_probs',
                   'mean_probs_expected_round', 'confidence_weighted_probs']
            )
            optimistic_methods = ['checkpoint_weighted_probs_optimistic',
                                  'checkpoint_weighted_logits_optimistic',
                                  'fold_wise_oracle_upper_bound',
                                  'oracle_pick_correct_candidate_upper_bound']
        summaries: List[Dict[str, Any]] = []
        predictions_by_method: Dict[str, List[Dict[str, Any]]] = {}
        for method in practical_methods + optimistic_methods:
            rows = _build_prediction_rows(method)
            predictions_by_method[method] = rows
            summaries.append(self._summarize_ensemble_method(method, rows, method in practical_methods))
        summaries.sort(key=lambda r: (float(r['accuracy']), float(r['balanced_accuracy']), -float(r['mae'])), reverse=True)
        practical = [s for s in summaries if s.get('practical')]
        return {
            'status': 'complete',
            'task': task,
            'candidates': candidates,
            'n_aligned_patients': len(complete_groups),
            'methods': summaries,
            'best_practical': practical[0] if practical else None,
            'best_balanced_practical': max(practical, key=lambda r: (float(r['balanced_accuracy']), float(r['accuracy']))) if practical else None,
            'predictions_by_method': predictions_by_method,
            'note': 'Computed automatically from best_patient_predictions.patient_logits/patient_probs saved by each candidate fold.',
        }


    def _combine_subtype_hierarchy_ensembles(self, td_ens: Dict[str, Any], pm_ens: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(td_ens, dict) or not isinstance(pm_ens, dict):
            return {'status': 'skipped', 'reason': 'missing component ensembles'}
        if td_ens.get('status') != 'complete' or pm_ens.get('status') != 'complete':
            return {'status': 'skipped', 'reason': 'component ensembles are incomplete'}
        td_preds = td_ens.get('predictions_by_method') or {}
        pm_preds = pm_ens.get('predictions_by_method') or {}
        methods = sorted(set(td_preds) & set(pm_preds))
        if not methods:
            return {'status': 'skipped', 'reason': 'no overlapping component ensemble methods'}

        summaries: List[Dict[str, Any]] = []
        predictions_by_method: Dict[str, List[Dict[str, Any]]] = {}
        for method in methods:
            td_map = {
                (int(r['fold']), str(r['patient_id'])): r
                for r in td_preds.get(method, [])
                if isinstance(r, dict) and r.get('patient_id') is not None
            }
            pm_map = {
                (int(r['fold']), str(r['patient_id'])): r
                for r in pm_preds.get(method, [])
                if isinstance(r, dict) and r.get('patient_id') is not None
            }
            rows: List[Dict[str, Any]] = []
            for key, gate in sorted(td_map.items()):
                y_gate = int(gate['y_true'])
                gate_pred = int(gate['y_pred'])
                gate_probs = self._safe_float_vec(gate.get('ensemble_probs')) or []
                pm = pm_map.get(key)
                pm_probs = self._safe_float_vec((pm or {}).get('ensemble_probs')) or []
                if y_gate == 0:
                    y_true = 0
                elif pm is not None:
                    y_true = int(pm['y_true']) + 1
                else:
                    continue
                if gate_pred == 0:
                    y_pred = 0
                elif pm is not None:
                    y_pred = int(pm['y_pred']) + 1
                else:
                    continue
                p_td = float(gate_probs[0]) if len(gate_probs) > 0 else float(gate_pred == 0)
                p_rest = float(gate_probs[1]) if len(gate_probs) > 1 else float(gate_pred != 0)
                p_mixed = p_rest * (float(pm_probs[0]) if len(pm_probs) > 0 else float((pm or {}).get('y_pred') == 0))
                p_pigd = p_rest * (float(pm_probs[1]) if len(pm_probs) > 1 else float((pm or {}).get('y_pred') == 1))
                probs = [p_td, p_mixed, p_pigd]
                denom = max(sum(probs), 1e-12)
                probs = [p / denom for p in probs]
                rows.append({
                    'method': method,
                    'fold': key[0],
                    'patient_id': key[1],
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'correct': int(y_true == y_pred),
                    'ensemble_probs': probs,
                    'extra': {
                        'tdrest_pred': gate_pred,
                        'pigd_mixed_pred': (None if pm is None else int(pm['y_pred'])),
                    },
                })
            predictions_by_method[method] = rows
            summaries.append(self._summarize_ensemble_method(method, rows, not method.startswith('checkpoint_') and not method.startswith('oracle_')))
        summaries.sort(key=lambda r: (float(r['accuracy']), float(r['balanced_accuracy']), -float(r['mae'])), reverse=True)
        practical = [s for s in summaries if s.get('practical')]
        return {
            'status': 'complete',
            'task': 'pd_subtype',
            'mode': 'subtype_hierarchy_component_ensemble',
            'component_tasks': ['pd_subtype_tdrest', 'pd_subtype_pigd_mixed'],
            'n_aligned_patients': max((int(s.get('n_patients_oof') or 0) for s in summaries), default=0),
            'methods': summaries,
            'best_practical': practical[0] if practical else None,
            'best_balanced_practical': max(practical, key=lambda r: (float(r['balanced_accuracy']), float(r['accuracy']))) if practical else None,
            'predictions_by_method': predictions_by_method,
            'note': 'Combined tdrest gate ensemble with pigd-vs-mixed ensemble per method to produce final 3-class subtype predictions.',
        }


    def _aggregate_oracle_results(self, group: str, child_summaries: List[Dict[str, Any]],
                                  tasks: List[str], select_field: str = 'best_val_acc',
                                  auto_patient_logit_ensemble: bool = True) -> Dict[str, Any]:
        stage2: Dict[str, Any] = {}
        candidate_results = {s.get('_group'): s for s in child_summaries if isinstance(s, dict)}
        for task in tasks:
            by_fold: Dict[int, List[Dict[str, Any]]] = {}
            for child in child_summaries:
                if not isinstance(child, dict):
                    continue
                result = (child.get('stage2_results') or {}).get(task)
                if not isinstance(result, dict) or 'error' in result:
                    continue
                for fold in result.get('fold_results', []) or []:
                    try:
                        fold_idx = int(fold.get('fold'))
                        if select_field == 'best_balanced_acc':
                            pc = fold.get('best_per_class_acc') or fold.get('pat_per_class_acc') or fold.get('per_class_acc') or {}
                            if not isinstance(pc, dict) or not pc:
                                continue
                            acc = sum(float(v) for v in pc.values()) / len(pc)
                        else:
                            acc = float(fold.get(select_field, fold.get('best_val_acc')))
                    except (TypeError, ValueError):
                        continue
                    row = dict(fold)
                    row['oracle_score'] = acc
                    row['oracle_source'] = child.get('_group')
                    by_fold.setdefault(fold_idx, []).append(row)
            oracle_folds: List[Dict[str, Any]] = []
            for fold_idx in sorted(by_fold):
                best = max(by_fold[fold_idx], key=lambda r: float(r.get('oracle_score', 0.0)))
                oracle_folds.append(best)
            vals = [float(r['oracle_score']) for r in oracle_folds]
            if vals:
                mean = sum(vals) / len(vals)
                stage2[task] = {
                    'best_val_acc': mean,
                    'best_val_acc_std': self._std(vals),
                    'oracle_select_field': select_field,
                    'evaluated_folds': len(vals),
                    'fold_results': oracle_folds,
                    'mode': 'oracle',
                    'oracle_candidates': list(candidate_results),
                    'status': 'complete',
                }
                if auto_patient_logit_ensemble:
                    stage2[task]['patient_logit_ensembles'] = self._aggregate_patient_logit_ensembles(child_summaries, task)
            else:
                stage2[task] = {
                    'error': 'no completed candidate fold results for oracle aggregation',
                    'oracle_candidates': list(candidate_results),
                }
        stage2['_oracle_meta'] = {
            'status': 'complete',
            'group': group,
            'candidates': list(candidate_results),
            'candidate_results': candidate_results,
        }
        return stage2


    def _fold_metric(self, row: Dict[str, Any], field: str) -> Optional[float]:
        for key in (field, 'best_patient_acc', 'best_val_acc'):
            try:
                if row.get(key) is not None:
                    return float(row.get(key))
            except (TypeError, ValueError):
                pass
        return None


    def _fold_class_acc(self, row: Dict[str, Any], cls: int) -> Optional[float]:
        pc = row.get('best_per_class_acc') or row.get('pat_per_class_acc') or row.get('per_class_acc') or {}
        if not isinstance(pc, dict):
            return None
        for key in (cls, str(cls)):
            if key in pc:
                try:
                    return float(pc[key])
                except (TypeError, ValueError):
                    return None
        return None


    def _patient_prediction_map(self, row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        preds = row.get('best_patient_predictions') or row.get('patient_predictions') or []
        if not isinstance(preds, list):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for item in preds:
            if not isinstance(item, dict) or item.get('patient_id') is None:
                continue
            try:
                out[str(item.get('patient_id'))] = {
                    'y_true': int(item.get('y_true')),
                    'y_pred': int(item.get('y_pred')),
                }
            except (TypeError, ValueError):
                continue
        return out


    def _aggregate_subtype_hierarchy_oracle(self, group: str, child_summaries: List[Dict[str, Any]],
                                            select_field: str = 'best_val_acc') -> Dict[str, Any]:
        component_tasks = ['pd_subtype_tdrest', 'pd_subtype_pigd_mixed']
        stage2 = self._aggregate_oracle_results(group, child_summaries, component_tasks, select_field)
        meta = stage2.pop('_oracle_meta', {})
        td_rows = {
            int(r['fold']): r for r in stage2.get('pd_subtype_tdrest', {}).get('fold_results', [])
            if r.get('fold') is not None
        }
        pm_rows = {
            int(r['fold']): r for r in stage2.get('pd_subtype_pigd_mixed', {}).get('fold_results', [])
            if r.get('fold') is not None
        }

        hierarchy_folds: List[Dict[str, Any]] = []
        for fold_idx in sorted(set(td_rows) & set(pm_rows)):
            td = td_rows[fold_idx]
            pm = pm_rows[fold_idx]
            gate_acc = self._fold_metric(td, 'best_patient_acc')
            pm_acc = self._fold_metric(pm, 'best_patient_acc')
            td_pred = self._patient_prediction_map(td)
            pm_pred = self._patient_prediction_map(pm)
            try:
                n_full = int(td.get('n_val'))
                n_rest = int(pm.get('n_val'))
            except (TypeError, ValueError):
                continue
            if gate_acc is None or pm_acc is None or n_full <= 0 or n_rest <= 0 or n_rest > n_full:
                continue
            n_td = n_full - n_rest
            td_recall = self._fold_class_acc(td, 0)
            rest_specificity = self._fold_class_acc(td, 1)
            if td_recall is None or rest_specificity is None:
                td_recall = gate_acc
                rest_specificity = gate_acc
                estimate_mode = 'neutral_gate_accuracy'
            else:
                estimate_mode = 'gate_per_class_accuracy'
            prediction_rows: List[Dict[str, Any]] = []
            per_class_total: Dict[int, int] = {}
            per_class_correct: Dict[int, int] = {}
            if td_pred and pm_pred:
                for pid, gate in sorted(td_pred.items()):
                    y_gate = int(gate['y_true'])
                    gate_out = int(gate['y_pred'])
                    if y_gate == 0:
                        y_true = 0
                        y_pred = 0 if gate_out == 0 else 1
                    else:
                        pm_item = pm_pred.get(pid)
                        if pm_item is None:
                            continue
                        y_true = int(pm_item['y_true']) + 1
                        y_pred = 0 if gate_out == 0 else int(pm_item['y_pred']) + 1
                    correct = int(y_true == y_pred)
                    prediction_rows.append({
                        'patient_id': pid,
                        'y_true': y_true,
                        'y_pred': y_pred,
                        'tdrest_pred': gate_out,
                        'pigd_mixed_pred': (pm_pred.get(pid) or {}).get('y_pred'),
                        'correct': correct,
                    })
                    per_class_total[y_true] = per_class_total.get(y_true, 0) + 1
                    per_class_correct[y_true] = per_class_correct.get(y_true, 0) + correct
            if prediction_rows:
                hierarchy_acc = sum(r['correct'] for r in prediction_rows) / len(prediction_rows)
                estimate_mode = 'paired_patient_predictions'
            else:
                hierarchy_acc = (n_td * td_recall + n_rest * rest_specificity * pm_acc) / n_full
            hierarchy_per_class_acc = {
                c: per_class_correct[c] / max(per_class_total[c], 1)
                for c in sorted(per_class_total)
            }
            hierarchy_folds.append({
                'fold': fold_idx,
                'best_val_acc': hierarchy_acc,
                'best_patient_acc': hierarchy_acc,
                'oracle_score': hierarchy_acc,
                'estimate_mode': estimate_mode,
                'n_val': len(prediction_rows) if prediction_rows else n_full,
                'n_td': n_td,
                'n_rest': n_rest,
                'td_recall_from_gate': td_recall,
                'rest_specificity_from_gate': rest_specificity,
                'pigd_mixed_acc': pm_acc,
                'tdrest_acc': gate_acc,
                'tdrest_source': td.get('oracle_source'),
                'pigd_mixed_source': pm.get('oracle_source'),
                'hierarchy_per_class_acc': hierarchy_per_class_acc,
                'best_patient_predictions': prediction_rows,
            })

        vals = [float(r['oracle_score']) for r in hierarchy_folds]
        hierarchy_patient_logit_ensembles = self._combine_subtype_hierarchy_ensembles(
            (stage2.get('pd_subtype_tdrest') or {}).get('patient_logit_ensembles') or {},
            (stage2.get('pd_subtype_pigd_mixed') or {}).get('patient_logit_ensembles') or {},
        )
        if vals:
            mean = sum(vals) / len(vals)
            stage2['pd_subtype'] = {
                'task': 'pd_subtype_hierarchy_oracle',
                'best_val_acc': mean,
                'best_val_acc_std': self._std(vals),
                'best_patient_acc': mean,
                'evaluated_folds': len(vals),
                'fold_results': hierarchy_folds,
                'mode': 'subtype_hierarchy_oracle',
                'oracle_select_field': select_field,
                'patient_logit_ensembles': hierarchy_patient_logit_ensembles,
                'status': 'complete',
            }
        else:
            stage2['pd_subtype'] = {
                'error': 'no overlapping tdrest/pigd_mixed oracle folds for hierarchy aggregation',
                'mode': 'subtype_hierarchy_oracle',
            }
        stage2['_oracle_meta'] = {
            **meta,
            'status': 'complete',
            'group': group,
            'oracle_mode': 'subtype_hierarchy',
            'component_tasks': component_tasks,
        }
        return stage2


    def _run_oracle_stage2(self, exp: Dict[str, Any], exp_dir: str, parent_ckpt: str,
                           summary: Dict[str, Any]) -> Dict[str, Any]:
        group = exp['_group']
        candidates = copy.deepcopy(exp.get('oracle_experiments') or [])
        if not candidates:
            summary['stage2_results']['_oracle_meta'] = {'error': 'oracle_experiments is empty'}
            self._save_summary(summary, exp_dir)
            return summary
        child_output_dir = os.path.join(exp_dir, 'oracle_candidates')
        os.makedirs(child_output_dir, exist_ok=True)
        for child in candidates:
            child['parent_ckpt_override'] = parent_ckpt
            child['stages'] = [2]
        summary['stage2_results']['_oracle_meta'] = {
            'status': 'running',
            'started': datetime.now().isoformat(),
            'candidates': [c.get('_group') for c in candidates],
            'tasks': exp.get('stage2_tasks', []),
        }
        self._save_summary(summary, exp_dir)
        oracle_gpus = list(exp.get('_oracle_available_gpus') or self.AVAILABLE_GPUS)
        print(f"[oracle][{group}] running {len(candidates)} candidate fdiff recipes in parallel "
              f"on gpus={oracle_gpus}")
        if self.PARALLEL_EXPERIMENTS and oracle_gpus:
            child_summaries = self._run_parallel(candidates, child_output_dir, oracle_gpus,
                                            concurrency_per_gpu=self.CONCURRENCY_PER_GPU)
        else:
            child_summaries = []
            for child in candidates:
                print(f"\n[oracle][{group}] -> {child['_group']}: {child.get('_desc','')}")
                child_summaries.append(self.run_experiment(child, child_output_dir))
        summary['stage2_results'].update(
            (self._aggregate_subtype_hierarchy_oracle(
                group,
                child_summaries,
                exp.get('oracle_select_field', 'best_val_acc'),
            ) if exp.get('oracle_mode') == 'subtype_hierarchy' else self._aggregate_oracle_results(
                group,
                child_summaries,
                exp.get('stage2_tasks', []),
                exp.get('oracle_select_field', 'best_val_acc'),
                exp.get('auto_patient_logit_ensemble', True),
            ))
        )
        self._save_summary(summary, exp_dir)
        self._save_patient_logit_ensemble_outputs(summary, exp_dir)
        return summary


    def _save_summary(self, summary: Dict[str, Any], exp_dir: str):
        with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)


    def _save_patient_logit_ensemble_outputs(self, summary: Dict[str, Any], exp_dir: str):
        stage2 = summary.get('stage2_results') or {}
        payload: Dict[str, Any] = {}
        flat_rows: List[Dict[str, Any]] = []
        for task, result in stage2.items():
            if str(task).startswith('_') or not isinstance(result, dict):
                continue
            ensembles = result.get('patient_logit_ensembles')
            if not isinstance(ensembles, dict) or ensembles.get('status') != 'complete':
                continue
            payload[str(task)] = ensembles
            for method in ensembles.get('methods') or []:
                if not isinstance(method, dict):
                    continue
                flat_rows.append({
                    'task': task,
                    'method': method.get('method'),
                    'practical': method.get('practical'),
                    'n_patients_oof': method.get('n_patients_oof'),
                    'accuracy': method.get('accuracy'),
                    'balanced_accuracy': method.get('balanced_accuracy'),
                    'mae': method.get('mae'),
                    'fold_accuracy_mean': method.get('fold_accuracy_mean'),
                    'fold_accuracy_std': method.get('fold_accuracy_std'),
                    'per_class_accuracy': json.dumps(method.get('per_class_accuracy') or {}, sort_keys=True),
                })
        if not payload:
            return
        json_path = os.path.join(exp_dir, 'patient_logit_ensembles.json')
        csv_path = os.path.join(exp_dir, 'patient_logit_ensembles.csv')
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['task', 'method', 'practical', 'n_patients_oof', 'accuracy',
                          'balanced_accuracy', 'mae', 'fold_accuracy_mean',
                          'fold_accuracy_std', 'per_class_accuracy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_rows:
                writer.writerow(row)



def run_paper_experiment(exp, output_dir, device='cpu', num_workers=0):
    """Run an explicitly configured pipeline in a single assigned GPU/CPU process."""
    recipe = copy.deepcopy(exp)
    settings = recipe.pop('_settings')
    settings.update(PARALLEL_EXPERIMENTS=False, AVAILABLE_GPUS=[], NUM_WORKERS=num_workers)
    for stage in (0, 1, 2):
        recipe.setdefault(f'stage{stage}_overrides', {}).update(device=device, num_workers=num_workers)
    return TopoDiSEExperiment(settings).run_experiment(recipe, output_dir)


if __name__ == '__main__':
    from experiments._execution import single_main
    single_main('experiments.topodise_experiments.main')
