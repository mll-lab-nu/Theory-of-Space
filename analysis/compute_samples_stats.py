#!/usr/bin/env python3
"""
Compute per-sample evaluation statistics from env_data.json files.

Usage:
  python analysis/compute_samples_stats.py --eval-files a.json b.json --cogmap-files c.json --exp-files d.json --samples 0-24
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Set, Tuple
sys.path.insert(0, str(Path(__file__).parent))
from vagen.env.spatial.Base.tos_base.utils.cogmap.correlation import compute_correlation_metrics

SAMPLE_RE = re.compile(r"(?:sample_run|sample_)(\d+)")

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def parse_indices(s: str) -> Set[int]:
    """Parse string like '0-24' or '0,1,5' into a set of integers."""
    res = set()
    if not s:
        return res
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                res.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                res.add(int(part))
            except ValueError:
                continue
    return res

def compute_stats(values: List[float], stdev_mean_values: List[float] | None = None) -> Dict[str, float]:
    """Return {count, mean, stdev, stdev_mean?}."""
    cnt = len(values)
    if cnt == 0:
        stats = {'count': 0, 'mean': float('nan'), 'stdev': float('nan')}
        if stdev_mean_values is not None:
            stats['stdev_mean'] = float('nan')
        return stats
    m = mean(values)
    s = stdev(values) if cnt > 1 else 0.0
    stats = {'count': cnt, 'mean': m, 'stdev': s}
    if stdev_mean_values is not None:
        cnt_sm = len(stdev_mean_values)
        stats['stdev_mean'] = stdev(stdev_mean_values) if cnt_sm > 1 else (0.0 if cnt_sm == 1 else float('nan'))
    return stats

def sample_idx_from_key(key: str) -> int | None:
    m = SAMPLE_RE.search(key)
    return int(m.group(1)) if m else None

def iter_samples(data: Dict[str, Any], sample_indices: Set[int]):
    samples = data.get('samples') or {}
    for k, v in samples.items():
        idx = sample_idx_from_key(k)
        if sample_indices and (idx is None or idx not in sample_indices):
            continue
        if isinstance(v, dict):
            yield k, idx, v

def iter_modality_metrics(sample_val: Dict[str, Any]):
    if not isinstance(sample_val, dict):
        return
    if isinstance(sample_val.get('metrics'), dict):
        yield "default", sample_val['metrics']
        return
    for mod, mod_data in sample_val.items():
        if isinstance(mod_data, dict) and isinstance(mod_data.get('metrics'), dict):
            yield mod, mod_data['metrics']

def read_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def _evaluation_key(eval_mode: str) -> str:
    return 'evaluation' if eval_mode == 'default' else f'evaluation_{eval_mode}'

def collect_eval_metrics(paths: List[Path], sample_indices: Set[int], eval_mode: str) -> Dict[str, Any]:
    data_map: Dict[str, Any] = {}
    eval_key = _evaluation_key(eval_mode)
    for path in paths:
        if not path.exists():
            print(f"Warning: File {path} does not exist, skipping.")
            continue
        data = read_json(path)
        if not data:
            continue
        for s_key, _idx, s_val in iter_samples(data, sample_indices):
            for mod, metrics in iter_modality_metrics(s_val):
                eval_m = (metrics or {}).get(eval_key) or {}
                overall = (eval_m.get('overall') or {}).get('avg_accuracy')
                per_task = eval_m.get('per_task') or {}
                if not isinstance(per_task, dict):
                    per_task = {}
                entry = data_map.setdefault(s_key, {}).setdefault(mod, {'overall': [], 'tasks': {}})
                if is_number(overall):
                    entry['overall'].append(float(overall))
                for t_name, t_val in per_task.items():
                    t_acc = (t_val or {}).get('avg_accuracy') if isinstance(t_val, dict) else None
                    if is_number(t_acc):
                        entry['tasks'].setdefault(t_name, []).append(float(t_acc))
    return data_map

def collect_eval_file_averages(paths: List[Path], sample_indices: Set[int], eval_mode: str) -> Dict[str, Any]:
    file_map: Dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            print(f"Warning: File {path} does not exist, skipping.")
            continue
        data = read_json(path)
        if not data:
            continue
        eval_summary = data.get('eval_summary') or {}
        if eval_mode == 'default':
            group_perf = (eval_summary.get('group_performance') or {})
        else:
            group_perf = ((eval_summary.get('group_performance_by_mode') or {}).get(eval_mode) or {})
        if not isinstance(group_perf, dict):
            continue
        for mod, mod_val in group_perf.items():
            overall = (mod_val or {}).get('avg_accuracy')
            if is_number(overall):
                file_map.setdefault(mod, {}).setdefault('overall', []).append(float(overall))
            task_metrics = (mod_val or {}).get('task_metrics') or {}
            if not isinstance(task_metrics, dict):
                continue
            for t_name, t_val in task_metrics.items():
                t_acc = (t_val or {}).get('accuracy') if isinstance(t_val, dict) else None
                if is_number(t_acc):
                    file_map.setdefault(mod, {}).setdefault('tasks', {}).setdefault(t_name, []).append(float(t_acc))
    return file_map

def collect_scalar_metrics(paths: List[Path], sample_indices: Set[int], extractor) -> Dict[str, Any]:
    data_map: Dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            print(f"Warning: File {path} does not exist, skipping.")
            continue
        data = read_json(path)
        if not data:
            continue
        for s_key, _idx, s_val in iter_samples(data, sample_indices):
            for mod, metrics in iter_modality_metrics(s_val):
                val = extractor(metrics or {})
                if is_number(val):
                    data_map.setdefault(s_key, {}).setdefault(mod, []).append(float(val))
    return data_map

def average_eval_metrics(eval_map: Dict[str, Any], file_avgs: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sample_avgs: Dict[str, Any] = {}
    overall_vals: Dict[str, List[float]] = {}
    task_vals: Dict[str, Dict[str, List[float]]] = {}

    for s_key, mod_map in eval_map.items():
        for mod, content in mod_map.items():
            entry = sample_avgs.setdefault(s_key, {}).setdefault(mod, {'overall': None, 'tasks': {}})
            if content.get('overall'):
                avg = mean(content['overall'])
                entry['overall'] = avg
                overall_vals.setdefault(mod, []).append(avg)
            for t_name, vals in (content.get('tasks') or {}).items():
                if not vals:
                    continue
                t_avg = mean(vals)
                entry['tasks'][t_name] = t_avg
                task_vals.setdefault(mod, {}).setdefault(t_name, []).append(t_avg)

    overall_stats = {}
    for mod, vals in overall_vals.items():
        file_mod = (file_avgs or {}).get(mod) or {}
        file_tasks = file_mod.get('tasks') or {}
        tasks_stats = {
            t: compute_stats(v, file_tasks.get(t))
            for t, v in (task_vals.get(mod) or {}).items()
        }
        overall_stats[mod] = {
            'overall': compute_stats(vals, file_mod.get('overall')),
            'tasks': tasks_stats
        }

    return sample_avgs, overall_stats

def average_scalar_metrics(metric_map: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sample_avgs: Dict[str, Any] = {}
    overall_vals: Dict[str, List[float]] = {}

    for s_key, mod_map in metric_map.items():
        for mod, vals in mod_map.items():
            if not vals:
                continue
            avg = mean(vals)
            sample_avgs.setdefault(s_key, {})[mod] = avg
            overall_vals.setdefault(mod, []).append(avg)

    overall_stats = {mod: compute_stats(vals) for mod, vals in overall_vals.items()}
    return sample_avgs, overall_stats

def map_active_action_cost_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    for mod, val in stats.items():
        if 'vision_active' in mod:
            mapped['vision_active'] = val
        elif 'text_active' in mod:
            mapped['text_active'] = val
    return mapped

def _format_value(prefix: str, val: Any) -> str | None:
    if isinstance(val, dict) and 'mean' in val:
        extra = ""
        if is_number(val.get('stdev_mean')):
            extra = f", SD_mean={val['stdev_mean']:.4f}"
        return f"{prefix} Mean={val['mean']:.4f}, SD={val['stdev']:.4f}{extra}, N={val['count']}"
    if is_number(val):
        return f"{prefix} Avg={float(val):.4f}"
    return None

def format_stats(stats_dict: Dict[str, Any], title: str) -> str:
    lines = [title, '-' * len(title)]
    for key, info in sorted(stats_dict.items()):
        if isinstance(info, dict) and 'overall' in info and 'tasks' in info:
            mod = key
            ov_line = _format_value(f"  {mod:30s} Overall:", info.get('overall'))
            if ov_line:
                lines.append(ov_line)
            for t_name, t_s in sorted((info.get('tasks') or {}).items()):
                t_line = _format_value(f"    {t_name:30s}", t_s)
                if t_line:
                    lines.append(t_line)
        elif isinstance(info, dict) and 'mean' in info and 'stdev' in info:
            line = _format_value(f"  {key:30s}", info)
            if line:
                lines.append(line)
        else:
            lines.append(f"{key}:")
            if isinstance(info, dict):
                for mod, s in sorted(info.items()):
                    if is_number(s) or (isinstance(s, dict) and 'mean' in s):
                        line = _format_value(f"  {mod:30s}", s)
                        if line:
                            lines.append(line)
                        continue
                    if not isinstance(s, dict) or 'overall' not in s:
                        continue
                    ov_line = _format_value(f"  {mod:30s} Overall:", s.get('overall'))
                    if ov_line:
                        lines.append(ov_line)
                    for t_name, t_s in sorted((s.get('tasks') or {}).items()):
                        t_line = _format_value(f"    {t_name:30s}", t_s)
                        if t_line:
                            lines.append(t_line)
    return '\n'.join(lines)

def compute_cogmap_correlation_from_averages(
    eval_avgs: Dict[str, Any],
    cogmap_avgs: Dict[str, Any],
    infogain_avgs: Dict[str, Any],
) -> Dict[str, Any]:
    modality_samples: Dict[str, List[Dict[str, Any]]] = {}
    for s_key, mod_map in cogmap_avgs.items():
        for mod, cog_score in mod_map.items():
            if 'active' not in mod:
                continue
            eval_m = (eval_avgs.get(s_key) or {}).get(mod) or {}
            overall = eval_m.get('overall')
            tasks = eval_m.get('tasks') or {}
            if not is_number(cog_score) or (overall is None and not tasks):
                continue
            metrics = {
                'cogmap': {
                    'exploration': {
                        'correctness': {'last_global_vs_gt_full': {'overall': float(cog_score)}}
                    }
                },
                'evaluation': {}
            }
            if is_number(overall):
                metrics['evaluation']['overall'] = {'avg_accuracy': float(overall)}
            if tasks:
                metrics['evaluation']['per_task'] = {
                    t: {'avg_accuracy': float(v)} for t, v in tasks.items() if is_number(v)
                }
            infogain = (infogain_avgs.get(s_key) or {}).get(mod)
            if is_number(infogain):
                metrics['exploration'] = {'final_information_gain': float(infogain)}
            modality_samples.setdefault(mod, []).append({
                'metrics': metrics,
                'sample_idx': sample_idx_from_key(s_key),
            })

    all_results = {}
    for mod, env_data_list in modality_samples.items():
        corr = compute_correlation_metrics(env_data_list, exp_type='active')
        corr.update({
            'samples_averaged': len(env_data_list),
            'sample_indices': sorted({s.get('sample_idx') for s in env_data_list if s.get('sample_idx') is not None}),
        })
        all_results[mod] = corr

    return {'modalities': all_results, 'num_modalities': len(all_results)}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-files', nargs='*', default=[], help='paths to evaluation env_data.json files')
    ap.add_argument('--cogmap-files', nargs='*', default=[], help='paths to cogmap env_data.json files')
    ap.add_argument('--exp-files', nargs='*', default=[], help='paths to exploration env_data.json files')
    ap.add_argument('--samples', default='0-24', help='Range of samples to process (e.g. 0-24, 0,1,5)')
    ap.add_argument('--save-json', '-o', help='optional path to save JSON summary')
    ap.add_argument('--eval-mode', default='default', help='evaluation mode to read (default or custom)')
    ap.add_argument('--compute-cogmap-correlation', action='store_true', 
                    help='Compute cogmap correlation across paths')
    args = ap.parse_args(argv)

    sample_indices = parse_indices(args.samples)
    
    eval_files = [Path(p) for p in args.eval_files]
    cogmap_files = [Path(p) for p in args.cogmap_files]
    infogain_files = [Path(p) for p in args.exp_files]

    if not (eval_files or cogmap_files or infogain_files):
        print("No input files provided.")
        return

    eval_raw = collect_eval_metrics(eval_files, sample_indices, args.eval_mode)
    eval_file_avgs = collect_eval_file_averages(eval_files, sample_indices, args.eval_mode)
    eval_avgs, eval_stats = average_eval_metrics(eval_raw, eval_file_avgs)
    cogmap_raw = collect_scalar_metrics(
        cogmap_files, sample_indices,
        lambda m: (((m.get('cogmap') or {}).get('exploration') or {}).get('correctness') or {})
            .get('last_global_vs_gt_full', {}).get('overall')
    )
    cogmap_avgs, cogmap_stats = average_scalar_metrics(cogmap_raw)
    infogain_raw = collect_scalar_metrics(
        infogain_files, sample_indices,
        lambda m: (m.get('exploration') or {}).get('final_information_gain')
    )
    infogain_avgs, infogain_stats = average_scalar_metrics(infogain_raw)

    action_cost_stats = {}
    if eval_avgs:
        print('\n' + format_stats(eval_avgs, "Evaluation Averages (per sample)"))
        print('\n' + format_stats(eval_stats, "Evaluation Stats (across samples)"))
        if not args.compute_cogmap_correlation:
            action_cost_raw = collect_scalar_metrics(
                eval_files, sample_indices,
                lambda m: (m.get('exploration') or {}).get('action_cost')
            )
            _action_cost_avgs, action_cost_stats = average_scalar_metrics(action_cost_raw)
            action_cost_stats = map_active_action_cost_stats(action_cost_stats)
            if action_cost_stats:
                print('\n' + format_stats(action_cost_stats, "Avg Action Cost Stats (across samples)"))
    if cogmap_avgs:
        print('\n' + format_stats(cogmap_avgs, "Cogmap Averages (per sample)"))
        print('\n' + format_stats(cogmap_stats, "Cogmap Stats (across samples)"))
    if infogain_avgs:
        print('\n' + format_stats(infogain_avgs, "Info Gain Averages (per sample)"))
        print('\n' + format_stats(infogain_stats, "Info Gain Stats (across samples)"))

    corr_results = None
    if args.compute_cogmap_correlation:
        print("\nComputing cogmap correlation across averaged samples...")
        corr_results = compute_cogmap_correlation_from_averages(eval_avgs, cogmap_avgs, infogain_avgs)
        print("\n" + "=" * 80)
        print(f"COGMAP CORRELATION RESULTS ({corr_results.get('num_modalities', 0)} modalities)")
        print("=" * 80)
        for mod_name, corr_res in corr_results.get('modalities', {}).items():
            print(f"\n{'=' * 80}\nMODALITY: {mod_name}\n{'=' * 80}")
            print(f"Samples: {corr_res.get('samples_averaged', 0)} | "
                  f"Valid: {corr_res.get('n_samples', 0)}")
            for task, data in (corr_res.get('cogmap_acc_correlations') or {}).items():
                if isinstance(data, dict) and data.get('pearson_r') is not None:
                    sig = " *" if data.get('significant', False) else ""
                    print(f"  {task:40s}: r={data['pearson_r']:7.4f}, "
                          f"p={data['p_value']:7.4f}, n={data['n_samples']}{sig}")
            data = corr_res.get('cogmap_infogain_correlation')
            if isinstance(data, dict) and data.get('pearson_r') is not None:
                sig = " *" if data.get('significant', False) else ""
                print(f"  {'Information Gain':40s}: r={data['pearson_r']:7.4f}, "
                      f"p={data['p_value']:7.4f}, n={data['n_samples']}{sig}")

    if args.save_json:
        out_obj = {
            'eval_files': [str(p) for p in eval_files],
            'cogmap_files': [str(p) for p in cogmap_files],
            'infogain_files': [str(p) for p in infogain_files],
            'evaluation': {'sample_avgs': eval_avgs, 'stats': eval_stats},
            'cogmap': {'sample_avgs': cogmap_avgs, 'stats': cogmap_stats},
            'infogain': {'sample_avgs': infogain_avgs, 'stats': infogain_stats},
            'avg_action_cost': {'stats': action_cost_stats} if action_cost_stats else {},
            'correlation': corr_results,
        }
        Path(args.save_json).write_text(json.dumps(out_obj, indent=2))
        print(f"Saved JSON summary to {args.save_json}")

if __name__ == '__main__':
    main()
