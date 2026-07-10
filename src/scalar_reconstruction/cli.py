from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, config_to_dict
from .io_utils import write_json_atomic
from .jobs import make_jobs_from_design, requeue_failed, requeue_orphaned_running, runtime_paths
from .data import audit_market
from .design_seed import make_seed_design
from .design_ioptimal import make_ioptimal_design
from .design_spacefill import make_spacefill_design
from .design_union import make_union
from .controls import make_null_control_design
from .aggregate import collect_results, summarize_field, noise_summary, bootstrap_summary
from .monitor import print_status
from .worker import run_worker
from .surrogate import write_prediction_grid
from .acquisition_imspe import make_imspe_actions
from .acquisition_levelset import make_levelset_actions
from .proxy_distance import compute_proxy_distances
from .envelopes import envelope_tables, posterior_volume_ci
from .cross_market import cross_market_correlations, budget_deformation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Noisy scalar transfer-field reconstruction')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('init-run'); s.add_argument('--config', required=True)
    s = sub.add_parser('audit-data'); s.add_argument('--config', required=True)
    s = sub.add_parser('make-seed-design'); s.add_argument('--config', required=True)
    s = sub.add_parser('make-ioptimal-design'); s.add_argument('--config', required=True); s.add_argument('--n', type=int, default=None)
    s = sub.add_parser('make-spacefill-design'); s.add_argument('--config', required=True); s.add_argument('--n', type=int, default=None)
    s = sub.add_parser('make-controls'); s.add_argument('--config', required=True)
    s = sub.add_parser('make-union'); s.add_argument('--config', required=True); s.add_argument('--inputs', required=True, help='comma-separated design CSVs'); s.add_argument('--out', default=None)
    s = sub.add_parser('make-jobs'); s.add_argument('--config', required=True); s.add_argument('--design_csv', required=True); s.add_argument('--stage', required=True); s.add_argument('--markets', default=None); s.add_argument('--budgets', default=None); s.add_argument('--reps', type=int, default=None); s.add_argument('--eval_split', default='val')
    s = sub.add_parser('worker'); s.add_argument('--config', required=True); s.add_argument('--worker_id', required=True); s.add_argument('--threads', type=int, default=None); s.add_argument('--max_jobs', type=int, default=None)
    s = sub.add_parser('monitor'); s.add_argument('--config', required=True); s.add_argument('--top_k', type=int, default=8); s.add_argument('--no_summary', action='store_true')
    s = sub.add_parser('aggregate'); s.add_argument('--config', required=True)
    s = sub.add_parser('bootstrap-ci'); s.add_argument('--config', required=True); s.add_argument('--n_boot', type=int, default=2000)
    s = sub.add_parser('fit-surrogate'); s.add_argument('--config', required=True); s.add_argument('--field_csv', default=None); s.add_argument('--metric', default=None)
    s = sub.add_parser('make-imspe-actions'); s.add_argument('--config', required=True); s.add_argument('--field_csv', default=None); s.add_argument('--n_actions', type=int, default=24); s.add_argument('--metric', default=None); s.add_argument('--explore_frac', type=float, default=0.6)
    s = sub.add_parser('make-levelset-actions'); s.add_argument('--config', required=True); s.add_argument('--field_csv', default=None); s.add_argument('--n_actions', type=int, default=None); s.add_argument('--metric', default=None); s.add_argument('--acquisition', default='straddle')
    s = sub.add_parser('proxy-distances'); s.add_argument('--config', required=True); s.add_argument('--theta_csv', required=True)
    s = sub.add_parser('envelopes'); s.add_argument('--config', required=True); s.add_argument('--prediction_csv', default=None)
    s = sub.add_parser('posterior-volume-ci'); s.add_argument('--config', required=True); s.add_argument('--field_csv', default=None); s.add_argument('--n_samples', type=int, default=500)
    s = sub.add_parser('cross-market'); s.add_argument('--config', required=True); s.add_argument('--prediction_csv', default=None)
    s = sub.add_parser('requeue-failed'); s.add_argument('--config', required=True); s.add_argument('--max_retries', type=int, default=None)
    s = sub.add_parser('requeue-orphaned'); s.add_argument('--config', required=True); s.add_argument('--older_than_s', type=float, default=600.0)
    return p


def _split_list(s: str | None, cast=str):
    if s is None or s == '': return None
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == 'init-run':
        cfg = load_config(args.config); paths = runtime_paths(cfg)
        write_json_atomic(paths['run_dir'] / 'config_resolved.json', config_to_dict(cfg))
        print(paths['run_dir'])
    elif args.cmd == 'audit-data':
        cfg = load_config(args.config); paths = runtime_paths(cfg); rows = []
        for m in cfg.markets:
            rec = audit_market(m, cfg.data); rows.append(rec)
            write_json_atomic(paths['targets_dir'] / f'{m.name}_signature.json', rec)
        import pandas as pd
        out = paths['tables_dir'] / 'data_audit.csv'; pd.DataFrame(rows).to_csv(out, index=False); print(out)
    elif args.cmd == 'make-seed-design':
        cfg = load_config(args.config); print(make_seed_design(cfg, runtime_paths(cfg)['design_dir']))
    elif args.cmd == 'make-ioptimal-design':
        cfg = load_config(args.config); print(make_ioptimal_design(cfg, runtime_paths(cfg)['design_dir'], n=args.n))
    elif args.cmd == 'make-spacefill-design':
        cfg = load_config(args.config); print(make_spacefill_design(cfg, runtime_paths(cfg)['design_dir'], n=args.n))
    elif args.cmd == 'make-controls':
        cfg = load_config(args.config); print(make_null_control_design(cfg, runtime_paths(cfg)['design_dir']))
    elif args.cmd == 'make-union':
        cfg = load_config(args.config); out = args.out or str(runtime_paths(cfg)['design_dir'] / 'theta_union.csv'); print(make_union([x.strip() for x in args.inputs.split(',')], out))
    elif args.cmd == 'make-jobs':
        print(make_jobs_from_design(args.config, args.design_csv, args.stage, markets=_split_list(args.markets), budgets=_split_list(args.budgets, int), default_replicates=args.reps, eval_split=args.eval_split))
    elif args.cmd == 'worker':
        run_worker(args.config, args.worker_id, threads=args.threads, max_jobs=args.max_jobs)
    elif args.cmd == 'monitor':
        print_status(args.config, top_k=args.top_k, summarize=not args.no_summary)
    elif args.cmd == 'aggregate':
        df = collect_results(args.config); print(f'field rows: {len(df)}'); print(summarize_field(args.config)); print(noise_summary(args.config))
    elif args.cmd == 'bootstrap-ci':
        print(bootstrap_summary(args.config, n_boot=args.n_boot))
    elif args.cmd == 'fit-surrogate':
        cfg = load_config(args.config); paths = runtime_paths(cfg); field_csv = args.field_csv or str(paths['tables_dir'] / 'field_observations.csv'); print(write_prediction_grid(field_csv, cfg, paths['tables_dir'], metric=args.metric))
    elif args.cmd == 'make-imspe-actions':
        cfg = load_config(args.config); paths = runtime_paths(cfg); field_csv = args.field_csv or str(paths['tables_dir'] / 'field_observations.csv'); print(make_imspe_actions(field_csv, cfg, paths['design_dir'], n_actions=args.n_actions, metric=args.metric, explore_frac=args.explore_frac))
    elif args.cmd == 'make-levelset-actions':
        cfg = load_config(args.config); paths = runtime_paths(cfg); field_csv = args.field_csv or str(paths['tables_dir'] / 'field_observations.csv'); print(make_levelset_actions(field_csv, cfg, paths['design_dir'], n_actions=args.n_actions, metric=args.metric, acquisition=args.acquisition))
    elif args.cmd == 'proxy-distances':
        cfg = load_config(args.config); print(compute_proxy_distances(args.config, args.theta_csv, runtime_paths(cfg)['tables_dir']))
    elif args.cmd == 'envelopes':
        cfg = load_config(args.config); paths = runtime_paths(cfg); pred = args.prediction_csv or str(paths['tables_dir'] / 'surrogate_predictions.csv'); print(envelope_tables(args.config, pred))
    elif args.cmd == 'posterior-volume-ci':
        cfg = load_config(args.config); paths = runtime_paths(cfg); fcsv = args.field_csv or str(paths['tables_dir'] / 'field_observations.csv'); print(posterior_volume_ci(args.config, fcsv, n_samples=args.n_samples))
    elif args.cmd == 'cross-market':
        cfg = load_config(args.config); paths = runtime_paths(cfg); pred = args.prediction_csv or str(paths['tables_dir'] / 'surrogate_predictions.csv'); print(cross_market_correlations(args.config, pred)); print(budget_deformation(args.config, pred))
    elif args.cmd == 'requeue-failed':
        print(requeue_failed(args.config, max_retries=args.max_retries))
    elif args.cmd == 'requeue-orphaned':
        print(requeue_orphaned_running(args.config, older_than_s=args.older_than_s))

if __name__ == '__main__':
    main()
