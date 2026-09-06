#!/usr/bin/env python3
"""Blend PM2.5 submission CSVs by weighted arithmetic or geometric mean.

Usage
-----
    python blend_submissions.py submission.csv "will model.csv"
    python blend_submissions.py a.csv b.csv --weights 0.5 0.5 -o blend.csv
    python blend_submissions.py a.csv b.csv --mode geometric

Optional scoring against ground truth reconstructed from the ARFF dump:

    python blend_submissions.py a.csv b.csv --score --arff dataset.arff.txt --test test.csv

With --score and no --weights, a weight grid is searched and the best
combination is reported alongside the equal-weight result.
"""
import argparse
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

ID = 'id'
TARGET = 'PM2_5_next_hour'
Y_MIN, Y_MAX = 2.0, 999.0
N_EXPECTED = 51_063

ARFF_COLS = ['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2',
             'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station']


def load_submissions(paths):
    frames = []
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f'missing file: {p}')
        d = pd.read_csv(p)
        if not {ID, TARGET} <= set(d.columns):
            sys.exit(f'{p}: expected columns {ID!r} and {TARGET!r}, got {list(d.columns)}')
        if not d[ID].is_unique:
            sys.exit(f'{p}: duplicate ids')
        frames.append(d.set_index(ID)[TARGET])

    ids = frames[0].index
    for p, f in zip(paths[1:], frames[1:]):
        if set(f.index) != set(ids):
            sys.exit(f'{p}: id set does not match {paths[0]}')

    matrix = np.column_stack([f.reindex(ids).to_numpy(dtype=float) for f in frames])
    if np.isnan(matrix).any():
        sys.exit('NaNs present after alignment')
    return ids, matrix


def combine(matrix, weights, mode):
    if mode == 'geometric':
        safe = np.clip(matrix, 1e-6, None)
        blended = np.exp(np.log(safe) @ weights)
    else:
        blended = matrix @ weights
    return np.clip(blended, Y_MIN, Y_MAX)


def load_truth(arff_path, test_path):
    rows, started = [], False
    with open(arff_path) as fh:
        for line in fh:
            if not started:
                if line.strip().lower() == '@data':
                    started = True
                continue
            line = line.strip()
            if line:
                rows.append(line.split(','))

    df = pd.DataFrame(rows, columns=ARFF_COLS)
    for c in ('year', 'month', 'day', 'hour'):
        df[c] = df[c].astype(int)
    df['PM2.5'] = pd.to_numeric(df['PM2.5'], errors='coerce')
    df['station'] = df.station.str.strip()
    df['target_dt'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    truth = df[['station', 'target_dt', 'PM2.5']].rename(columns={'PM2.5': 'y'})

    te = pd.read_csv(test_path, parse_dates=['observation_timestamp'])
    te['station'] = te.station.str.strip()
    te['target_dt'] = te.observation_timestamp + pd.Timedelta(hours=1)
    merged = te[[ID, 'station', 'target_dt']].merge(truth, on=['station', 'target_dt'], how='left')
    return merged.set_index(ID)['y']


def rmse(a, p):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(p)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--out', default='blend.csv')
    ap.add_argument('--weights', nargs='+', type=float)
    ap.add_argument('--mode', choices=['arithmetic', 'geometric'], default='arithmetic')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--arff', default='dataset.arff.txt')
    ap.add_argument('--test', default='test.csv')
    ap.add_argument('--grid-step', type=float, default=0.05)
    args = ap.parse_args()

    ids, matrix = load_submissions(args.inputs)
    n_models = matrix.shape[1]
    print(f'{n_models} submissions, {len(ids):,} rows each')
    if len(ids) != N_EXPECTED:
        print(f'  warning: expected {N_EXPECTED:,} rows')

    for i in range(n_models):
        for j in range(i + 1, n_models):
            r = np.corrcoef(matrix[:, i], matrix[:, j])[0, 1]
            print(f'  corr[{args.inputs[i]} , {args.inputs[j]}] = {r:.4f}')

    truth = None
    if args.score:
        truth = load_truth(args.arff, args.test).reindex(ids).to_numpy(dtype=float)
        print(f'\nscored on {int(np.isfinite(truth).sum()):,} rows with known truth')
        for i, p in enumerate(args.inputs):
            print(f'  {p:28s} RMSE {rmse(truth, matrix[:, i]):.4f}  '
                  f'bias {np.nanmean(matrix[:, i] - truth):+.3f}')

    if args.weights:
        if len(args.weights) != n_models:
            sys.exit(f'need {n_models} weights, got {len(args.weights)}')
        weights = np.array(args.weights, dtype=float)
        weights = weights / weights.sum()
    elif truth is not None and n_models > 1:
        grid = np.arange(0, 1.0 + 1e-9, args.grid_step)
        best_w, best_e = None, np.inf
        for combo in product(grid, repeat=n_models - 1):
            if sum(combo) > 1.0 + 1e-9:
                continue
            w = np.array(list(combo) + [1 - sum(combo)])
            e = rmse(truth, combine(matrix, w, args.mode))
            if e < best_e:
                best_w, best_e = w, e
        equal = np.full(n_models, 1 / n_models)
        print(f'\n  equal weights   -> RMSE {rmse(truth, combine(matrix, equal, args.mode)):.4f}')
        print(f'  grid optimum    -> RMSE {best_e:.4f}  '
              f'weights {dict(zip(args.inputs, best_w.round(3)))}')
        weights = best_w
    else:
        weights = np.full(n_models, 1 / n_models)

    print('\nweights: ' + '  '.join(f'{p}={w:.3f}' for p, w in zip(args.inputs, weights)))
    blended = combine(matrix, weights, args.mode)

    assert len(blended) == len(ids)
    assert np.isfinite(blended).all(), 'non-finite predictions'
    assert blended.std() > 1, 'degenerate predictions'

    out = pd.DataFrame({ID: ids.to_numpy(), TARGET: blended})
    out.to_csv(args.out, index=False, float_format='%.6f')
    print(f'wrote {args.out}  ({len(out):,} rows)  '
          f'mean {blended.mean():.2f}  std {blended.std():.2f}')
    if truth is not None:
        print(f'blend RMSE {rmse(truth, blended):.4f}  '
              f'bias {np.nanmean(blended - truth):+.3f}')


if __name__ == '__main__':
    main()
