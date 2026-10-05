"""Nonlinear gamma sweep for IPW (Hájek-type weighted mean difference) with selectable weights.

真のアウトカムモデルに gamma * x1^2 を加えた DGP(両群共通、真の効果は 1 のまま)で、
過去対照群を現在試験の共変量分布へ輸送する重み付き平均差

    Δ̂ = ȳ_c − Σ_j w_j y_j^h / Σ_j w_j

を評価する。重み w は weight.py の compute_weights で選ぶ:
    propensity     : ロジスティック傾向スコアのオッズ e/(1-e)(--propensity-penalty で罰則を選択)
    ulsif          : uLSIF の密度比(IW-Learn / AIPW-Learn と同一手順)
    uniform        : 一様(輸送なしの平均差)
    oracle         : 真のガウス分布からの密度比
    propensity_ate : 旧実装(target 1/e, source 1/(1-e))

IW-Learn の scripts/nonlinear_gamma_experiment.py と同一のシード・乱数消費順序で
データを生成するため、trial 番号で突き合わせれば IWL と同一データのペア比較になる。
計算はこのリポジトリ内で完結する(IW-Learn への依存なし)。

出力(IW-Learn の summary.csv と同じ列 + 重み診断):
    results/<out-dir>/trials_<scenario>_N<n_h>_gamma<g>.csv
    results/<out-dir>/summary.csv

Usage:
    .venv/bin/python scripts/nonlinear_gamma_ipw.py --weight-method propensity     # 既定
    .venv/bin/python scripts/nonlinear_gamma_ipw.py --weight-method ulsif --out-dir nonlinear_gamma_target50_ulsif
    .venv/bin/python scripts/nonlinear_gamma_ipw.py --scenarios data_scenario_2 --trials 200 --bootstraps 200
    .venv/bin/python scripts/nonlinear_gamma_ipw.py --smoke
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from weight import (  # noqa: E402
    WEIGHT_METHODS,
    compute_weights,
    effective_sample_size,
    weighted_mean_difference,
)

# numpy 2.0 + Accelerate の matmul 誤検知警告を抑える(値は正常)
warnings.filterwarnings("ignore", message=".*encountered in matmul")

TRUE_VALUE = 1.0
DEFAULT_TARGET_N = 50
METHOD_LABEL = "IPW"
PREFIX = "ipw"

# ---- IW-Learn scripts/nonlinear_gamma_experiment.py と同一のデータ生成 ----
MEANS = {
    "data_scenario_1": {"target": [0.0, 1.0], "source": [0.0, 0.0]},
    "data_scenario_2": {"target": [2.0, 3.0], "source": [0.0, 0.0]},
    "data_scenario_3": {"target": [0.0, 1.0], "source": [0.0, 0.0]},
}
COVS = {
    "data_scenario_1": {
        "target": [[1.5, -0.7], [-0.7, 1.5]],
        "source": [[1.7, 0.7], [0.7, 1.7]],
    },
    "data_scenario_2": {
        "target": [[1.5, -0.7], [-0.7, 1.5]],
        "source": [[1.7, 0.7], [0.7, 1.7]],
    },
    "data_scenario_3": {
        "target": [[1.5, -0.7], [-0.7, 1.5]],
        "source": [[3.5, 0.7], [0.7, 3.5]],
    },
}
COEFFICIENT = np.array([1.5, -2.0, 0.8])  # [intercept, x1, x2]


def generate_data(scenario, source_n, gamma, target_n):
    """IW-Learn main.py / nonlinear_gamma_experiment.py と同一の乱数消費順序。"""
    mean = MEANS[scenario]
    cov = COVS[scenario]

    x_t = np.random.multivariate_normal(mean["target"], cov["target"], target_n)
    target_x = np.insert(x_t, 0, 1, axis=1)
    eps_t = np.random.normal(0, 1, target_n)
    y_t = (
        COEFFICIENT @ target_x.T
        + gamma * target_x[:, 1] ** 2
        + TRUE_VALUE
        + eps_t
    )
    target_y = y_t.reshape(-1, 1)

    x_s = np.random.multivariate_normal(mean["source"], cov["source"], source_n)
    source_x = np.insert(x_s, 0, 1, axis=1)
    eps_s = np.random.normal(0, 1, source_n)
    y_s = COEFFICIENT @ source_x.T + gamma * source_x[:, 1] ** 2 + eps_s
    source_y = y_s.reshape(-1, 1)

    return target_x, target_y, source_x, source_y
# ---------------------------------------------------------------------------


def _true_dist(scenario):
    means = {k: np.array(v) for k, v in MEANS[scenario].items()}
    covs = {k: np.array(v) for k, v in COVS[scenario].items()}
    return means, covs


def _estimate(tx, ty, sx, sy, method, means, covs, wkw):
    wt, ws = compute_weights(method, tx, sx, means=means, covariances=covs, **wkw)
    return weighted_mean_difference(wt, ty, ws, sy), ws


def run_one_trial(i, scenario, source_n, gamma, n_bootstraps, target_n, method, wkw):
    random.seed(i)
    np.random.seed(i)
    target_x, target_y, source_x, source_y = generate_data(
        scenario, source_n, gamma, target_n
    )
    means, covs = _true_dist(scenario)
    est, ws = _estimate(target_x, target_y, source_x, source_y, method, means, covs, wkw)

    # target・source とも復元抽出し、重み(傾向スコア / uLSIF の σ, λ 選択)を反復ごとに推定し直す
    rng = np.random.default_rng(i)
    boot = []
    for _ in range(n_bootstraps):
        t_idx = rng.choice(target_n, size=target_n, replace=True)
        s_idx = rng.choice(source_n, size=source_n, replace=True)
        try:
            b, _ = _estimate(target_x[t_idx], target_y[t_idx],
                             source_x[s_idx], source_y[s_idx], method, means, covs, wkw)
        except Exception:
            continue
        if np.isfinite(b):
            boot.append(b)
    boot = np.array(boot)
    return {
        "trial": i,
        f"{PREFIX}_estimate": float(est),
        f"{PREFIX}_boot_se": float(boot.std(ddof=1)),
        f"{PREFIX}_ci_lower": float(np.percentile(boot, 2.5)),
        f"{PREFIX}_ci_upper": float(np.percentile(boot, 97.5)),
        f"{PREFIX}_boot_n": int(boot.size),
        f"{PREFIX}_ess": effective_sample_size(ws),
        f"{PREFIX}_max_weight": float(np.max(ws)),
    }


def summarize(df, prefix):
    est = df[f"{prefix}_estimate"]
    lo = df[f"{prefix}_ci_lower"]
    hi = df[f"{prefix}_ci_upper"]
    stats = {
        "mean": est.mean(),
        "bias": est.mean() - TRUE_VALUE,
        "sd": est.std(ddof=1),
        "rmse": float(np.sqrt(((est - TRUE_VALUE) ** 2).mean())),
        "coverage": float(((lo <= TRUE_VALUE) & (TRUE_VALUE <= hi)).mean()),
        "ci_width": float((hi - lo).mean()),
    }
    se = df[f"{prefix}_boot_se"]
    nlo, nhi = est - 1.96 * se, est + 1.96 * se
    stats["coverage_normal"] = float(((nlo <= TRUE_VALUE) & (TRUE_VALUE <= nhi)).mean())
    stats["ci_width_normal"] = float((nhi - nlo).mean())
    stats["boot_n_mean"] = float(df[f"{prefix}_boot_n"].mean())
    stats["boot_n_min"] = int(df[f"{prefix}_boot_n"].min())
    stats["ess_median"] = float(df[f"{prefix}_ess"].median())
    stats["max_weight_mean"] = float(df[f"{prefix}_max_weight"].mean())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET_N)
    parser.add_argument("--source-ns", type=int, nargs="+",
                        default=[20, 50, 100, 200, 500])
    parser.add_argument("--scenarios", type=str, nargs="+",
                        default=["data_scenario_1", "data_scenario_2", "data_scenario_3"])
    parser.add_argument("--weight-method", choices=WEIGHT_METHODS, default="propensity")
    parser.add_argument("--propensity-penalty", choices=["l2", "none"], default="l2")
    parser.add_argument("--propensity-C", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="results/ 配下の出力ディレクトリ名。既定は nonlinear_gamma_target<N>_<weight>")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from joblib import Parallel, delayed

    scenarios, source_ns, gammas = args.scenarios, args.source_ns, args.gammas
    n_trials, n_boot = args.trials, args.bootstraps
    if args.smoke:
        scenarios, source_ns, gammas = ["data_scenario_2"], [100], [0.5]
        n_trials, n_boot = 4, 20
    wkw = {"propensity_penalty": args.propensity_penalty, "propensity_C": args.propensity_C}

    out_dir = PROJECT_ROOT / "results" / (
        args.out_dir or f"nonlinear_gamma_target{args.target_n}_{args.weight_method}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"

    # scenario 2(シフト最大)を先に回す(IW-Learn と同じ順序)
    conditions = [
        (sc, sn, g)
        for sc in ["data_scenario_2", "data_scenario_1", "data_scenario_3"]
        if sc in scenarios
        for g in gammas
        for sn in source_ns
    ]

    summary_rows = []
    for cond_i, (sc, sn, g) in enumerate(conditions, start=1):
        t0 = time.time()
        csv_path = out_dir / f"trials_{sc}_N{sn}_gamma{g:g}.csv"
        df = None
        if csv_path.exists():
            cached = pd.read_csv(csv_path)
            if len(cached) == n_trials and f"{PREFIX}_boot_se" in cached.columns:
                df = cached
        status = "skip (cached)" if df is not None else "done"
        if df is None:
            print(f"[{cond_i}/{len(conditions)}] start {sc} N{sn} gamma={g:g} "
                  f"({n_trials} trials x B={n_boot}, weights={args.weight_method})", flush=True)
            rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=args.verbose)(
                delayed(run_one_trial)(i, sc, sn, g, n_boot, args.target_n,
                                       args.weight_method, wkw)
                for i in range(n_trials)
            )
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
        summary_rows.append({"scenario": sc, "source_n": sn, "gamma": g,
                             "method": METHOD_LABEL, "weight_method": args.weight_method,
                             "propensity_penalty": args.propensity_penalty,
                             **summarize(df, PREFIX)})
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        s = summary_rows[-1]
        print(f"[{cond_i}/{len(conditions)}] {status} {sc} N{sn} gamma={g:g} "
              f"bias={s['bias']:+.3f} cov={s['coverage']:.3f} ess={s['ess_median']:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(summary_path)


if __name__ == "__main__":
    main()
