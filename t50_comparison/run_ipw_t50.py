"""IPW (propensity-score odds weighting) comparator at target_N=50.

推定対象は現在試験集団における治療効果の差 Δ_c = E[Y(1) - Y(0) | R=1]。
現在試験(R=1)は重みなし、過去対照群(R=0)には傾向スコアのオッズ
    w(x) = e(x) / (1 - e(x)),  e(x) = P(R=1 | x)
を掛けて現在試験の共変量分布へ輸送する(Hájek 型の自己正規化):

    Δ̂_IPW = mean(y^c) - Σ_j w_j y_j^h / Σ_j w_j

傾向スコアはロジスティック回帰(最尤、数値安定化のための微小 ridge)。
- "linear": 切片 + x1 + x2                      (主解析)
- "quad":   切片 + x1 + x2 + x1^2 + x2^2 + x1x2 (感度解析。2つのガウス分布の
            対数密度比は2次式なので、こちらが正しく特定されたモデル)

データ生成は scripts/nonlinear_gamma_experiment.py と同一の乱数消費順序
(同じ trial 番号 = 同じデータ)。区間は全パイプライン bootstrap(B=1000)の
パーセンタイル法(主)と bootstrap SE による正規近似(副)。

Usage:
    .venv/bin/python scripts/run_ipw_t50.py            # full run
    .venv/bin/python scripts/run_ipw_t50.py --smoke    # timing check
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nonlinear_gamma_experiment import generate_data  # noqa: E402

TRUE_VALUE = 1.0
TARGET_N = 50
SOURCE_NS = [20, 50, 100, 200, 500]
SCENARIOS = ["data_scenario_1", "data_scenario_2", "data_scenario_3"]
GAMMAS = [0.0, 0.25, 0.5]
N_TRIAL = 1000
N_BOOT = 1000
RIDGE = 1e-4          # 非切片係数への微小 L2(分離時の数値安定化)
NEWTON_ITERS = 30
BOOT_CHUNK = 100      # bootstrap 反復をこの単位でバッチ処理(メモリ抑制)
OUT_DIR = PROJECT_ROOT / "results" / "additional" / "ipw_t50"
PS_SPECS = ("linear", "quad")


def design(x_raw, spec):
    """x_raw: (..., n, 2) の生共変量 → 傾向スコアの計画行列 (..., n, p)。"""
    x1, x2 = x_raw[..., 0], x_raw[..., 1]
    cols = [np.ones_like(x1), x1, x2]
    if spec == "quad":
        cols += [x1 ** 2, x2 ** 2, x1 * x2]
    return np.stack(cols, axis=-1)


def fit_logistic_batched(X, d):
    """バッチ Newton–Raphson によるロジスティック回帰。

    X: (B, n, p), d: (B, n) ∈ {0,1}。戻り値 beta: (B, p)。
    """
    B, n, p = X.shape
    beta = np.zeros((B, p))
    pen = np.full(p, RIDGE)
    pen[0] = 0.0
    eye_pen = np.diag(pen)
    for _ in range(NEWTON_ITERS):
        eta = np.einsum("bnp,bp->bn", X, beta)
        eta = np.clip(eta, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-eta))
        w = prob * (1.0 - prob)
        grad = np.einsum("bnp,bn->bp", X, d - prob) - beta * pen
        # (B,n,p)^T @ ((B,n,1) * (B,n,p)) → (B,p,p)。(B,n,p,q) の中間配列を作らない
        hess = np.matmul(np.transpose(X, (0, 2, 1)), w[..., None] * X) + eye_pen
        try:
            step = np.linalg.solve(hess, grad[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = np.linalg.solve(hess + 1e-6 * np.eye(p), grad[..., None])[..., 0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def ipw_estimate_batched(tx_raw, ty, sx_raw, sy, spec):
    """tx_raw: (B, n_c, 2), ty: (B, n_c), sx_raw: (B, n_h, 2), sy: (B, n_h)。

    戻り値: estimate (B,), ess (B,), max_weight (B,)
    """
    Xt = design(tx_raw, spec)
    Xs = design(sx_raw, spec)
    X = np.concatenate([Xt, Xs], axis=1)
    d = np.concatenate([np.ones(Xt.shape[:2]), np.zeros(Xs.shape[:2])], axis=1)
    beta = fit_logistic_batched(X, d)
    logit_s = np.clip(np.einsum("bnp,bp->bn", Xs, beta), -30, 30)
    w = np.exp(logit_s)                         # e/(1-e) = exp(logit)
    wsum = w.sum(axis=1)
    est = ty.mean(axis=1) - (w * sy).sum(axis=1) / wsum
    ess = wsum ** 2 / (w ** 2).sum(axis=1)
    return est, ess, w.max(axis=1)


def run_one_trial(i, scenario, source_n, gamma, n_boot):
    random.seed(i)
    np.random.seed(i)
    target_x, target_y, source_x, source_y = generate_data(
        scenario, source_n, gamma, TARGET_N
    )
    tx, sx = target_x[:, 1:], source_x[:, 1:]
    ty, sy = target_y.ravel(), source_y.ravel()

    rng = np.random.default_rng(i)
    t_idx = rng.choice(TARGET_N, size=(n_boot, TARGET_N), replace=True)
    s_idx = rng.choice(source_n, size=(n_boot, source_n), replace=True)

    row = {"trial": i}
    for spec in PS_SPECS:
        est, ess, wmax = ipw_estimate_batched(tx[None], ty[None], sx[None], sy[None], spec)
        boot = np.concatenate([
            ipw_estimate_batched(tx[t_idx[k:k + BOOT_CHUNK]], ty[t_idx[k:k + BOOT_CHUNK]],
                                 sx[s_idx[k:k + BOOT_CHUNK]], sy[s_idx[k:k + BOOT_CHUNK]], spec)[0]
            for k in range(0, n_boot, BOOT_CHUNK)
        ])
        boot = boot[np.isfinite(boot)]
        se = float(boot.std(ddof=1))
        row.update({
            f"{spec}_estimate": float(est[0]),
            f"{spec}_ess": float(ess[0]),
            f"{spec}_max_weight": float(wmax[0]),
            f"{spec}_boot_se": se,
            f"{spec}_ci_lower": float(np.percentile(boot, 2.5)),
            f"{spec}_ci_upper": float(np.percentile(boot, 97.5)),
            f"{spec}_normal_lower": float(est[0] - 1.96 * se),
            f"{spec}_normal_upper": float(est[0] + 1.96 * se),
            f"{spec}_boot_n": int(len(boot)),
        })
    return row


def summarize(df, scenario, source_n, gamma):
    out = {"scenario": scenario, "source_n": source_n, "gamma": gamma,
           "target_n": TARGET_N, "n_trial": len(df)}
    for spec in PS_SPECS:
        e = df[f"{spec}_estimate"]
        cov = ((df[f"{spec}_ci_lower"] <= TRUE_VALUE) & (TRUE_VALUE <= df[f"{spec}_ci_upper"])).mean()
        cov_n = ((df[f"{spec}_normal_lower"] <= TRUE_VALUE) & (TRUE_VALUE <= df[f"{spec}_normal_upper"])).mean()
        out.update({
            f"{spec}_mean": e.mean(),
            f"{spec}_bias": e.mean() - TRUE_VALUE,
            f"{spec}_sd": e.std(ddof=1),
            f"{spec}_rmse": np.sqrt(((e - TRUE_VALUE) ** 2).mean()),
            f"{spec}_coverage": cov,
            f"{spec}_ci_width": (df[f"{spec}_ci_upper"] - df[f"{spec}_ci_lower"]).mean(),
            f"{spec}_coverage_normal": cov_n,
            f"{spec}_ci_width_normal": (df[f"{spec}_normal_upper"] - df[f"{spec}_normal_lower"]).mean(),
            f"{spec}_ess_median": df[f"{spec}_ess"].median(),
            f"{spec}_max_weight_mean": df[f"{spec}_max_weight"].mean(),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="trials_*.csv が既に n_trial 行あれば再計算せず summary に取り込む")
    args = ap.parse_args()

    n_trial = 20 if args.smoke else N_TRIAL
    n_boot = 200 if args.smoke else N_BOOT
    out_dir = OUT_DIR / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    t_all = time.time()
    for gamma in GAMMAS:
        for scenario in SCENARIOS:
            for source_n in SOURCE_NS:
                t0 = time.time()
                trials_path = out_dir / f"trials_{scenario}_N{source_n}_gamma{gamma:g}.csv"
                if args.resume and trials_path.exists():
                    df = pd.read_csv(trials_path)
                    if len(df) == n_trial:
                        summaries.append(summarize(df, scenario, source_n, gamma))
                        print(f"skip (exists): γ={gamma:g} {scenario} N={source_n}", flush=True)
                        continue
                rows = Parallel(n_jobs=args.n_jobs)(
                    delayed(run_one_trial)(i, scenario, source_n, gamma, n_boot)
                    for i in range(n_trial)
                )
                df = pd.DataFrame(rows)
                df.to_csv(trials_path, index=False)
                s = summarize(df, scenario, source_n, gamma)
                summaries.append(s)
                print(f"[{time.time() - t_all:7.0f}s] γ={gamma:g} {scenario} N={source_n}: "
                      f"lin bias={s['linear_bias']:+.3f} cov={s['linear_coverage']:.3f} | "
                      f"quad bias={s['quad_bias']:+.3f} cov={s['quad_coverage']:.3f} "
                      f"({time.time() - t0:.1f}s)", flush=True)
                pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    print("done", out_dir / "summary.csv")


if __name__ == "__main__":
    main()
