# results/ の構成

## 旧結果

| 場所 | 内容 |
|---|---|
| `results/result_*.csv`, `results/summary_*.yaml` | 2026-02 の初期実験(旧 propensity_ate 重み、bootstrap なし) |
| `results_20/`, `10_dim_result/`, `no_difference_10dim_results/`, `RIWvsIPW-results/` | 初期の sweep と 10 次元・帰無効果の実験 |
| `t50_comparison/results/` | **論文 v1 の IPW 比較**(target 50、ロジスティック傾向スコア linear/quad、1000 × B=1000)。IW-Learn の `external/ipw_t50_summary.csv` の元 |

## 新しい結果の置き場所(2026-10-05 以降の規約、4 リポジトリ共通)

```
results/v2/<purpose>/<run_name>/
    trials_<scenario>_N<n_h>_gamma<g>.csv   試行ごとの推定値・区間(1 行 = 1 trial、trial 番号 = シード)
    summary.csv                             条件ごとの要約(IW-Learn の load_all() が読む列 + n_trial, B)
    run_meta.json                           引数、重み設定、git コミット、日時、ホスト名
```

- `purpose`
  - `paper`: 論文・資料に使う本番。試行数は後から追記で増やせる(試行単位のレジューム)。
  - `dev`: 動作確認・縮小規模・探索的な実験。論文には使わない。
  - `smoke`: `--smoke` の出力。git 管理外。
- `run_name` の既定は `<手法>_<重み>_t<target_n>_B<bootstraps>`。試行数は名前に含めない
  (同じディレクトリで `--trials` を増やすと足りない試行だけ追加計算される)。
  B が違う実験は別ディレクトリになる(bootstrap 区間が混ざらないようにするため)。
- 例: `results/v2/paper/ipw_propensity-l2_t50_B1000/`
