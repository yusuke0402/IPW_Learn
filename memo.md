# todoリスト
## スクリプトのバージョンの更新 (済 2026-10-05)
IW-Learn と data.py / result.py / configs を揃えた。true_value は DataSets.__init__ で一度だけ読む。
同一シードで生成データが IW-Learn とビット単位で一致することを確認済み。

## 重みの変更 (済 2026-10-05)
weight.py で重みの種類を選べるようにした。configs/config.yaml の `hyperparameters.weight_method` で切り替える。
- uniform: 一様重み（輸送なしの平均差）
- propensity: 傾向スコアのオッズ e/(1-e) を source に掛ける（推定対象は現在試験集団の Δ_c）
- ulsif: uLSIF による密度比（IW-Learn と同じ CV 手順）
- oracle: 真のガウス分布からの密度比

### 残課題
- (済 2026-10-05) 旧実装の両側重み付け(target 1/e, source 1/(1-e)、混合集団が推定対象)は削除した。
  10_dim_result / results / results_20 / no_difference_10dim_results などの既存結果はこの旧実装で生成されたもの。
- (済 2026-10-05) 傾向スコアの罰則を `propensity_penalty`("l2" = sklearn 既定 C=1(既定) / "none" = 最尤)と `propensity_C` で選べるようにした。
  既定は元の実装どおり L2・C=1 のまま。PSS-Learn の propensityscore.py と同じ推定量で、傾向スコア系の手法間で揃えるため。
  最尤("none")は分離が頻発する(10 次元・target 20 で 37〜86%)ので感度解析用。t50_comparison/run_ipw_t50.py はほぼ無罰則なので比較時は注意。
- (済 2026-10-05) `scripts/nonlinear_gamma_ipw.py` を追加。IW-Learn と同一データ生成の非線形 γ 実験で、
  `--weight-method` で propensity / ulsif / uniform / oracle を選べる(bootstrap 付き、summary は IW-Learn 形式)。
- (済 2026-10-05) main.py に任意の full-pipeline bootstrap(`n_bootstraps`、0 で無効)を追加。
- oracle は 10 次元・大きなシフトで重みが極端になり分散が大きい（想定どおりだが要注意）
- numpy 2.0.2 + Accelerate で `divide by zero encountered in matmul` の誤検知警告が出る（値は正常）
