# todoリスト
## スクリプトのバージョンの更新 (済 2026-10-05)
IW-Learn と data.py / result.py / configs を揃えた。true_value は DataSets.__init__ で一度だけ読む。
同一シードで生成データが IW-Learn とビット単位で一致することを確認済み。

## 重みの変更 (済 2026-10-05)
weight.py で重みの種類を選べるようにした。configs/config.yaml の `hyperparameters.weight_method` で切り替える。
- uniform: 一様重み（輸送なしの平均差）
- propensity: 傾向スコアのオッズ e/(1-e) を source に掛ける（推定対象は現在試験集団の Δ_c）
- propensity_ate: 旧実装（target 1/e, source 1/(1-e)。混合集団への重み付けなので Δ_c とは一致しない）
- ulsif: uLSIF による密度比（IW-Learn と同じ CV 手順）
- oracle: 真のガウス分布からの密度比

### 残課題
- 傾向スコアは sklearn の既定（L2, C=1）のまま。t50_comparison/run_ipw_t50.py はほぼ無罰則なので揃えるか検討
- oracle は 10 次元・大きなシフトで重みが極端になり分散が大きい（想定どおりだが要注意）
- numpy 2.0.2 + Accelerate で `divide by zero encountered in matmul` の誤検知警告が出る（値は正常）
