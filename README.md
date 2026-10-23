

## 2026-10-20：IWLと揃えた再実験の規則

- ESS下限（既定3、感度分析5・10）は元標本だけに適用します。
- bootstrapではESSを記録し、正の重みの行数が設計行列の列数（今回3）以上であることを求めます。AIPW/IPWでは比較条件を揃えるための共通ルールです。
- bootstrapの計算失敗が5%を超えた場合、または元標本で点推定できなかった場合はCIを作成しません。元標本が失敗しても再標本の診断を保存します。
- uLSIFのσ倍率は0.3・0.5・0.7・1・1.4・2・3、λ候補は維持し、予測値を非負化します。bootstrapでもハイパーパラメータを再選択します。
- bootstrapでは全重み0を一様重みに戻しません。`min_ess: 0`による旧救済は元標本の再現用途だけです。

## ESS による実行可否判定

全ての重み方式に共通して、過去標本の Kish ESS = (Σw)² / Σw² を計算します。
既定は `ESS >= 3`。`configs/config.yaml` の `hyperparameters.min_ess` を
5、10 などへ変更できます。非線形シナリオの CLI では `--min-ess` が設定ファイルを
上書きします。正の重みの人数と最大の正規化重みも保存します。

点推定と bootstrap 各反復で重みを推定し直し、元標本のESS未達・bootstrapの正の重み人数不足は失敗として
記録します。既定では失敗反復の割合が 5% を超えると信頼区間を欠測にします
(`max_bootstrap_failure_rate` / `--max-bootstrap-failure-rate`)。
点推定失敗時も診断と bootstrap 結果を保存しますが、CI は作成しません。
被覆率は有効 CI のみを分母とし、成功数・ESS 未達率・bootstrap 失敗率を併記します。
AIPW の HT 補正版と自己正規化版には同一の ESS 判定を適用します。

ESS 有効時は全重み0を一様重みに置き換えません。
`min_ess: 0` は元標本のESS閾値を無効化し、元標本だけでuLSIFの従来の全重み0救済も復元する
過去実験再現用の指定です。bootstrapでは救済しません。感度分析と旧結果の再現は新しい出力先で実行してください。
ESS の設定・bootstrap 失敗率の基準・ソースコードが異なる出力先への追記は拒否します。

現在の AIPW / IPW には、重みが確定した後の回帰 CV や外側クロスフィットはありません。
uLSIF の密度比ハイパーパラメータ選択用 CV は、まだ推定されていない重みによる
層別化の対象とはせず、その手順は維持します。IWL の回帰用 CV は正の重みによる
層別分割を行い、正の重みが少なければ fold 数を減らします。

## Logistic重みのCV選択（2026-10-17）

logistic重みはIWL・IPW・AIPW-SNで同じ実装を使います。既定では現在／過去の所属群で層化した5-fold CVのlog lossを最小化するCを選択します。候補は1e-4〜1e4の対数等間隔17点、分割seedは2024です。Newton法（newton-cholesky）、tol=1e-8、最大1000反復、確率クリップ1e-12に統一しています。アウトカム・真の効果・ESSでCを選びません。

設定は propensity_cv、propensity_C_grid、propensity_cv_folds。CLIは --propensity-cv / --no-propensity-cv、--propensity-C-grid、--propensity-cv-folds。CVを無効にすると設定ファイルの propensity_C（既定1）を使い、CLIで指定した値を優先します。bootstrapでもCを再選択します。重み確定後、元標本はESS、bootstrapは正の重み人数で判定します。CVは重みモデルの調整であり、推定量全体のcross-fittingではありません。bootstrapの重複標本は通常のCV分割で扱います。

## ２次元のシナリオ４

`data_scenario_4` はシナリオ２の target trial と source trial の x1 の平均だけを
入れ替えた設定です。平均ベクトル `(x1, x2)` は target が `(0, 3)`、source が `(2, 0)`。
x2 の平均、各群の共分散、アウトカムの生成式はシナリオ２と共通です。
`configs/config.yaml` の `n_features: 2` と `data_scenario_id: "data_scenario_4"` で選択でき、
非線形 γ 実験では `--scenarios data_scenario_4` を指定できます。既定の一括実行にも含まれます。

## uLSIF の CV（2026-10-22）

元標本とbootstrapの両方で、同じ元観測の複製を同じCV foldへまとめます。
CVの基底中心は各foldの学習側targetだけから選び、候補間で固定します。
元観測IDを省略した配列APIでは、同一の共変量行を同じgroupとして扱います。
group数が少なければfold数を減らし、いずれかの群が２group未満なら計算失敗として記録します。
σ候補・距離中央値のアンカー・λ候補・予測値の非負化・ESS判定は維持します。
bootstrapでもσ・λを選び直します。これはロジスティック重みやoracle重みを変更しません。

非線形実験の `run_meta.json` には `ratio_cv_policy` を保存します。
CV修正前の結果への追記はコードと設定の照合で拒否するため、新しい出力先を指定してください。

## 真の密度比によるIPW参照

`hyperparameters.weight_method: "oracle"` を指定すると、`data.py` が読み込んだ
真の平均・共分散から `p_target(x) / p_source(x)` を計算します。通常の設定実験は
２次元（SC1〜SC4）と10次元（SC1〜SC3）で利用できます。密度計算から切片列を除外し、
密度比をクリップせず、IPWの平均差で自己正規化します。σ・λの推定やCVはありません。
平均の計算内だけ最大重みで共通スケールを相殺し、有限の大きな重みによる和のoverflowを防ぎます。
返す重み・ESS診断・最大重みには、生の密度比を使います。
bootstrapでも真分布のパラメータを固定したまま、再抽出したsource行の密度比を評価します。

```bash
uv run scripts/nonlinear_gamma_ipw.py --weight-method oracle --scenarios data_scenario_4 \
  --target-n 50 --source-ns 20 100 500 --gammas 0 0.5 --trials 200 --bootstraps 200
```

非線形実験のCLIは２次元用です。APIの `weight.oracle_density_ratio(x, means, covariances)` は
切片なしの配列を受け取ります。真分布は試行の診断と、非線形実験の `run_meta.json` に保存し、
再開時には真分布も照合します。ESSとbootstrap失敗率の判定は他の重み方式と共通です。
非対称・非正定値・非有限の共分散や次元不一致はエラーとし、密度比のoverflowも計算失敗として記録します。
underflowによる０はそのままESS判定へ渡し、一様重みに置き換えません。

## 2026-10-22 パイロット実験

指定条件の実験を、このリポジトリ単独で実行・停止・再開・集計できます。
[別マシン用の実行手順](experiments/pilot_20261022/README.md)を参照してください。

### プログラムの構成

現在試験（target）のアウトカム平均から、過去対照群（source）を現在試験の
共変量分布へ重み付けしたアウトカム平均を引き、治療効果を推定するシミュレーションです。
パイロットは2次元で、アウトカムに `gamma * x1^2` を加え、真の治療効果を1に固定します。

| ファイル | 役割 |
|---|---|
| `main.py` / `data.py` / `configs/` | 通常の設定ファイルによる人工データ生成・実験 |
| `scripts/nonlinear_gamma_ipw.py` | パイロットが使用する2次元データ生成と1試行の入口 |
| `weight.py` | uLSIF・logistic・oracle重みと自己正規化した平均差 |
| `logistic_weight.py` / `propensityscore.py` | 所属群ラベルを使うロジスティック回帰とCのCV選択 |
| `oracle_weight.py` | 真のガウス分布から計算する密度比 |
| `experiment_execution.py` / `experiment_diagnostics.py` | bootstrap、ESS、点推定・区間の失敗判定 |
| `experiments/pilot_20261022/prepare.py` | 固定コミットの書き出しと固定依存環境の準備 |
| `experiments/pilot_20261022/runner.py` | 実行計画、並列実行、checkpoint、停止・再開・照合 |
| `experiments/pilot_20261022/pilot_weighted.py` | 固定科学コードを呼び出し、データハッシュと共通形式の結果を作成 |
| `experiments/pilot_20261022/pilot_report.py` | 試行CSV、条件別集計、図の出力 |

パイロットの設定は `experiments/pilot_20261022/design.json`、科学コードの版は
`source_pins.json` が正本です。`main.py` の通常実行や `configs/config.yaml` の変更は
パイロットの設定変更にはなりません。別リポジトリとの比較設計も記録されていますが、
このリポジトリのランナーが実行するのはIPWの3種類の重みだけです。
