

## ESS による実行可否判定

全ての重み方式に共通して、過去標本の Kish ESS = (Σw)² / Σw² を計算します。
既定は `ESS >= 3`。`configs/config.yaml` の `hyperparameters.min_ess` を
5、10 などへ変更できます。非線形シナリオの CLI では `--min-ess` が設定ファイルを
上書きします。正の重みの人数と最大の正規化重みも保存します。

点推定と bootstrap 各反復で重みを推定し直してから判定し、ESS 未達は失敗として
記録します。既定では失敗反復の割合が 5% を超えると信頼区間を欠測にします
(`max_bootstrap_failure_rate` / `--max-bootstrap-failure-rate`)。
点推定失敗時も診断と bootstrap 結果を保存しますが、CI は作成しません。
被覆率は有効 CI のみを分母とし、成功数・ESS 未達率・bootstrap 失敗率を併記します。
AIPW の HT 補正版と自己正規化版には同一の ESS 判定を適用します。

ESS 有効時は全重み0を一様重みに置き換えません。
`min_ess: 0` は ESS 閾値を無効化し、uLSIF の従来の全重み0救済も復元する
過去実験再現用の指定です。感度分析と旧結果の再現は新しい出力先で実行してください。
ESS の設定・bootstrap 失敗率の基準・ソースコードが異なる出力先への追記は拒否します。

現在の AIPW / IPW には、重みが確定した後の回帰 CV や外側クロスフィットはありません。
uLSIF の密度比ハイパーパラメータ選択用 CV は、まだ推定されていない重みによる
層別化の対象とはせず、その手順は維持します。IWL の回帰用 CV は正の重みによる
層別分割を行い、正の重みが少なければ fold 数を減らします。

## ２次元のシナリオ４

`data_scenario_4` はシナリオ２の target trial と source trial の x1 の平均だけを
入れ替えた設定です。平均ベクトル `(x1, x2)` は target が `(0, 3)`、source が `(2, 0)`。
x2 の平均、各群の共分散、アウトカムの生成式はシナリオ２と共通です。
`configs/config.yaml` の `n_features: 2` と `data_scenario_id: "data_scenario_4"` で選択でき、
非線形 γ 実験では `--scenarios data_scenario_4` を指定できます。既定の一括実行にも含まれます。
