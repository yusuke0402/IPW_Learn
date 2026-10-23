# 別マシン用パイロット実験

このリポジトリの実行対象は **IPW自己正規化版（uLSIF・真の密度比・logistic）**（family: `ipw`）です。[design.json](design.json) の設定に従い、この手法だけを単独で実行・停止・再開・集計できます。

取得する科学コードは自分のリポジトリの [source_pins.json](source_pins.json) に記録したコミットです。他の実験リポジトリのcloneや、NPPのコード・JAX・NumPyro環境は必要ありません。

[2026年10月23日完了の共有結果](../../results/v2/dev/pilot_20261022_ipw_t50_B200/README.md)を保存しています。共有結果は実行先へ自動取り込みされず、空の出力先では各条件をseed 0から実行します。

## 実験条件

科学設定の正本は [design.json](design.json) です。

| 項目 | 設定 |
|---|---|
| 試行 | 各条件でseed 0〜199、200試行 |
| Target | 50人 |
| Source | 20人・100人・500人 |
| γ | 0・0.5 |
| IWL・AIPW・IPW | シナリオ1〜4、uLSIF・真の密度比・logisticの3重み |
| IWL | L1正則化なし、固定λ₁=0、Target全例使用 |
| AIPW・IPW | 自己正規化版 |
| G-computation・Conditional PP・Commensurate PP | シナリオ4のみ、Source×γの全6条件 |
| 頻度論的区間 | 各試行でbootstrap 200回、percentile 95%信頼区間 |
| Bootstrap失敗率 | 5%超、すなわち200回中11回以上の失敗でCI欠測 |
| 重みの点推定 | 最低ESS=3 |
| ベイズ区間 | 元実装の事後95%信用区間 |
| Conditional PPのa₀ | 0.1・0.25・0.5・0.75・1 |
| Commensurate PPのMCMC | 4,000反復、burn-in 1,000、thin 2、その他も固定ソースの既定値 |
| 真の効果 | 1 |

各リポジトリの予定件数は次のとおりです。実行時には自分の行だけを処理します。

| リポジトリ | family | ジョブ数 | 出力行数 | 出力条件数 |
|---|---|---:|---:|---:|
| IW-Learn | `iwl` | 14,400 | 14,400 | 72 |
| AIPW-Learn | `aipw` | 14,400 | 14,400 | 72 |
| IPW_Learn | `ipw` | 14,400 | 14,400 | 72 |
| G-computation | `gcomp` | 1,200 | 1,200 | 6 |
| Conditional_Powerprior | `conditional` | 1,200 | 6,000 | 30 |
| Commensurate_powerprior | `commensurate` | 1,200 | 1,200 | 6 |
| 合計（別々に実行） | | 46,800 | 51,600 | 258 |

Conditional PPは1ジョブでa₀別に5行を返します。6リポジトリを合わせた頻度論44,400試行では、合計8,880,000回のbootstrap計算を予定しています。

## 必要なもの

- LinuxまたはmacOSの端末環境（WindowsではUbuntu WSLを利用。下記参照）
- Git、[uv](https://docs.astral.sh/uv/getting-started/installation/)
- 実行するリポジトリを取得できる通信・認証設定
- Python 3.12（以下のコマンドではuvが選択します）

計算はCPUで行います。依存関係は、このディレクトリ内に同梱した `runtime/pyproject.toml` と `runtime/uv.lock` から固定インストールします。IWL・AIPW・IPW・G-computationは共通の依存版、Conditional PPとCommensurate PPはそれぞれ元リポジトリの依存版を使用します。頻度論の4リポジトリは、図の作成用に同梱した別の固定環境も準備します。このために他の実験リポジトリをcloneする必要はありません。

## 初回セットアップ

実行したいリポジトリの `codex/pilot-20261022-portable` ブランチを取得し、`experiments/pilot_20261022/` に移動します。以下はIPW_Learnの場合です。他のリポジトリではリポジトリ名を置き換えます。

```bash
git clone --branch codex/pilot-20261022-portable https://github.com/yusuke0402/IPW_Learn.git
```

`PILOT_WORK` は科学コード・環境・結果を保存する作業用ディレクトリです。省略時はこのディレクトリの `work/` を使用します。リポジトリごとに別の作業ディレクトリを使用してください。

```bash
cd /path/to/IPW_Learn/experiments/pilot_20261022
PILOT_WORK="$PWD/work"

uv run --no-project --python 3.12 prepare.py --workdir "$PILOT_WORK"
uv run --no-project --python 3.12 runner.py prepare --workdir "$PILOT_WORK"
uv run --no-project --python 3.12 runner.py plan --workdir "$PILOT_WORK"
```

`prepare.py` は自分のリポジトリの固定コミットを `sources/<family>/` に書き出し、同梱の依存宣言とロックファイルを `runtime/` にコピーして、`uv sync --frozen --no-install-project` で環境を準備します。フルcloneでは自分のGit履歴を利用します。浅いcloneに対象コミットがない場合は、自分のリモートから取得します。

`runner.py prepare` は、実験条件・コード・ソース・環境を記録したmanifestを作ります。ここまでは推定計算を始めません。

`prepare.py --sources-only` は科学コードの準備だけを行う診断用です。依存関係が未準備の状態では実験を開始できません。

## 起動確認と本実験

まず、少数の試行で実際の推定・保存・照合を確認します。重み付き手法では3種類の重みを各1試行、その他は自分の手法を1試行だけ実行します。頻度論の検証でもbootstrapは200回です。検証出力は本実験と分けます。

```bash
uv run --no-project --python 3.12 runner.py preflight --workdir "$PILOT_WORK"
```

本実験は次で開始します。`--workers` は同時に計算するプロセス数です。以下は4並列の例です。省略時は利用可能なCPU割当を基準に選びます。BLAS等の内部スレッド数は1に設定します。

同じマシンで複数リポジトリを同時に動かす場合は、**各リポジトリの `--workers` の合計が利用可能なCPU割当を超えないように指定**してください。省略時は各実験がそれぞれ利用可能なCPUを選ぶため、複数実験を同時に既定値で開始すると競合します。別々のマシンで動かす場合は、それぞれの割当に合わせます。

```bash
uv run --no-project --python 3.12 runner.py run \
  --workdir "$PILOT_WORK" --workers 4 --batch-size 10
```

バッチは同一手法・条件の最大10seedを担当しますが、保存は試行ごとです。バッチ全体が終わるまで保存を待つ必要はありません。全件完了後にCSVと図を集計します。

長時間の計算では、そのマシンの運用に合わせてターミナルの保持やジョブ管理を行ってください。ファイルの保存と、プロセスの存続は別です。

## 状態確認・停止・再開

別のターミナルでも同じ作業ディレクトリを指定します。

```bash
uv run --no-project --python 3.12 runner.py status --workdir "$PILOT_WORK"
uv run --no-project --python 3.12 runner.py stop --workdir "$PILOT_WORK"
```

停止要求を出すと、各ワーカーは計算中の1試行を終えて保存してから終了します。bootstrapやMCMCの途中で即時停止する処理ではありません。

停止後に続ける場合は、停止要求を明示的に解除します。

```bash
uv run --no-project --python 3.12 runner.py run \
  --workdir "$PILOT_WORK" --workers 4 --batch-size 10 --clear-stop
```

完了済みcheckpointを検証し、未完了の試行のみ計算します。同じ出力先の二重起動を拒否します。科学設定、コード、ソース、依存関係、保存済み結果の内容に不一致がある場合も停止し、古い結果を自動で流用しません。設定やコードを変える実験には新しい出力先を使用します。

## 保存先と集計

本実験の既定出力は `PILOT_WORK/results/`、起動確認は `PILOT_WORK/verification/preflight/` です。`runner.py` の `--output` で別の出力先を指定できます。その場合は準備・起動・状態確認・停止・集計で同じ出力先を指定してください。起動確認の出力は `PILOT_WORK/verification/` 配下に限ります。

| 場所 | 内容 |
|---|---|
| `prepared.json` | 科学コードと作成した環境の記録 |
| `sources/<family>/` | 自分のリポジトリの固定科学コード |
| `runtime/` | 同梱の依存宣言・ロックファイルと `.venv` |
| `reporting/` | 頻度論4リポジトリの図作成用の固定環境 |
| `results/manifest.json` | 実験条件・コード・ソース・環境の記録 |
| `results/raw/<method>/scenario_04/gamma_0p5/source_020/trial_000000.json` | 手法・条件・seed別のcheckpoint。元結果・共通形式の行・実データハッシュ・実行時間を保存 |
| `results/logs/` | バッチ別のログ |
| `results/progress.json` | 約15秒ごとの状態・保存件数。整合性検証済みの最新件数は `status` で確認 |
| `results/errors/` | 予期しない例外の条件とtraceback |
| `results/STOP` | 停止要求 |
| `results/summary/normalized_trials.csv` | 共通列名の試行別結果 |
| `results/summary/per_condition_summary.csv` | bias・SD・RMSE・coverage・平均区間幅・失敗件数等 |
| `results/summary/report_status.json` | 集計した件数・残数・有効推定/区間数 |
| `results/figures/` | シナリオ・γ別集計図 |

完了済みcheckpointは途中でも集計できます。

```bash
uv run --no-project --python 3.12 runner.py report --workdir "$PILOT_WORK" --plots
```

checkpointは一時ファイルから置換して確定します。内容のchecksum、設定のfingerprint、同条件・同seedで実際に推定へ渡した配列のハッシュを照合します。予期しないプログラムエラーはログに残し、追加のジョブ投入を停止します。

欠測はJSONで `null`、CSVで `NA` として保存します。ESS不足やCI失敗を0として扱いません。bias・RMSEは使用できる点推定、coverageと区間幅は使用できる区間で計算し、それぞれの分母と失敗数を出力します。

ベイズ点推定は元実装どおり、Conditional PPでは事後平均、Commensurate PPでは事後中央値です。Commensurate PPでは受理率等の診断を保存します。

各リポジトリの出力先は独立しています。他リポジトリの結果を自動でコピーしたり、G-computationとの差を別リポジトリから自動収集したりはしません。

## 検証

[動作確認の記録](VALIDATION.md)を参照してください。推定を実行しないランナーの検証は、次で実行できます。

```bash
uv run --no-project --python 3.12 -m unittest discover -s . -p test_runner.py -v
```

## このWindows PCでの実行（Ubuntu WSL）

ランナーは `fcntl` によるプロセス間ロックを使うため、WindowsのPythonから直接起動せず、
Ubuntu WSLで実行します。作業場所は
`C:\Users\yusuke\Documents\IPW_Learn\experiments\pilot_20261022\work` です。
ルートの `.venv` やWindows側のPythonは、このパイロットの実行環境ではありません。

このPC用の準備では `work/tools/` にLinux版uvとPythonを配置しています。
環境を再確認する場合は、Ubuntu WSL内で次を実行します。

```bash
cd /mnt/c/Users/yusuke/Documents/IPW_Learn/experiments/pilot_20261022
export PATH="$PWD/work/tools/uv-x86_64-unknown-linux-gnu:$PATH"
export UV_CACHE_DIR="$PWD/work/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/work/tools/python"
python3 prepare.py --workdir "$PWD/work"
./work/runtime/.venv/bin/python runner.py prepare
./work/runtime/.venv/bin/python runner.py plan
```

セットアップ後はuvを介さず固定環境のPythonで起動できます。
以下は **WindowsのPowerShell** からのコマンドです。

```powershell
$pilotDir = '/mnt/c/Users/yusuke/Documents/IPW_Learn/experiments/pilot_20261022'

# 状態確認（計算は開始しない）
wsl -d Ubuntu --cd $pilotDir --exec ./work/runtime/.venv/bin/python runner.py status

# 3重み各1試行の起動確認。bootstrapは各200回。
wsl -d Ubuntu --cd $pilotDir --exec ./work/runtime/.venv/bin/python runner.py preflight --workers 3

# 全72条件・14,400試行を開始（4並列）
wsl -d Ubuntu --cd $pilotDir --exec ./work/runtime/.venv/bin/python runner.py run --workers 4 --batch-size 10

# 停止要求（別のPowerShellから実行可能）
wsl -d Ubuntu --cd $pilotDir --exec ./work/runtime/.venv/bin/python runner.py stop

# 停止解除・再開
wsl -d Ubuntu --cd $pilotDir --exec ./work/runtime/.venv/bin/python runner.py run --workers 4 --batch-size 10 --clear-stop
```

本実験は計2,880,000回のbootstrapを含みます。起動確認の結果は
`work/verification/preflight/`、本実験の結果は `work/results/` に分かれます。
実行中はWSLと起動したターミナルを維持してください。
