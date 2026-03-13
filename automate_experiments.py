import yaml
import copy
import os
from main import run_experiment


def automate():
    # ベースとなる設定を読み込む
    with open("configs/config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # 実験パターンの定義
    scenarios = [
        "data_scenario_1",
        "data_scenario_2",
        "data_scenario_3",
        "data_scenario_4",
    ]
    # 人数パターン
    n_patterns = [20, 50, 100, 200, 500, 1000]

    total_experiments = len(scenarios) * len(n_patterns)
    current_count = 0

    print("=" * 50)
    print(f"全 {total_experiments} パターンの自動実験を開始します。")
    print(f"保存先: 10_dim_result/")
    print("=" * 50)

    for scenario_id in scenarios:
        for n in n_patterns:
            current_count += 1
            print(f"\n[{current_count}/{total_experiments}] 実行中:")
            print(f"  > Scenario: {scenario_id}")
            print(f"  > Dataset Size (Target=Source=N): {n}")

            # 設定の書き換え
            config = copy.deepcopy(base_config)
            config["scenario"]["data_scenario_id"] = scenario_id
            config["dataset"]["source_number"] = n
            # target_numberはbase_configの値を維持（固定）

            try:
                # 実験の実行
                run_experiment(config)
                print(f"  ✓ 完了: {scenario_id} (N={n})")
            except Exception as e:
                print(f"  ✗ エラー発生 (Scenario: {scenario_id}, N: {n}): {e}")

    print("\n" + "=" * 50)
    print("すべての実験が完了しました。")
    print(
        "10_dim_result フォルダ内にシナリオ別のファイルが生成されていることを確認してください。"
    )
    print("=" * 50)


if __name__ == "__main__":
    automate()
