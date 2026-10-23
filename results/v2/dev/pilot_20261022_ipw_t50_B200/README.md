# IPWパイロット実験の完了結果

2026年10月23日14:42頃（日本時間）に集計完了。探索用パイロットの結果です。

- 3重み（uLSIF・oracle・logistic）×4シナリオ×Source人数3条件×gamma 2条件＝72条件。
- Target 50人、Source 20/100/500人、gamma 0/0.5、各条件seed 0〜199。
- 全14,400試行、各bootstrap 200回。各重み4,800試行。
- 有効点推定12,652、有効区間12,612。欠測はNAとして保存しています。
- 4並列から、12,131試行を保存して12並列へ変更。予期しない実行エラー0件。
- 科学コードは `source_pins.json` の固定コミット、実験条件は `design.json` を参照してください。

`summary/normalized_trials.csv` が試行別結果、`summary/per_condition_summary.csv` が条件別集計、
`figures/` が図8枚です。`summary/summary.csv` は条件別集計の互換用コピーです。
`paired_vs_gcomp.csv` はこのIPW単独実行では比較対象がなく、ヘッダーのみです。

`manifest.json` にコード・環境・設定の照合情報、`planned_conditions.csv` に計画を保存しています。
`progress.json` の `jobs_saved` は途中の定期観測値です。最終件数は
`jobs_complete` と `summary/report_status.json` の `completed_jobs`（ともに14,400）を使用してください。

この共有版は集計・参照用です。再開用raw checkpoint、仮想環境、キャッシュは含めません。
元の完全な実行出力は `experiments/pilot_20261022/work/results/` に保持しています。
各条件200行、試行キーの重複なし、72条件、有効推定・区間件数と完了記録の一致を確認しました。
