| Config | FAR@TAR90 full | FAR@TAR90 LV75 | FAR@TAR90 LV90 | TAR@FAR1% full | TAR@FAR1% LV90 | AUROC full | AUROC LV90 |
|---|---|---|---|---|---|---|---|
| A0 baseline: single pass, max-sim, global threshold | 14.5±3.3 | 23.0±9.2 | 32.1±26.8 | 54.1±14.7 | 44.0±20.5 | 0.9628 | 0.9422 |
| A1 + flip test-time augmentation | 13.0±3.8 | 19.4±6.8 | 29.7±26.8 | 54.9±14.9 | 46.4±22.4 | 0.9649 | 0.9446 |
| A2 + multi-shot identity aggregation (max) | 13.0±3.8 | 19.4±6.8 | 29.7±26.8 | 54.9±14.9 | 46.4±22.4 | 0.9649 | 0.9446 |
| A3 + margin to runner-up identity  [s1, margin] | 10.8±3.2 | 18.8±8.1 | 29.9±25.6 | 58.0±15.1 | 44.3±24.6 | 0.9697 | 0.9409 |
| A4 + query-adaptive z-score (full Discern rule) | 10.8±3.2 | 18.4±8.5 | 30.1±26.0 | 57.7±14.5 | 43.0±23.6 | 0.9697 | 0.9402 |