| Training config (30 epochs) | mAP | R1 | FAR@TAR90 all | FAR@TAR90 LV90 | TAR@FAR1% LV90 |
|---|---|---|---|---|---|
| T0 ID loss only (CE+LS) | 83.5 | 93.5 | 14.92 | 49.05 | 47.4 |
| T1 + batch-hard triplet | 85.1 | 93.5 | 17.23 | 53.96 | 39.1 |
| T2 + identity-level look-alike mining | 85.0 | 94.1 | 13.64 | 37.82 | 38.3 |
| T3 + EMA weights (full recipe) | 85.1 | 93.9 | 14.59 | 31.66 | 38.0 |