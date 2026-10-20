# Decisive patterns: new physical descriptors (W=5s, hop=2s)

Columns: univariate AUC per split + sign-consistency flag.
`*` = decisive: |AUC|>=0.75 on >=2 splits AND same direction on all available splits.

| descriptor | train | val | calib | testml | stable |
|---|---|---|---|---|---|
| rd_motion | 0.579(+) | 0.511(+) | - | 0.564(+) | Y |
| rd_still | 0.541(+) | 0.529(+) | - | - | Y |
| rd_burst | 0.485(-) | 0.481(-) | - | 0.500(-) | Y |
| rd_hotn | 0.534(+) | 0.508(+) | - | 0.593(+) | Y |
| rd_rshift | 0.550(+) | 0.505(+) | - | 0.591(+) | Y |
| rd_dopbw | 0.486(-) | 0.474(-) | - | 0.450(-) | Y |
| snr_breath | 0.585(+) | 0.540(+) | - | 0.603(+) | Y |
| snr_ac1 | 0.569(+) | 0.537(+) | - | 0.596(+) | Y |
| csi_tv_s | 0.216(-) | 0.701(+) | - | 0.375(-) | n |
| csi_ac1 | 0.771(+) | 0.530(+) | - | 0.484(-) | n |
| csi_xcorr | 0.333(-) | 0.668(+) | - | - | n |
| csi_envbw | 0.222(-) | 0.468(-) | - | 0.138(-) | Y * |
| csi_imp | - | 0.499(-) | - | - | n |
| csi_madc | 0.461(-) | 0.574(+) | - | 0.613(+) | n |
| agree_mr | 0.477(-) | 0.505(+) | - | 0.571(+) | n |

## Class profiles (median per class, train split)
| descriptor | empty med | occupied med | cohen d |
|---|---|---|---|
| rd_motion | 196.6 | 200.2 | -0.32 |
| rd_still | 0.5 | 0.5 | -0.17 |
| rd_burst | 0.3333 | 0.25 | -0.00 |
| rd_hotn | 20 | 20.4 | -0.16 |
| rd_rshift | 0.01265 | 0.01623 | -0.29 |
| rd_dopbw | 5.851 | 5.852 | +0.35 |
| snr_breath | 0.05909 | 0.0804 | -0.41 |
| snr_ac1 | 0.1062 | 0.1422 | -0.34 |
| csi_tv_s | 1.155 | 0.7487 | +1.11 |
| csi_ac1 | 0.09709 | 0.4937 | -1.12 |
| csi_xcorr | 0.3179 | 0.2494 | +0.53 |
| csi_envbw | 22.91 | 13.7 | +1.16 |
| csi_madc | 1.538 | 1.487 | +0.20 |
| agree_mr | 0.4 | 0.4 | +0.09 |