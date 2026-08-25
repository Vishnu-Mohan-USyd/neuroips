# Transplant-surround Phase 2 — results tables (generated from n4/n5/n6/n7 JSON)

Cell ID = (CELL, FB, GAINS); P = pretrain, T = trained, R/N/Q = fresh FB controls (random / norm-matched random / rotated-trained). `*` = CE-tripped seed (competence coords UNRESOLVABLE, house rule). `unrd` = coordinate UNREADABLE (|TTT-PPP| below its floor).

Gate chain: G6 PASS + re-verified after measurement (48/48 files unchanged); G1 8/8, G2 4/4, G0 exact (s8 abs diff 0.0 vs sha-pinned artifacts), G3 4/4, G5 all, control gates + null-edit 8/8, determinism repeat exact, EC1 4/4, shared-cell dual-file bitwise 5/5.

## Raw markers — alpha0.0 (sharpening)

### seed 8

| cell | hit | decode | rate | M | center | flank | trip |
|---|---|---|---|---|---|---|---|
| PPP | 0.4769 | 0.7578 | 0.0469 | 1.3554 | 1.8070 | 0.8102 |  |
| TPP | 0.2778 | 0.4087 | 0.0334 | 1.3233 | 1.8046 | 0.8568 |  |
| PTP | 0.2454 | 0.3511 | 0.0438 | 1.3481 | 1.6354 | 1.0393 |  |
| PPT | 0.0972 | 0.0826 | 0.0083 | 1.0080 | 1.0205 | 1.0199 |  |
| TTP | 0.2176 | 0.3132 | 0.0322 | 1.3001 | 1.7131 | 0.7587 |  |
| TPT | 0.6620 | 0.2345 | 0.0114 | 0.9758 | 1.1383 | 0.8505 |  |
| PTT | 0.1204 | 0.1013 | 0.0100 | 1.0067 | 1.0231 | 1.0103 | TRIP |
| TTT | 0.9907 | 0.3825 | 0.0129 | 0.9672 | 1.1895 | 0.8279 |  |
| TRT | 0.0370 | 0.0017 | -0.0001 | 1.0196 | 1.0004 | 1.0045 |  |
| TNT | 0.0324 | 0.0111 | -0.0003 | 1.1057 | 1.0015 | 1.0035 | TRIP |
| TQT | 0.0417 | -0.0388 | -0.0010 | 1.1118 | 1.0001 | 1.0054 |  |
| PRP | 0.0324 | 0.0373 | -0.0003 | 1.4582 | 1.0304 | 1.2163 |  |
| PNP | 0.0139 | -0.0111 | -0.0040 | 1.8935 | 1.0082 | 1.3691 | TRIP |
| PQP | 0.0370 | -0.0496 | -0.0097 | 1.8739 | 1.0509 | 1.3581 | TRIP |

### seed 9

| cell | hit | decode | rate | M | center | flank | trip |
|---|---|---|---|---|---|---|---|
| PPP | 0.4491 | 0.7503 | 0.0468 | 1.3541 | 1.8149 | 0.8092 |  |
| TPP | 0.3472 | 0.4421 | 0.0349 | 1.3132 | 1.8867 | 0.7954 |  |
| PTP | 0.2454 | 0.3001 | 0.0438 | 1.3447 | 1.6075 | 1.0683 |  |
| PPT | 0.1435 | 0.0475 | 0.0123 | 1.0142 | 1.0402 | 1.0102 |  |
| TTP | 0.2500 | 0.3043 | 0.0309 | 1.2995 | 1.8163 | 0.7767 |  |
| TPT | 0.7269 | 0.2606 | 0.0125 | 0.9826 | 1.1565 | 0.8468 |  |
| PTT | 0.1574 | 0.0456 | 0.0136 | 1.0119 | 1.0445 | 0.9952 | TRIP |
| TTT | 0.9676 | 0.4019 | 0.0140 | 0.9740 | 1.2132 | 0.8253 |  |
| TRT | 0.0556 | 0.0017 | 0.0000 | 1.0257 | 1.0016 | 1.0038 |  |
| TNT | 0.0463 | 0.0405 | -0.0007 | 1.1246 | 1.0121 | 1.0016 | TRIP |
| TQT | 0.0139 | -0.0169 | -0.0018 | 1.1234 | 0.9993 | 1.0273 |  |

### seed 10

| cell | hit | decode | rate | M | center | flank | trip |
|---|---|---|---|---|---|---|---|
| PPP | 0.4815 | 0.7678 | 0.0472 | 1.3570 | 1.8026 | 0.8139 |  |
| TPP | 0.4444 | 0.5159 | 0.0376 | 1.3081 | 1.9333 | 0.7581 |  |
| PTP | 0.2269 | 0.3403 | 0.0464 | 1.3398 | 1.6294 | 0.9794 |  |
| PPT | 0.1528 | 0.0135 | 0.0130 | 1.0197 | 1.0381 | 1.0280 |  |
| TTP | 0.2361 | 0.3281 | 0.0334 | 1.2955 | 1.7748 | 0.7538 |  |
| TPT | 0.6898 | 0.2650 | 0.0127 | 0.9844 | 1.1629 | 0.8449 |  |
| PTT | 0.1343 | 0.0402 | 0.0142 | 1.0177 | 1.0442 | 1.0098 |  |
| TTT | 0.9907 | 0.4219 | 0.0141 | 0.9759 | 1.2201 | 0.8245 |  |
| TRT | 0.0370 | 0.0009 | -0.0000 | 1.0221 | 1.0011 | 1.0063 |  |
| TNT | 0.0417 | 0.0163 | 0.0008 | 1.1332 | 1.0114 | 0.9973 |  |
| TQT | 0.0324 | -0.0247 | 0.0009 | 1.1268 | 0.9996 | 1.0352 |  |

### seed 11

| cell | hit | decode | rate | M | center | flank | trip |
|---|---|---|---|---|---|---|---|
| PPP | 0.4583 | 0.7507 | 0.0480 | 1.3565 | 1.7972 | 0.8221 |  |
| TPP | 0.3426 | 0.4439 | 0.0366 | 1.3159 | 1.8699 | 0.8068 |  |
| PTP | 0.2685 | 0.3760 | 0.0425 | 1.3474 | 1.5921 | 1.0498 |  |
| PPT | 0.1528 | 0.0576 | 0.0125 | 1.0164 | 1.0421 | 1.0123 |  |
| TTP | 0.2546 | 0.2501 | 0.0318 | 1.3077 | 1.6713 | 0.8141 |  |
| TPT | 0.6528 | 0.2535 | 0.0128 | 0.9854 | 1.1581 | 0.8489 |  |
| PTT | 0.1250 | 0.0783 | 0.0138 | 1.0161 | 1.0432 | 1.0078 |  |
| TTT | 0.9861 | 0.4327 | 0.0141 | 0.9760 | 1.2216 | 0.8240 |  |
| TRT | 0.0648 | -0.0010 | 0.0002 | 1.0211 | 1.0017 | 1.0059 |  |
| TNT | 0.0602 | 0.0072 | 0.0016 | 1.1167 | 1.0108 | 1.0062 |  |
| TQT | 0.0278 | 0.0107 | -0.0009 | 1.1264 | 1.0055 | 1.0267 | TRIP |

## Raw markers — alpha0.5 (dampening)

### seed 8

| cell | hit | decode | rate | M | center | flank | trip |
|---|---|---|---|---|---|---|---|
| PPP | 0.4769 | 0.7578 | 0.0469 | 1.3554 | 1.8070 | 0.8102 |  |
| TPP | 0.0880 | -0.4236 | -0.0287 | 1.5772 | 1.1343 | 1.8844 |  |
| PTP | 0.1991 | 0.5305 | 0.0435 | 1.3682 | 1.6723 | 1.0022 |  |
| PPT | 0.0833 | -0.2847 | 0.0377 | 0.6230 | 0.5635 | 0.6626 | TRIP |
| TTP | 0.0602 | -0.3970 | -0.0309 | 1.6210 | 1.1268 | 1.8875 |  |
| TPT | 0.1806 | 0.0065 | 0.0408 | 0.3188 | 0.1507 | 0.5319 |  |
| PTT | 0.0972 | -0.2745 | 0.0386 | 0.6171 | 0.5543 | 0.6644 |  |
| TTT | 0.1991 | -0.0285 | 0.0424 | 0.2961 | 0.1436 | 0.4999 |  |
| TRT | 0.0417 | -0.0752 | 0.0011 | 0.7611 | 0.8185 | 0.7044 |  |
| TNT | 0.0417 | -0.0370 | -0.0014 | 0.8485 | 0.8834 | 0.8230 |  |
| TQT | 0.0139 | -0.0311 | 0.0000 | 0.8693 | 0.9044 | 0.8423 |  |
| PRP | 0.0324 | 0.0373 | -0.0003 | 1.4582 | 1.0304 | 1.2163 |  |
| PNP | 0.0370 | -0.0598 | -0.0011 | 1.9959 | 1.0424 | 1.4203 |  |
| PQP | 0.0231 | -0.1710 | -0.0203 | 1.8206 | 1.0498 | 1.4241 |  |

### seed 9

| cell | hit | decode | rate | M | center | flank | trip |
|---|---|---|---|---|---|---|---|
| PPP | 0.4491 | 0.7503 | 0.0468 | 1.3541 | 1.8149 | 0.8092 |  |
| TPP | 0.0648 | -0.4999 | -0.0373 | 1.6160 | 1.0845 | 2.0391 |  |
| PTP | 0.1944 | 0.3957 | 0.0409 | 1.3662 | 1.6071 | 1.0635 |  |
| PPT | 0.1250 | -0.3147 | 0.0394 | 0.6714 | 0.6321 | 0.7231 | TRIP |
| TTP | 0.0417 | -0.5035 | -0.0383 | 1.6593 | 1.0792 | 2.0180 |  |
| TPT | 0.1806 | 0.0009 | 0.0396 | 0.3436 | 0.1649 | 0.5780 |  |
| PTT | 0.0972 | -0.3128 | 0.0389 | 0.6750 | 0.6380 | 0.7226 | TRIP |
| TTT | 0.2037 | -0.0563 | 0.0452 | 0.2637 | 0.0927 | 0.4930 |  |
| TRT | 0.0324 | -0.1170 | 0.0013 | 0.7455 | 0.7969 | 0.6993 |  |
| TNT | 0.0463 | -0.0431 | 0.0002 | 0.8376 | 0.8899 | 0.7953 |  |
| TQT | 0.0556 | -0.0545 | -0.0036 | 0.7161 | 0.7405 | 0.6990 |  |

### seed 10

| cell | hit | decode | rate | M | center | flank | trip |
|---|---|---|---|---|---|---|---|
| PPP | 0.4815 | 0.7678 | 0.0472 | 1.3570 | 1.8026 | 0.8139 |  |
| TPP | 0.0648 | -0.4421 | -0.0345 | 1.6294 | 1.1074 | 1.9836 |  |
| PTP | 0.1852 | 0.4489 | 0.0413 | 1.3746 | 1.5871 | 1.1024 |  |
| PPT | 0.0972 | -0.3106 | 0.0358 | 0.7116 | 0.6883 | 0.7411 | TRIP |
| TTP | 0.0556 | -0.3795 | -0.0334 | 1.6655 | 1.1247 | 1.8722 |  |
| TPT | 0.2130 | -0.0428 | 0.0454 | 0.3218 | 0.1362 | 0.5704 |  |
| PTT | 0.0880 | -0.3452 | 0.0375 | 0.6975 | 0.6703 | 0.7311 | TRIP |
| TTT | 0.2130 | -0.0692 | 0.0496 | 0.2820 | 0.1290 | 0.5033 |  |
| TRT | 0.0185 | -0.0622 | -0.0021 | 0.7467 | 0.8095 | 0.6815 |  |
| TNT | 0.0417 | -0.0166 | -0.0016 | 0.8305 | 0.8655 | 0.8053 |  |
| TQT | 0.0139 | -0.0842 | 0.0066 | 0.8405 | 0.8715 | 0.8120 |  |

### seed 11

| cell | hit | decode | rate | M | center | flank | trip |
|---|---|---|---|---|---|---|---|
| PPP | 0.4583 | 0.7507 | 0.0480 | 1.3565 | 1.7972 | 0.8221 |  |
| TPP | 0.0741 | -0.3351 | -0.0236 | 1.5405 | 1.1595 | 1.8677 |  |
| PTP | 0.2870 | 0.4889 | 0.0440 | 1.3680 | 1.6715 | 1.0054 |  |
| PPT | 0.1111 | -0.3215 | 0.0426 | 0.6439 | 0.5838 | 0.7140 | TRIP |
| TTP | 0.0741 | -0.3406 | -0.0235 | 1.5853 | 1.1765 | 1.8459 |  |
| TPT | 0.2037 | -0.0628 | 0.0486 | 0.3045 | 0.1202 | 0.5494 |  |
| PTT | 0.1204 | -0.3122 | 0.0410 | 0.6567 | 0.6100 | 0.7236 | TRIP |
| TTT | 0.1759 | -0.1094 | 0.0488 | 0.3091 | 0.1639 | 0.5110 |  |
| TRT | 0.0324 | -0.0530 | 0.0031 | 0.7169 | 0.7597 | 0.6678 |  |
| TNT | 0.0324 | -0.0654 | 0.0027 | 0.7932 | 0.8192 | 0.7667 |  |
| TQT | 0.0370 | -0.0515 | 0.0017 | 0.8437 | 0.8553 | 0.8365 |  |

## rho — alpha0.0 (primaries: center, flank, hit)

### seed 8 — denominators: hit +0.5139, decode -0.3753, rate -0.0340, M -0.3882, center -0.6175, flank +0.0176 (FLOORED)

| cell | rho_hit | rho_decode | rho_rate | rho_M | rho_center | rho_flank |
|---|---|---|---|---|---|---|
| PPP | +0.000 | -0.000 | -0.000 | -0.000 | -0.000 | unrd |
| TPP | -0.387 | +0.930 | +0.395 | +0.083 | +0.004 | unrd |
| PTP | -0.450 | +1.084 | +0.090 | +0.019 | +0.278 | unrd |
| PPT | -0.739 | +1.799 | +1.135 | +0.895 | +1.274 | unrd |
| TTP | -0.505 | +1.185 | +0.433 | +0.142 | +0.152 | unrd |
| TPT | +0.360 | +1.394 | +1.042 | +0.978 | +1.083 | unrd |
| PTT | -0.694* | +1.749* | +1.084 | +0.898 | +1.269 | unrd |
| TTT | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | unrd |
| TRT | -0.856 | +2.015 | +1.383 | +0.865 | +1.306 | unrd |
| TNT | -0.865* | +1.990* | +1.389 | +0.643 | +1.304 | unrd |
| TQT | -0.847 | +2.123 | +1.407 | +0.628 | +1.307 | unrd |
| PRP | -0.865 | +1.920 | +1.387 | -0.265 | +1.258 | unrd |
| PNP | -0.901* | +2.049* | +1.497 | -1.386 | +1.294 | unrd |
| PQP | -0.856* | +2.152* | +1.663 | -1.336 | +1.224 | unrd |

### seed 9 — denominators: hit +0.5185, decode -0.3484, rate -0.0328, M -0.3801, center -0.6017, flank +0.0160 (FLOORED)

| cell | rho_hit | rho_decode | rho_rate | rho_M | rho_center | rho_flank |
|---|---|---|---|---|---|---|
| PPP | +0.000 | -0.000 | -0.000 | -0.000 | -0.000 | unrd |
| TPP | -0.196 | +0.885 | +0.362 | +0.108 | -0.119 | unrd |
| PTP | -0.393 | +1.292 | +0.092 | +0.025 | +0.345 | unrd |
| PPT | -0.589 | +2.017 | +1.051 | +0.894 | +1.287 | unrd |
| TTP | -0.384 | +1.280 | +0.484 | +0.144 | -0.002 | unrd |
| TPT | +0.536 | +1.406 | +1.044 | +0.977 | +1.094 | unrd |
| PTT | -0.563* | +2.023* | +1.011 | +0.900 | +1.280 | unrd |
| TTT | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | unrd |
| TRT | -0.759 | +2.149 | +1.424 | +0.864 | +1.352 | unrd |
| TNT | -0.777* | +2.037* | +1.448 | +0.604 | +1.334 | unrd |
| TQT | -0.839 | +2.202 | +1.481 | +0.607 | +1.355 | unrd |

### seed 10 — denominators: hit +0.5093, decode -0.3459, rate -0.0331, M -0.3811, center -0.5825, flank +0.0107 (FLOORED)

| cell | rho_hit | rho_decode | rho_rate | rho_M | rho_center | rho_flank |
|---|---|---|---|---|---|---|
| PPP | +0.000 | -0.000 | -0.000 | -0.000 | -0.000 | unrd |
| TPP | -0.073 | +0.728 | +0.290 | +0.128 | -0.224 | unrd |
| PTP | -0.500 | +1.236 | +0.025 | +0.045 | +0.297 | unrd |
| PPT | -0.645 | +2.181 | +1.032 | +0.885 | +1.312 | unrd |
| TTP | -0.482 | +1.271 | +0.416 | +0.161 | +0.048 | unrd |
| TPT | +0.409 | +1.453 | +1.043 | +0.978 | +1.098 | unrd |
| PTT | -0.682 | +2.103 | +0.997 | +0.890 | +1.302 | unrd |
| TTT | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | unrd |
| TRT | -0.873 | +2.217 | +1.426 | +0.879 | +1.376 | unrd |
| TNT | -0.864 | +2.172 | +1.403 | +0.587 | +1.358 | unrd |
| TQT | -0.882 | +2.291 | +1.400 | +0.604 | +1.379 | unrd |

### seed 11 — denominators: hit +0.5278, decode -0.3180, rate -0.0338, M -0.3805, center -0.5757, flank +0.0020 (FLOORED)

| cell | rho_hit | rho_decode | rho_rate | rho_M | rho_center | rho_flank |
|---|---|---|---|---|---|---|
| PPP | +0.000 | -0.000 | -0.000 | -0.000 | -0.000 | unrd |
| TPP | -0.219 | +0.965 | +0.337 | +0.107 | -0.126 | unrd |
| PTP | -0.360 | +1.178 | +0.161 | +0.024 | +0.356 | unrd |
| PPT | -0.579 | +2.180 | +1.048 | +0.894 | +1.312 | unrd |
| TTP | -0.386 | +1.574 | +0.479 | +0.128 | +0.219 | unrd |
| TPT | +0.368 | +1.564 | +1.039 | +0.975 | +1.110 | unrd |
| PTT | -0.632 | +2.115 | +1.009 | +0.895 | +1.310 | unrd |
| TTT | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | unrd |
| TRT | -0.746 | +2.364 | +1.412 | +0.881 | +1.382 | unrd |
| TNT | -0.754 | +2.338 | +1.372 | +0.630 | +1.366 | unrd |
| TQT | -0.816* | +2.327* | +1.444 | +0.605 | +1.375 | unrd |

## rho — alpha0.5 (primaries: M, center; rate raw-only per 3.2)

### seed 8 — denominators: hit -0.2778, decode -0.7863, M -1.0593, center -1.6633, flank -0.3103

| cell | rho_hit | rho_decode | rho_M | rho_center | rho_flank |
|---|---|---|---|---|---|
| PPP | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 |
| TPP | +1.400 | +1.502 | -0.209 | +0.404 | -3.462 |
| PTP | +1.000 | +0.289 | -0.012 | +0.081 | -0.619 |
| PPT | +1.417* | +1.326* | +0.691 | +0.748 | +0.476 |
| TTP | +1.500 | +1.469 | -0.251 | +0.409 | -3.472 |
| TPT | +1.067 | +0.955 | +0.979 | +0.996 | +0.897 |
| PTT | +1.367 | +1.313 | +0.697 | +0.753 | +0.470 |
| TTT | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 |
| TRT | +1.567 | +1.059 | +0.561 | +0.594 | +0.341 |
| TNT | +1.567 | +1.011 | +0.478 | +0.555 | -0.041 |
| TQT | +1.667 | +1.003 | +0.459 | +0.543 | -0.103 |
| PRP | +1.600 | +0.916 | -0.097 | +0.467 | -1.309 |
| PNP | +1.583 | +1.040 | -0.605 | +0.460 | -1.966 |
| PQP | +1.633 | +1.181 | -0.439 | +0.455 | -1.978 |

### seed 9 — denominators: hit -0.2454, decode -0.8066, M -1.0904, center -1.7221, flank -0.3163

| cell | rho_hit | rho_decode | rho_M | rho_center | rho_flank |
|---|---|---|---|---|---|
| PPP | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 |
| TPP | +1.566 | +1.550 | -0.240 | +0.424 | -3.889 |
| PTP | +1.038 | +0.440 | -0.011 | +0.121 | -0.804 |
| PPT | +1.321* | +1.320* | +0.626 | +0.687 | +0.272 |
| TTP | +1.660 | +1.554 | -0.280 | +0.427 | -3.822 |
| TPT | +1.094 | +0.929 | +0.927 | +0.958 | +0.731 |
| PTT | +1.434* | +1.318* | +0.623 | +0.683 | +0.274 |
| TTT | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 |
| TRT | +1.698 | +1.075 | +0.558 | +0.591 | +0.348 |
| TNT | +1.642 | +0.984 | +0.474 | +0.537 | +0.044 |
| TQT | +1.604 | +0.998 | +0.585 | +0.624 | +0.349 |

### seed 10 — denominators: hit -0.2685, decode -0.8370, M -1.0750, center -1.6737, flank -0.3106

| cell | rho_hit | rho_decode | rho_M | rho_center | rho_flank |
|---|---|---|---|---|---|
| PPP | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 |
| TPP | +1.552 | +1.446 | -0.253 | +0.415 | -3.767 |
| PTP | +1.103 | +0.381 | -0.016 | +0.129 | -0.929 |
| PPT | +1.431* | +1.289* | +0.600 | +0.666 | +0.234 |
| TTP | +1.586 | +1.371 | -0.287 | +0.405 | -3.408 |
| TPT | +1.000 | +0.969 | +0.963 | +0.996 | +0.784 |
| PTT | +1.466* | +1.330* | +0.614 | +0.677 | +0.266 |
| TTT | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 |
| TRT | +1.724 | +0.992 | +0.568 | +0.593 | +0.426 |
| TNT | +1.638 | +0.937 | +0.490 | +0.560 | +0.027 |
| TQT | +1.741 | +1.018 | +0.480 | +0.556 | +0.006 |

### seed 11 — denominators: hit -0.2824, decode -0.8601, M -1.0474, center -1.6333, flank -0.3111

| cell | rho_hit | rho_decode | rho_M | rho_center | rho_flank |
|---|---|---|---|---|---|
| PPP | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 |
| TPP | +1.361 | +1.262 | -0.176 | +0.390 | -3.361 |
| PTP | +0.607 | +0.304 | -0.011 | +0.077 | -0.589 |
| PPT | +1.230* | +1.247* | +0.680 | +0.743 | +0.347 |
| TTP | +1.361 | +1.269 | -0.218 | +0.380 | -3.291 |
| TPT | +0.902 | +0.946 | +1.004 | +1.027 | +0.876 |
| PTT | +1.197* | +1.236* | +0.668 | +0.727 | +0.316 |
| TTT | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 |
| TRT | +1.508 | +0.934 | +0.611 | +0.635 | +0.496 |
| TNT | +1.508 | +0.949 | +0.538 | +0.599 | +0.178 |
| TQT | +1.492 | +0.933 | +0.490 | +0.577 | -0.046 |

## Pre-registered classification (3.2/3.4 verbatim rules)

### alpha0.0

| cell | verdict |
|---|---|
| PPP | center:0/0/0/0; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:0/0/0/0 |
| TPP | center:0/0_below_baseline/0_below_baseline/0_below_baseline; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:0_below_baseline/0_below_baseline/0_below_baseline/0_below_baseline |
| PTP | center:partial/partial/partial/partial; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:0_below_baseline/0_below_baseline/0_below_baseline/0_below_baseline |
| PPT | center:F/F/F/F; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:0_below_baseline/0_below_baseline/0_below_baseline/0_below_baseline |
| TTP | center:0/0_below_baseline/0/0; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:0_below_baseline/0_below_baseline/0_below_baseline/0_below_baseline |
| TPT | center:F/F/F/F; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:partial/partial/partial/partial |
| PTT | center:F/F/F/F; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:UNRESOLVABLE_TRIP/UNRESOLVABLE_TRIP/0_below_baseline/0_below_baseline [2 seed(s) CE-tripped -> UNRESOLVABLE for competence claims; verdict rests on untripped seeds] |
| TTT | center:F/F/F/F; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:F/F/F/F |
| TRT | center:F/F/F/F; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:0_below_baseline/0_below_baseline/0_below_baseline/0_below_baseline |
| TNT | center:F/F/F/F; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:UNRESOLVABLE_TRIP/UNRESOLVABLE_TRIP/0_below_baseline/0_below_baseline [2 seed(s) CE-tripped -> UNRESOLVABLE for competence claims; verdict rests on untripped seeds] |
| TQT | center:F/F/F/F; flank:UNREADABLE/UNREADABLE/UNREADABLE/UNREADABLE; hit:0_below_baseline/0_below_baseline/0_below_baseline/UNRESOLVABLE_TRIP [1 seed(s) CE-tripped -> UNRESOLVABLE for competence claims; verdict rests on untripped seeds] |
| PRP | center:F; flank:UNREADABLE; hit:0_below_baseline |
| PNP | center:F; flank:UNREADABLE; hit:UNRESOLVABLE_TRIP [1 seed(s) CE-tripped -> UNRESOLVABLE for competence claims; verdict rests on untripped seeds] |
| PQP | center:F; flank:UNREADABLE; hit:UNRESOLVABLE_TRIP [1 seed(s) CE-tripped -> UNRESOLVABLE for competence claims; verdict rests on untripped seeds] |

### alpha0.5

| cell | verdict |
|---|---|
| PPP | M:0/0/0/0; center:0/0/0/0 |
| TPP | M:0_below_baseline/0_below_baseline/0_below_baseline/0_below_baseline; center:partial/partial/partial/partial |
| PTP | M:0_below_baseline/0_below_baseline/0_below_baseline/0_below_baseline; center:0/0/0/0 |
| PPT | M:partial/partial/partial/partial; center:partial/partial/partial/partial [4 seed(s) CE-tripped -> UNRESOLVABLE for competence claims; verdict rests on untripped seeds] |
| TTP | M:0_below_baseline/0_below_baseline/0_below_baseline/0_below_baseline; center:partial/partial/partial/partial |
| TPT | CARRIES (4/4) |
| PTT | M:partial/partial/partial/partial; center:F/partial/partial/partial [3 seed(s) CE-tripped -> UNRESOLVABLE for competence claims; verdict rests on untripped seeds] |
| TTT | CARRIES (4/4) |
| TRT | M:partial/partial/partial/partial; center:partial/partial/partial/partial |
| TNT | M:partial/partial/partial/partial; center:partial/partial/partial/partial |
| TQT | M:partial/partial/partial/partial; center:partial/partial/partial/partial |
| PRP | M:0_below_baseline; center:partial |
| PNP | M:0_below_baseline; center:partial |
| PQP | M:0_below_baseline; center:partial |

## 4.6 predicted strategy map — confrontation

| prediction | outcome |
|---|---|
| sharpening: full carry TTT only | TTT carries: no; other carries: none (flank unreadable 4/4 -> carry read on center+hit per 3.3 fallback) |
| sharpening: TPT hit partial 0.4-0.5 | rho_hit s8 0.360, s9 0.536, s10 0.409, s11 0.368 — HIT (partial 4/4, band 0.36-0.54) |
| sharpening: TTP partial-F on FLANK | rho_flank UNREADABLE 4/4 (floored); descriptively: raw flank TTP 0.754-0.814 <= TTT 0.824-0.828 on 3/4 seeds, s->0 delta -0.150 (kernel active) — descriptive HIT, rho-level UNREADABLE |
| dampening: carry TPT | CARRIES (4/4) — HIT |
| dampening: carry TNT (control) | M:partial/partial/partial/partial; center:partial/partial/partial/partial — MISS (partial, not carry) |
| dampening: carry TQT (control) | M:partial/partial/partial/partial; center:partial/partial/partial/partial — MISS (partial, not carry) |
| dampening: TRT partial | M:partial/partial/partial/partial; center:partial/partial/partial/partial — HIT |
| dampening: TTP overshoots | s8 yes, s9 yes, s10 yes, s11 yes — HIT 4/4 |
| FB geometry: a0.0 low row-cos | measured 0.873-0.883 — MISS (direction largely kept; original A2-c said rewritten) |
| FB geometry: a0.0 E_proj above null (0.078) | measured 0.049, 0.050, 0.054, 0.047 — MISS (below null) |
| FB geometry: a0.5 high row-cos | measured 0.978-0.985 — HIT |
| FB geometry: a0.5 E_proj near null | measured 0.261, 0.252, 0.315, 0.264 — MISS (3-4x above null) |

## Registered questions (4.4) and control hypotheses (4.5)

**Q1 (FB-alone flank at s=0.04):** rho_flank UNREADABLE 4/4 (a0.0 denominators +0.0176/+0.0161/+0.0106/+0.0019, all < 0.05 floor; host flank already at TTT level — see U1). Descriptive: PTP raw flank 0.979-1.068 (ABOVE baseline — FB-alone produces no flank suppression); TTP raw flank 0.754-0.814 (at-or-below TTT); s->0 seed 8: flank(s)-flank(0) = -0.150 (TTP), -0.145 (TTT), -0.063 (PTP), -0.092 (PPP) — the surround path does the flank work when f is well-placed. Placement: hit stays 0_below_baseline for PTP and TTP 4/4. Verdict: flank suppression is more transplantable than placement (prediction HIT descriptively; rho-level unreadable).

**Q2 (dampening GAINS lock):** s8 rho_M(TTP) -0.251, M 1.621 vs host 1.355; s9 rho_M(TTP) -0.280, M 1.659 vs host 1.354; s10 rho_M(TTP) -0.287, M 1.665 vs host 1.357; s11 rho_M(TTP) -0.218, M 1.585 vs host 1.356. Overshoot repeats 4/4 — prediction HIT; dampening remains GAINS-locked.

**Q3 (softmax temperature, R vs N):** prediction was TNT carries (~TPT) and TRT <= TNT. Measured: TNT partial (rho_M 0.474-0.538), and TRT ABOVE TNT on both primaries 4/4 (e.g. s8 rho_M 0.561 vs 0.478) — both clauses MISS. The magnitude-preserving qualifier is NOT what separates controls from TPT; no control reaches carry.

**Q4 (CE-trip census):** a0.5 PPT trips [8, 9, 10, 11] (4/4, repeat); a0.5 PTT trips [9, 10, 11] (3/4 — s8 now untripped, weaker than original 4/4). NEW trips outside the original class: s11:alpha0.0:TQT, s8:alpha0.0:PNP, s8:alpha0.0:PQP, s8:alpha0.0:PTT, s8:alpha0.0:TNT, s9:alpha0.0:PTT, s9:alpha0.0:TNT (a0.0 control/FB-GAINS chimeras trip at s=0.04). Prediction partially HIT (same class repeats) with new a0.0 fragility.

**H-C1 (dampening genericity): REFUTED** (confirmed=False). No control carries on any seed; all three are partial on both primaries 4/4 (rho_M 0.459-0.611, rho_center 0.537-0.635) vs TPT 0.927-1.027. The trained-norm random-direction FB does NOT suffice at s=0.04; the pretrain FB (task-informative) does. Dampening needs a meaningful FB direction, not just magnitude.

**H-C2 (sharpening alignment): CONFIRMED on every resolvable seed.** All controls fail hit; measured rho_hit -0.75..-0.88 (below baseline — a wrong-direction FB actively destroys placement); TQT <= TPT on hit on all 3 resolvable seeds (s11 TQT tripped). (s8/s9 TNT tripped, s11 TQT tripped -> those cells UNRESOLVABLE there.)

**Alignment-criticality A_align = rho_primary(TPT) - rho_primary(TQT):** s8 a0.0 1.207 vs a0.5 0.520; s9 a0.0 1.375 vs a0.5 0.342; s10 a0.0 1.291 vs a0.5 0.482; s11 a0.0 unresolvable vs a0.5 0.515. Sharpening MORE alignment-critical: 3/3 resolvable seeds TRUE (1 seed(s) unresolvable by trip).

**FB premium on hit (1 - rho_hit(TPT), a0.0):** s8 0.640, s9 0.464, s10 0.591, s11 0.632 vs original [0.51, 0.58] — premium PERSISTS (prediction HIT); the kernel did not absorb the FB's placement role.

**U1 (host flank first check): prediction MISSED 4/4** — PPP flank_ratio s8 0.8102, s9 0.8092, s10 0.8139, s11 0.8221 (predicted band 0.85-0.97). The pretrained host already sits at TTT-level flank suppression (0.824-0.828) => the a0.0 flank denominator floors on every seed (R1 materialized; registered fallback applied).

## 8/11 vs 9/10 split (R4)

- alpha0.0 TPP center: 8/11 bands ['0', '0_below_baseline'] vs 9/10 ['0_below_baseline']
- alpha0.0 TQT hit: 8/11 bands ['0_below_baseline', 'UNRESOLVABLE_TRIP'] vs 9/10 ['0_below_baseline']
- alpha0.0 TTP center: 8/11 bands ['0'] vs 9/10 ['0', '0_below_baseline']
- alpha0.5 PTT center: 8/11 bands ['F', 'partial'] vs 9/10 ['partial']

All splits are band-edge wobbles at 0/0_below_baseline or F/partial boundaries plus one trip asymmetry (a0.0 TQT s11); no systematic in-band (8,11) vs sub-band (9,10) divergence on any primary. Reported per R4, not averaged away.

## Deeper analyses (4.1-4.3)

### 4.1 registered question — a0.0 relative ||Delta_fb|| vs original

| seed | rel s=0.04 | rel original | smaller? |
|---|---|---|---|
| 8 | 1.274 | 1.290 | yes |
| 9 | 1.175 | 1.160 | no |
| 10 | 1.155 | 1.152 | no |
| 11 | 1.183 | 1.176 | no |

No decrease (3/4 marginally larger, s8 marginally smaller) — the kernel did NOT absorb part of the FB rewrite (labeled prediction 'modest decrease or no change': lands on 'no change').

### 4.2 FB geometry

| regime x seed | row-cos median | whole-matrix inner | E_proj (null 0.078) | e5(Delta_hh) |
|---|---|---|---|---|
| seed10_alpha0.0 | 0.883 | 0.883 | 0.049 | 0.366 |
| seed10_alpha0.5 | 0.985 | 0.984 | 0.261 | 0.797 |
| seed11_alpha0.0 | 0.881 | 0.878 | 0.050 | 0.366 |
| seed11_alpha0.5 | 0.984 | 0.983 | 0.252 | 0.804 |
| seed8_alpha0.0 | 0.873 | 0.874 | 0.054 | 0.370 |
| seed8_alpha0.5 | 0.978 | 0.977 | 0.315 | 0.800 |
| seed9_alpha0.0 | 0.880 | 0.880 | 0.047 | 0.368 |
| seed9_alpha0.5 | 0.984 | 0.983 | 0.264 | 0.813 |

Original e5(Delta_hh) targets reproduce (a0.5 ~0.80, a0.0 ~0.37). E_proj INVERTS the labeled prediction: the a0.5 FB micro-adjustment reads the Delta_hh top-5 subspace (0.25-0.32 >> null) while the large a0.0 FB rewrite is spread (0.047-0.054, below null).

### 4.3 gains/k

| regime x seed | k | k original (no-surround) | |k| smaller? | som_margin | k pretrain |
|---|---|---|---|---|---|
| seed10_alpha0.0 | 0.0540 | 0.0482 | no | 0.564 | 0.5457 |
| seed10_alpha0.5 | -3.2608 | -3.4653 | yes | 1.774 | 0.5457 |
| seed11_alpha0.0 | 0.0542 | 0.0454 | no | 0.563 | 0.5457 |
| seed11_alpha0.5 | -3.3540 | -3.5278 | yes | 1.797 | 0.5457 |
| seed8_alpha0.0 | 0.0473 | 0.0366 | no | 0.568 | 0.5457 |
| seed8_alpha0.5 | -3.5016 | -3.6932 | yes | 1.833 | 0.5457 |
| seed9_alpha0.0 | 0.0525 | 0.0454 | no | 0.565 | 0.5457 |
| seed9_alpha0.5 | -3.3001 | -3.9058 | yes | 1.784 | 0.5457 |

Same qualitative family both regimes (small-positive vs deep-negative k). |k| smaller than original 4/4 in a0.5 (3.26-3.50 vs 3.69-3.77) as predicted; a0.0 slightly LARGER 4/4 (+0.047..+0.054 vs +0.036..+0.040) — prediction half-MISS.

## s->0 counterfactual (2.4; evidence, never a bar)

Selection: registered 10; rule extras (readable rho_flank >= 0.25): [('alpha0.5', 'PTT'), ('alpha0.5', 'TRT')]; a0.0 factorial floored -> full descriptive set (9 cells) per 3.3.

### registered

| cell | flank(s) | flank(0) | dflank | center(s) | center(0) | M(s) | M(0) |
|---|---|---|---|---|---|---|---|
| alpha0.0_PPP | 0.8102 | 0.9019 | -0.0917 | 1.8070 | 1.8321 | 1.3554 | 1.4127 |
| alpha0.0_PPT | 1.0199 | 1.1573 | -0.1374 | 1.0205 | 1.0479 | 1.0080 | 1.0909 |
| alpha0.0_PTP | 1.0393 | 1.1026 | -0.0633 | 1.6354 | 1.6694 | 1.3481 | 1.4022 |
| alpha0.0_TPT | 0.8505 | 0.9821 | -0.1317 | 1.1383 | 1.1725 | 0.9758 | 1.0640 |
| alpha0.0_TTT | 0.8279 | 0.9730 | -0.1452 | 1.1895 | 1.2235 | 0.9672 | 1.0635 |
| alpha0.5_PPP | 0.8102 | 0.9019 | -0.0917 | 1.8070 | 1.8321 | 1.3554 | 1.4127 |
| alpha0.5_PPT | 0.6626 | 0.7493 | -0.0867 | 0.5635 | 0.5898 | 0.6230 | 0.6842 |
| alpha0.5_PTP | 1.0022 | 1.0833 | -0.0811 | 1.6723 | 1.6985 | 1.3682 | 1.4247 |
| alpha0.5_TPT | 0.5319 | 0.6315 | -0.0996 | 0.1507 | 0.1584 | 0.3188 | 0.3794 |
| alpha0.5_TTT | 0.4999 | 0.6004 | -0.1005 | 0.1436 | 0.1419 | 0.2961 | 0.3500 |

### extras

| cell | flank(s) | flank(0) | dflank | center(s) | center(0) | M(s) | M(0) |
|---|---|---|---|---|---|---|---|
| alpha0.5_PTT | 0.6644 | 0.7475 | -0.0831 | 0.5543 | 0.5786 | 0.6171 | 0.6761 |
| alpha0.5_TRT | 0.7044 | 0.7172 | -0.0128 | 0.8185 | 0.8214 | 0.7611 | 0.7718 |

### extras_floored_descriptive

| cell | flank(s) | flank(0) | dflank | center(s) | center(0) | M(s) | M(0) |
|---|---|---|---|---|---|---|---|
| alpha0.0_PNP | 1.3691 | 1.4234 | -0.0543 | 1.0082 | 1.0220 | 1.8935 | 1.9227 |
| alpha0.0_PQP | 1.3581 | 1.3971 | -0.0391 | 1.0509 | 1.0576 | 1.8739 | 1.9067 |
| alpha0.0_PRP | 1.2163 | 1.2332 | -0.0169 | 1.0304 | 1.0329 | 1.4582 | 1.4809 |
| alpha0.0_PTT | 1.0103 | 1.1553 | -0.1450 | 1.0231 | 1.0529 | 1.0067 | 1.0932 |
| alpha0.0_TNT | 1.0035 | 1.0827 | -0.0792 | 1.0015 | 1.0168 | 1.1057 | 1.1679 |
| alpha0.0_TPP | 0.8568 | 0.9788 | -0.1220 | 1.8046 | 1.8008 | 1.3233 | 1.3852 |
| alpha0.0_TQT | 1.0054 | 1.0783 | -0.0728 | 1.0001 | 1.0116 | 1.1118 | 1.1745 |
| alpha0.0_TRT | 1.0045 | 1.0223 | -0.0178 | 1.0004 | 1.0030 | 1.0196 | 1.0424 |
| alpha0.0_TTP | 0.7587 | 0.9086 | -0.1500 | 1.7131 | 1.7121 | 1.3001 | 1.3691 |

## CE trip census (3.5; threshold 3*ln36 = 10.7506)

- seed 8: alpha0.0:PNP, alpha0.0:PQP, alpha0.0:PTT, alpha0.0:TNT, alpha0.5:PPT
- seed 9: alpha0.0:PTT, alpha0.0:TNT, alpha0.5:PPT, alpha0.5:PTT
- seed 10: alpha0.5:PPT, alpha0.5:PTT
- seed 11: alpha0.0:TQT, alpha0.5:PPT, alpha0.5:PTT

