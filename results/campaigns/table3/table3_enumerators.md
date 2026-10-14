# Table III — Shor–Laflamme weight enumerators (hardware basis, j ≤ 2)

Wire-grouped path (reproduce_tables.enumerators). B_w − A_w = 0 for w < dist is the (exact-)detection condition; trained codes satisfy it only approximately (the KL residual is a conservative certificate and does not order finite-p LER).

| code | K | A1 | A2 | B1 | B2 | B1−A1 | B2−A2 | shown in |
|---|---|---|---|---|---|---|---|---|
| ((5,3,3))_3 published L10 ring | 3 | 2.5009 | 2.7525 | 2.5082 | 2.8499 | 0.0073 | 0.0974 | Fig 4b, Fig 6 n/K/q/dist (central code, loss_ler_diag) |
| ((5,3,3))_3 central code L2 a2a | 3 | 2.4408 | 2.4229 | 2.4631 | 3.1225 | 0.0223 | 0.6996 | Fig 4b, Fig 6 n/K/q/dist (central code, r1_campaign) |
| [[5,1,3]]_Z3 (analytic) | 3 | 2.2222 | 1.9753 | 2.2222 | 1.9753 | -0.0000 | 0.0000 | Fig 4b, Fig 6 n/K/q/dist (stabilizer benchmark) |
| ((5,3,3))_3 L16 a2a | 3 | 3.5135 | 6.4561 | 3.5156 | 6.5717 | 0.0021 | 0.1155 | Fig 4, Fig 6 n/K/q/dist |
| ((6,3,3))_3 L16 a2a | 3 | 2.9644 | 3.8211 | 2.9652 | 3.8636 | 0.0008 | 0.0425 | Fig 4, Fig 6 n |
| ((7,3,3))_3 L16 a2a | 3 | 3.2730 | 4.5651 | 3.2733 | 4.5862 | 0.0003 | 0.0212 | Fig 4, Fig 6 n |
| ((8,3,3))_3 L16 a2a | 3 | 3.6471 | 5.8226 | 3.6472 | 5.8325 | 0.0001 | 0.0099 | Fig 4, Fig 6 n |
| ((9,3,3))_3 L16 a2a | 3 | 4.0247 | 7.1854 | 4.0248 | 7.1899 | 0.0001 | 0.0046 | Fig 4, Fig 6 n |
| ((5,2,3))_3 L16 a2a | 2 | 3.3825 | 5.5743 | 3.3825 | 5.5774 | 0.0001 | 0.0032 | Fig 6 K |
| ((5,3,2))_3 L16 a2a | 3 | 2.2782 | 2.1423 | 2.3098 | 3.6405 | 0.0316 | 1.4982 | Fig 6 dist |
| ((5,3,3))_4 L16 a2a | 3 | 8.0789 | 26.3321 | 8.0793 | 26.3794 | 0.0004 | 0.0472 | Fig 6 q |
| ((5,3,3))_5 L16 a2a | 3 | 14.6250 | 85.6191 | 14.6251 | 85.6446 | 0.0001 | 0.0254 | Fig 6 q |
| ((9,3,3))_3 weighted winner | 3 | 3.9972 | 7.0944 | 3.9972 | 7.0996 | 0.0000 | 0.0052 | Fig 10 |
| ((7,3,2))_3 distance panel L2 | 3 | 3.1101 | 4.1689 | 3.1195 | 4.5015 | 0.0094 | 0.3326 | distance panel (t1b_dist_panel_n7) |
| ((7,3,3))_3 distance panel L2 | 3 | 3.1706 | 4.3496 | 3.1735 | 4.5137 | 0.0030 | 0.1641 | distance panel + matched family n7 |
| ((7,3,4))_3 distance panel L2 | 3 | 3.2137 | 4.5295 | 3.2249 | 4.7140 | 0.0111 | 0.1845 | distance panel |
| ((6,3,3))_3 matched L2 | 3 | 2.7699 | 3.2730 | 2.7783 | 3.6047 | 0.0084 | 0.3317 | matched family (t4_matched_family) |
| ((8,3,3))_3 matched L2 | 3 | 3.6052 | 5.7419 | 3.6062 | 5.8135 | 0.0010 | 0.0716 | matched family |
| ((9,3,3))_3 matched L2 | 3 | 4.0181 | 7.1794 | 4.0186 | 7.2096 | 0.0005 | 0.0302 | matched family |
| ((5,2,3))_3 minimal-L L2 | 2 | 2.3935 | 2.5172 | 2.3966 | 2.6648 | 0.0031 | 0.1476 | K panel |
| ((7,2,3))_3 minimal-L L2 | 2 | 3.1458 | 4.2597 | 3.1461 | 4.2918 | 0.0003 | 0.0320 | K panel (n=7 trio) |
| ((5,3,3))_4 minimal-L L2 | 3 | 7.9495 | 25.2795 | 7.9552 | 25.6077 | 0.0057 | 0.3281 | L-optimality / q panel |
| ((5,3,3))_4 weighted L2 | 3 | 7.7117 | 23.9410 | 7.7163 | 24.2102 | 0.0045 | 0.2691 | rebuilt q panel (fig2_q_panel_r3) |
| ((5,3,3))_5 weighted L2 | 3 | 14.6438 | 85.7968 | 14.6453 | 85.9356 | 0.0014 | 0.1387 | rebuilt q panel (fig2_q_panel_r3) |
| ((7,3,4))_3 WEIGHTED L2 | 3 | 3.1867 | 4.3922 | 3.1923 | 4.5561 | 0.0056 | 0.1639 | weighted distance trio (t2c; d2/d3 arms reproduce the r2_campaign t1a codes bit-identically -- rows r2_d2n7/r2_d3n7 stand) |
