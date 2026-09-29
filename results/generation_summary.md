Judge verdicts:

| subset | n | correct | incorrect | refusal | accuracy |
|---|---|---|---|---|---|
| all | 150 | 83 | 15 | 52 | 0.553 |
| domain-relevant | 50 | 18 | 7 | 25 | 0.360 |
| metrics-generated | 50 | 38 | 2 | 10 | 0.760 |
| novel-generated | 50 | 27 | 6 | 17 | 0.540 |
| gold page retrieved | 92 | 70 | 9 | 13 | 0.761 |
| gold page not retrieved | 58 | 13 | 6 | 39 | 0.224 |

Answer status (rule) vs judge verdict:

| status | correct | incorrect | refusal |
|---|---|---|---|
| answered | 82 | 13 | 1 |
| partial | 1 | 2 | 4 |
| refusal | 0 | 0 | 47 |

Citation check on 103 answered or partial answers (20 not judged correct):

| metric | verbatim | arithmetic | arithmetic_loose |
|---|---|---|---|
| flagged | 36 (0.3495) | 13 (0.1262) | 3 (0.0291) |
| wrong answers flagged / unflagged | 4 / 16 | 1 / 19 | 0 / 20 |
| judge accuracy, flagged | 0.8889 | 0.9231 | 1.0 |
| judge accuracy, unflagged | 0.7612 | 0.7889 | 0.8 |
| accuracy gap [95% CI] | 0.1277 [-0.0192, 0.266] | 0.1342 [-0.066, 0.2747] | 0.2 [0.1208, 0.2828] |
| AUROC flag -> wrong [95% CI] | 0.4072 [0.309, 0.515] | 0.4527 [0.4, 0.5245] | 0.4819 [0.4593, 0.5] |
| false-accept rate, x1.1, all numbers | 0.1325 (60/453) | 0.1414 (70/495) | 0.1593 (83/521) |
| false-accept rate, x1.1, numbers <100 | 0.3677 (57/155) | 0.3495 (65/186) | 0.4053 (77/190) |
| false-accept rate, x1.1, numbers >=100 | 0.0101 (3/298) | 0.0162 (5/309) | 0.0181 (6/331) |
| false-accept rate, digit_swap, all numbers | 0.1059 (43/406) | 0.1253 (56/447) | 0.1459 (69/473) |
| false-accept rate, digit_swap, numbers <100 | 0.3419 (40/117) | 0.3311 (49/148) | 0.3947 (60/152) |
| false-accept rate, digit_swap, numbers >=100 | 0.0104 (3/289) | 0.0234 (7/299) | 0.028 (9/321) |
| false-accept rate, random, all numbers | 0.1694 (82/484) | 0.1657 (88/531) | 0.182 (101/555) |
| false-accept rate, random, numbers <100 | 0.414 (77/186) | 0.3874 (86/222) | 0.4375 (98/224) |
| false-accept rate, random, numbers >=100 | 0.0168 (5/298) | 0.0065 (2/309) | 0.0091 (3/331) |

AUROC low answer/page similarity -> wrong: 0.4199
