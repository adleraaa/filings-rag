| subset | n | correct | incorrect | refusal | accuracy |
|---|---|---|---|---|---|
| all | 150 | 83 | 15 | 52 | 0.553 |
| domain-relevant | 50 | 18 | 7 | 25 | 0.360 |
| metrics-generated | 50 | 38 | 2 | 10 | 0.760 |
| novel-generated | 50 | 27 | 6 | 17 | 0.540 |
| gold page retrieved | 92 | 70 | 9 | 13 | 0.761 |
| gold page not retrieved | 58 | 13 | 6 | 39 | 0.224 |

Citation check on 96 non-refusal answers:

| metric | verbatim | arithmetic |
|---|---|---|
| flagged | 35 (0.3646) | 3 (0.0312) |
| judge accuracy, flagged | 0.9143 | 1.0 |
| judge accuracy, unflagged | 0.8197 | 0.8495 |
| AUROC flag -> wrong | 0.412 | 0.4817 |

AUROC low answer/page similarity -> wrong: 0.4268
