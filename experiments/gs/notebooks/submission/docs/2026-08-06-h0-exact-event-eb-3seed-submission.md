# H0 + exact-event Empirical-Bayes 3-seed 제출

검증된 `H0_selective_EB`의 LR 분기만 자동 exact-event EB 분기로 교체한 제출 구성이다.

## 고정 구성

- seeds: `42, 777, 2024`, 각 확률을 동일 가중치로 평균
- 구조화 입력: mutation binary, 변이량/유형, truncation, fold-train recurrent missense, A-pair `log1p`, topology, gene×event-type enrichment
- 신규 입력: full-train에서 관찰된 모든 `gene__normalized_event`의 posterior-shrunk 26-class EB score
- 선택적 gate: exact-EB LR 확률의 top-1/top-2 margin `< 0.05`이면 non-EB LR, 그 외 exact-EB LR
- 자동 specialist: full-train 유전자 binary 중심 centroid로 유사한 암종쌍 2개를 자동 발견하고, LGBM으로 pair 내부 확률만 교체
- 최종 seed 확률: `0.80 × gated LR + 0.20 × automatic specialist LGBM`

## 검증 결과

3-seed OOF에서 exact-event EB는 H0 대비 `+0.021186 ± 0.002115` Macro F1을 기록했고, 세 seed 및 15개 fold 모두 상승했다.

## 규정 계약

event vocabulary, EB 통계, 표준화, recurrent feature, specialist 쌍은 seed별 **full train만**으로 fit한다. test는 final transformation/prediction에만 사용하며, train/test concat, test 기반 vocabulary/scaling/선택은 없다. WT·빈 문자열·NaN은 event로 만들지 않는다.

## 실행

```bash
/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python \
  experiments/gs/notebooks/submission/generate_submission_h0_exact_event_eb_3seed.py
```

생성 파일은 `experiments/gs/notebooks/submission/submission_h0_exact_event_eb_seed42_777_2024_bagged.csv` 및 동일 이름의 audit JSON이다.
