# H0 Selective-EB 3-seed 재현 제출

실행 파일: `reproduce_h0_selective_eb_3seed.py`

```bash
/path/to/.venv/bin/python reproduce_h0_selective_eb_3seed.py
```

고정 계약:

- full train 학습 seed: `42`, `777`, `2024`
- 세 확률 행렬은 각 `1/3`로 평균하며 가중치 탐색은 하지 않는다.
- train에서만 vocabulary, recurrent event, EB 통계, 표준화, 자동 specialist 쌍을 학습한다.
- test는 train-fitted 변환 및 최종 예측에만 사용한다.
- 고정 암종명·유전자명·exact mutation 목록은 사용하지 않는다.
- 출력: `submission_h0_selective_eb_lr_lgbm_specialist_seed42_777_2024_bagged.csv` 및 같은 이름의 audit JSON.

실행 종료 시 audit이 `leakage_check=True`, `nan_as_mutation_count=0`,
`raw_train_test_concat=False`인지 자동 검증한다.
