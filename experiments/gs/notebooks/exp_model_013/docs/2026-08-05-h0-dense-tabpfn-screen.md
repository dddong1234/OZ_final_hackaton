# exp-h0-dense-tabpfn-screen-01

## 목적

H0 Selective-EB + automatic LGBM specialist의 확률을 유지한 채, H0에서 이미 검증된 암종별 EB 증거와 구조 요약을 저차원 dense 입력으로 TabPFN에 제공한다. 고차원 원본 mutation one-hot을 TabPFN에 입력하지 않는다.

## 고정 구성

- seed42, Stratified 5-fold.
- H0: 현재 규정 안전 Selective-EB LR + automatic specialist LGBM.
- TabPFN: 고정 TabPFN V3 classifier 한 개(`n_estimators=8`, `random_state=4201~4205`). AutoTabPFN/HPO/feature grid 없음.
- 결합: `0.80 × H0 + 0.20 × TabPFN` 한 가지만 평가.
- screen에서 `test.csv`를 읽지 않는다.

## 입력 피처

- fold-train gene×event-type Empirical-Bayes 26 class score.
- mutation burden, event-type count, truncation count.
- A-pair `log1p` count와 topology 요약.
- EB 양/음/절대 evidence 합과 최댓값 요약.

vocabulary, EB 가중치, recurrent event, 선택 열, standardization은 모두 outer-fold train에서만 fit한다. validation은 transform/predict/evaluation만 한다. 특정 암종명·유전자·exact mutation 목록은 코드에 고정하지 않는다.

## 실행 전 준비

공식 패키지 설치 후 최초 실행 시 model weight 다운로드 및 라이선스 인증이 필요할 수 있다.

```bash
/Users/admin/Documents/FinalProject/OZ_fianl_hackaton/.venv/bin/python -m pip install tabpfn
```

CPU에서는 6,201행 × 5-fold가 매우 오래 걸릴 수 있다. 가능하면 CUDA 환경에서 실행하고, CPU이면 실행기의 `TABPFN_ALLOW_CPU_LARGE_DATASET=true` 설정을 인지한 상태로 진행한다. 본 실험은 외부 환자 데이터·유전자 주석·단백질 서열을 사용하지 않는다.

## 판정

H0 seed42 OOF가 `0.547915 ± 0.001`을 먼저 재현해야 TabPFN fit을 시작한다.

- H0 대비 `+0.015` 이상, 5 fold 중 4 fold 이상 상승: 3-seed 검증 후보.
- 그 외: 미검출이며 TabPFN 축 종료.

결과에는 package version, device, leakage check, NaN mutation count, fold/class/Top-k 지표를 저장한다.
