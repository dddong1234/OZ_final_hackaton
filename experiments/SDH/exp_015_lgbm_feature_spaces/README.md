# SDH exp_015 — LGBM 전용 피처 공간 탐색

## 실험 질문

exp14의 balanced multiclass LGBM 파라미터를 고정한 채, 어떤 피처 공간이 LGBM에 가장 잘 맞는가?

## 고정 조건

- 5-fold Stratified CV
- 1차 screen: seed 42
- 최종 확인: 상위 후보 seed 42/52/62
- LGBM: 400 trees, learning rate 0.05, leaves 25, balanced class weight
- 외부 데이터와 test 통계 미사용
- vocabulary, enrichment, recurrent support, gain 선택은 매 outer fold의 train에서만 학습
- 고정 암종쌍 `C__`와 고정 exact mutation `D__exact`는 생성 결과에서 제거하며 assertion으로 확인
- exact 위치 신호는 fold-train support 기반 `R__` recurrent missense만 사용

## 비교 축

총 26개 case를 네 묶음으로 실행한다.

1. 표현 축: full, core, enrichment 제거/단독, gene 조합
2. 희귀 gene 축: mutation gene 최소 support 2/5/10/20/30/50
3. 블록 축: truncation, recurrent missense, amino pair, exact event의 독립 기여와 고정 count bin
4. 선택 축: fold-train gain Top-250/500/1000/2000

피처 screen 뒤에는 다음 앙상블도 검사한다.

- LR 챔피언 대비 LGBM 예측 불일치율
- LR 90~50% + LGBM 10~50% 확률 혼합
- 상위 LGBM 피처 공간끼리 25:75, 50:50, 75:25 혼합
- seed 42 상위 LR+LGBM 조합의 3-seed 재확인

세부 정의는 `EXPERIMENT_CASES.md`에 있다. 결과 파일은 `results/`에 저장하되 Git에는 올리지 않는다.

## 실행

저장소 루트에서 JupyterLab을 열고 `experiment.ipynb`를 위에서부터 한 셀씩 실행한다. 오래 걸리는 그룹은 셀이 분리되어 있으므로 중간 결과를 확인하고 다음 그룹으로 진행할 수 있다.

## 확정 결과

### Seed 42 단일 LGBM

| Case | OOF Macro F1 | 평균 피처 수 | Full 대비 |
| --- | ---: | ---: | ---: |
| Gain Top-250 | **0.486070** | 250 | +0.009758 |
| Full + count bins | 0.484221 | 8,259.2 | +0.007909 |
| Gain Top-1000 | 0.480387 | 1,000 | +0.004075 |
| Full | 0.476313 | 8,193.2 | 기준 |

### 상위 단일 LGBM 3-seed

| Case | 평균 | 표준편차 |
| --- | ---: | ---: |
| Gain Top-1000 | **0.482325** | 0.003168 |
| Gain Top-250 | 0.480477 | 0.006432 |
| Full + count bins | 0.480286 | 0.007836 |

seed 42 최고는 Top-250이지만, 3-seed 평균과 안정성은 Top-1000이 가장 좋다.

### LR 85% + LGBM 15% 3-seed

| LGBM 피처 | Blend 평균 | LR 대비 평균 | 최소 상승 |
| --- | ---: | ---: | ---: |
| Gain Top-1000 | **0.537380** | +0.009771 | +0.006460 |
| Full + count bins | 0.537041 | +0.009432 | +0.006949 |
| Gain Top-250 | 0.536647 | +0.009039 | +0.006294 |

세 후보 모두 3개 seed에서 안전 LR baseline을 개선했다. 그러나 exp14의
train-discovered specialist 앙상블 평균 `0.539845`보다 낮아 새 챔피언으로
채택하지 않는다.

## 피처 ablation 결론

- amino-pair 제거: `0.476313 → 0.431736`으로 크게 하락하므로 필수 블록이다.
- mutation gene 제거: `0.476313 → 0.463374`로 하락하므로 유지한다.
- truncation/recurrent 제거: `0.476313 → 0.473678`로 소폭 하락한다.
- core에서 enrichment 제거: `0.372325 → 0.180322`로 붕괴하므로 축소 표현의 핵심이다.
- gene support 2/5/10은 같은 `0.428866`이며, 20 이상부터 하락해 강한 support 필터는 기각한다.
- 고정 count bins는 seed 42에서 유효하지만 Top-1000보다 seed 변동성이 크다.
- LGBM끼리의 최고 blend `0.490092`보다 LR과의 blend가 훨씬 높아 모델 계열 다양성이 중요하다.

## 최종 판정

- 채택 지식: Gain Top-1000, fixed count bins의 유효성, A/G/E 블록 유지
- exp15 모델 자체: 보조 후보
- 최종 챔피언: exp14 `LR 80% + train-discovered hard specialist LGBM 20%`
- 15% blend weight는 seed 42 OOF 탐색값이므로 최종 제출 확정 전 nested 검증이 필요하다.

세부 수치와 해석은 [RESULT_SUMMARY.md](RESULT_SUMMARY.md)에 기록한다.
