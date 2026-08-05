# Complement NB Mutation-profile Blend 실험 결과

## 결론

`Complement NB` 변이 프로필 모델은 H0와 독립적인 확률을 만들었지만, 사전 고정한 `0.80 × H0 + 0.20 × Complement NB` 결합에서 최종 1위 예측을 바꾸지 못했습니다. 따라서 **seed42 screen 기준 기각**이며, 3-seed 확정 검증이나 제출 후보로 승격하지 않습니다.

## 실험 목적

기준 H0는 Selective-EB Logistic Regression과 자동 혼동쌍 LGBM specialist를 결합한 판별 모델입니다. 이번 실험은 각 암종의 전형적인 **변이 유전자 조합(profile)** 을 보는 Complement NB가 H0와 다른 오류를 낼 수 있는지 확인하기 위해 설계했습니다.

단독 점수가 낮더라도, H0가 틀린 일부 행을 보완하면 확률 앙상블에서 성능이 오를 수 있다는 가설입니다.

## 고정 구성

| 구분 | 구성 |
| --- | --- |
| 기준 모델 | H0 Selective-EB LR + 자동 LGBM specialist |
| 보조 모델 | `ComplementNB(alpha=1.0, norm=True)` |
| NB 입력 | fold-train에서 만든 유전자별 mutation binary profile |
| 확률 결합 | `0.80 × H0 + 0.20 × Complement NB` |
| 탐색 여부 | NB 파라미터·앙상블 비율 재탐색 없음 |
| 검증 | Stratified 5-fold, seed 42 OOF screen |
| 지표 | Macro F1, Accuracy |

Complement NB는 유전자별 변이 여부의 클래스 조건부 분포를 이용해, 샘플의 변이 프로필이 각 암종에 얼마나 부합하는지 계산합니다. H0의 구조화·Empirical-Bayes 신호와는 다른 확률적 관점을 제공하는지가 핵심 검증 대상이었습니다.

## 규정 및 누수 방지

- seed42 OOF 실행에서 `train.csv`만 읽고 `test.csv`는 읽지 않음
- train/test 결합 없음
- 각 outer fold의 train으로만 NB vocabulary와 NB 모델을 학습
- H0의 vocabulary, EB 통계, 자동 specialist도 outer-fold train으로만 학습
- validation fold는 transform 및 평가에만 사용
- 고정 암종명·유전자명·exact mutation 목록을 규칙으로 사용하지 않음
- WT, 빈 문자열, NaN은 event로 만들지 않음

감사 결과는 모든 fold에서 `leakage_check=True`, `nan_as_mutation_count=0`, 수렴 경고 `0`건입니다.

## Seed42 OOF 결과

| 모델 | OOF Macro F1 | OOF Accuracy | 평균 피처 수 | H0 대비 |
| --- | ---: | ---: | ---: | ---: |
| H0 Selective-EB | 0.547915 | 0.558942 | 8,219.2 | — |
| Complement NB 단독 | 0.197222 | 0.251734 | 4,228.6 | -0.350693 |
| H0 + Complement NB | 0.547915 | 0.558942 | 12,447.8 | +0.000000 |

fold별 Macro F1도 H0와 blend가 완전히 동일했습니다.

| Fold | H0 | H0 + Complement NB | 변화 |
| --- | ---: | ---: | ---: |
| 1 | 0.528516 | 0.528516 | 0.000000 |
| 2 | 0.551891 | 0.551891 | 0.000000 |
| 3 | 0.543449 | 0.543449 | 0.000000 |
| 4 | 0.554759 | 0.554759 | 0.000000 |
| 5 | 0.558499 | 0.558499 | 0.000000 |

## 추가 확인

NB가 전혀 작동하지 않은 것은 아닙니다. 저장된 OOF 확률을 비교하면 blend와 H0 사이의 평균 절대 확률 변화는 `0.010169`, 최대 변화는 `0.192305`였습니다. 그러나 **6,201개 행 중 최종 Top-1 예측이 바뀐 행은 0개**였습니다. 따라서 Macro F1과 클래스별 F1도 전부 동일했습니다.

이는 현재의 20% NB 보정이 H0의 순위를 뒤집을 정도로 강한 보완 신호를 제공하지 못했다는 뜻입니다. NB 단독 Macro F1도 `0.197222`로 낮아, H0의 잔여 오류를 안정적으로 회복한다는 근거가 없습니다.

## 판정 및 다음 조치

사전 승격 기준은 H0 대비 `+0.003` 이상, 5개 fold 중 최소 4개 상승이었습니다. 이번 결과는 delta `0.000000`, 상승 fold `0/5`이므로 **기각**입니다.

- 3-seed 확정 검증: 진행하지 않음
- 앙상블 비율 또는 `alpha` 재탐색: 진행하지 않음
- 제출 모델 포함: 하지 않음

이번 결과는 “binary mutation-profile Complement NB를 H0에 20% 섞는 방식”이 H0의 의사결정을 바꾸지 못한다는 결론입니다. 이후 앙상블은 단독 점수뿐 아니라 H0와의 실제 Top-1 오류 다양성 및 fold별 개선을 먼저 확인하는 방향이 필요합니다.

## 재현 자료

- 실행 노트북: `exp/exp-h0-complement-nb-profile-blend-01.ipynb`
- 실행기: `common/run_h0_complement_nb_profile_blend.py`
- seed42 결과: `result/exp-h0-complement-nb-profile-blend-01_seed42_summary.csv`
- fold 감사: `result/exp-h0-complement-nb-profile-blend-01_seed42_fold_audit.csv`
- 누수 감사: `result/exp-h0-complement-nb-profile-blend-01_seed42_leakage_audit.json`
