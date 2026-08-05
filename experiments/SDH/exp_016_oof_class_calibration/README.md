# SDH exp_016 — OOF class-mass calibration

## 실험 질문

현재 Public LB 챔피언인 exp14의 `safe LR 80% + dynamic specialist LGBM 20%`
확률에 저차원 class-mass 보정을 적용하면 Macro F1이 안정적으로 개선되는가?

## 핵심 아이디어

모델이 어떤 클래스를 실제 train 비율보다 지속적으로 적게 예측하면 그 클래스의
확률에 작은 배율을 주고, 과하게 예측하면 낮춘다. 26개 배율을 직접 탐색하지 않고
다음 한 개의 `alpha`만 탐색한다.

`weight_c = clip((true_mass_c / predicted_mass_c) ** alpha, 0.8, 1.25)`

- `alpha=0`: 원본 exp14와 동일
- `alpha>0`: 예측 class mass를 train class mass 방향으로 일부 보정
- 탐색 범위: 0.0~1.0, 0.1 간격

## 검증 안전장치

1. exp14의 seed 42/52/62 OOF 확률만 사용한다.
2. 두 seed로 alpha와 class weight를 선택하고 남은 seed에서 평가하는 과정을
   세 번 반복한다.
3. holdout seed 3개가 모두 개선될 때만 `PASS`로 판정한다.
4. 26개 class weight를 개별 최적화하지 않아 자유도를 제한한다.
5. test의 class 개수·분포·통계는 보정 선택에 사용하지 않는다.

seed holdout은 동일 train 행에 대한 서로 다른 OOF 분할이므로 완전히 독립적인
외부 검증은 아니다. 따라서 PASS는 안정성 필터이며 LB 상승을 보장하지 않는다.

## 실행

`experiment.ipynb`를 위에서 아래로 한 셀씩 실행한다. OOF 3-seed 생성이 가장
오래 걸리며, 이후 alpha 탐색은 수초 이내다. PASS일 때만 마지막 full-train 및
submission 셀을 실행한다.

class-mass 보정이 FAIL이면 원본 확률을 유지한 채 두 번째 실험으로 진행한다.
두 번째 실험은 LGBM weight 0.05~0.35를 0.025 간격으로 비교하고, 두 seed에서
선택한 weight가 남은 seed에서도 기존 0.20보다 좋아지는지 순환 검증한다. 이 역시
세 holdout이 모두 개선될 때만 제출을 허용한다.

기존 class-mass 결과를 실행한 커널에서는 **1번 exp14 OOF 3-seed 재현 셀부터
다시 실행**해야 LR과 LGBM 확률이 각각 저장된다. 모델 재학습 후 혼합비 탐색은
수초 안에 끝난다.

출력은 `results/`에 저장되며 Git에는 커밋하지 않는다.

## 확정 결과

### 1. Class-mass calibration — FAIL

| Holdout seed | 선택 alpha | 기존 exp14 | 보정 후 | 변화 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.0 | 0.543679 | 0.543679 | 0.000000 |
| 52 | 0.2 | 0.540053 | 0.536669 | **-0.003383** |
| 62 | 0.0 | 0.535802 | 0.535802 | 0.000000 |

평균 변화는 `-0.001128`, 최저 변화는 `-0.003383`이다. 두 holdout은 보정하지
않는 alpha 0을 선택했고, 유일하게 alpha 0.2를 선택한 경우도 남은 seed에서
하락했다. 클래스 예측 질량의 오차 방향이 seed 간 일관적이지 않으므로 폐기한다.

### 2. LR/LGBM blend weight — FAIL

| Holdout seed | 두 fit seed가 선택한 LGBM 비중 | 기존 20% | 선택 비중 적용 | 변화 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 12.5% | 0.543679 | 0.540530 | **-0.003149** |
| 52 | 17.5% | 0.540053 | 0.541153 | +0.001100 |
| 62 | 17.5% | 0.535802 | 0.536235 | +0.000432 |

holdout seed 42에서 하락해 사전 판정 기준인 `3/3 seed 개선`을 통과하지 못했다.
따라서 exp14의 LGBM 20%를 변경하지 않고 제출 파일도 만들지 않았다.

전체 3-seed 결과를 사후 확인하면 LGBM 17.5%가 평균 `0.540513`으로 20%의
`0.539845`보다 `+0.000669` 높고 세 seed가 모두 소폭 개선된다. 그러나 이는 전체
curve를 본 뒤 고른 사후 결과이며, seed-holdout에서 12.5%와 17.5%로 선택이
일치하지 않았다. 상승폭도 작아 신규 챔피언이나 제출 후보로 채택하지 않는다.

## 결론

- class-mass calibration: 폐기
- LGBM 17.5%: 사후 참고값으로만 기록
- 최종 유지: exp14 `LR 80% + dynamic specialist LGBM 20%`
- Public LB 챔피언: **0.4489813603**
