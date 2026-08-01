# exp_009 실험 요약

## 질문

exp_007의 최고 FE(functional full)에 변이 표기에서 직접 읽을 수 있는 단백질
구조를 추가하면 성능이 개선되는가?

## 실험 설계

- LR: `lbfgs`, `C=0.07`, `max_iter=2000`, `class_weight=balanced`
- 검증: Stratified 5-fold
- 1차: seed 42로 9개 FE 후보 비교
- 확인: 1~3위 후보를 seed 42/52/62로 재평가
- 데이터 의존 목록과 zero-variance 열은 fold train에서만 산출
- 외부 유전자·암종 annotation 및 test 통계 미사용

## 후보 블록

- A: ref/alt 아미노산, ref→alt pair 380종, 단백질 위치 구간
- S: 한 유전자 내 이벤트 수, notation type 다양도·entropy·dominant share

## 결과

| 순위 | Case | 3-seed OOF Macro F1 | 기준 대비 |
| ---: | --- | ---: | ---: |
| 1 | `case_06_plus_A_pair` | **0.46360 ± 0.00109** | **+0.03171** |
| 2 | `case_04_plus_A_plus_S` | 0.43614 ± 0.00323 | +0.00425 |
| 3 | `case_01_functional_full_reference` | 0.43189 ± 0.00325 | 기준 |

모든 15개 확인 fold에서 수렴 경고는 0건이었다. pair 블록만 추가한 경우가 A/S
전체 조합보다 높아, 표기 구조를 많이 넣기보다 치환 방향의 빈도를 압축하는 편이
유리하다는 결과를 얻었다.

## 제출 결과와 해석

`case_06_plus_A_pair`를 전체 train에 fit해 만든 제출의 실제 LB Macro F1은
**0.34238**이었다. OOF 0.46360과의 차이는 **0.12122**이며, 참고 구현
`biodomain02`의 LB 0.35097보다 0.00859 낮다.

따라서 pair 블록은 로컬 CV 1위이지만 최종 일반화 FE로 확정하지 않는다. 실제 test의
pair 분포 차이, 고차원 pair 열의 과적합, 문자열 형식 차이를 후속 검증한다.

## 후속 연구

1. pair별 fold-train 등장 빈도 threshold ablation
2. pair count의 binary/log1p/raw 표현 비교
3. fold별 pair 안정성 및 train/test 표기 분포 점검
4. 위 검증 전에는 모델 파라미터나 외부 annotation을 추가하지 않음
