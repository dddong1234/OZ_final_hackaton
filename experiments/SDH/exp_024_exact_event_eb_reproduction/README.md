# SDH exp_024 — Exact-event EB 독립 재현

## 한 줄 요약

기존 `gene × event-type` 암종별 증거를 유지하면서, `TP53__R175H`처럼 정규화된 정확 변이 사건별 암종 증거 26개를 추가한 3-seed 제출 파이프라인을 독립 재현한다.

## 실험 질문

`TP53__MISSENSE`처럼 서로 다른 변이를 하나로 묶는 표현보다 exact event를 Empirical-Bayes class score로 압축한 표현이 일반화 성능을 높이는가?

## 제공 결과

| 항목 | 값 |
| --- | ---: |
| 3-seed OOF Macro F1 | 0.568441 ± 0.002310 |
| H0 대비 평균 OOF 변화 | +0.021186 |
| Public LB | 0.5086 |
| seeds | 42 / 777 / 2024 |

위 값은 제공 문서의 기준값이다. 이 폴더에서 직접 실행한 결과는 실행 후 별도로 기록한다.

## 핵심 구조

1. mutation 문자열을 행 단위로 파싱한다.
2. 구조화 H0 피처와 기존 gene×event-type EB score를 생성한다.
3. fold-train에서 관측된 모든 비상수 exact event에 암종별 posterior log-odds를 학습한다.
4. 희귀 event는 `shrinkage=20`으로 전역 prior에 수축한다.
5. train용 supervised score는 inner 5-fold OOF로 만들고, validation/test에는 train fit을 적용만 한다.
6. exact-event EB LR의 margin이 0.05 미만이면 non-EB LR로 되돌린다.
7. selective LR 0.8과 automatic LGBM specialist 0.2를 결합한다.
8. seeds 42/777/2024의 test 확률을 동일 가중 평균한다.

## 누수 계약

- raw train/test concat 금지
- vocabulary, support, EB weight, scaling은 train에서만 학습
- supervised train feature는 inner cross-fit으로 생성
- 고정 암종·유전자·exact mutation 목록 없음
- test는 최종 transform/predict 단계에서만 읽음

## 실행 방법

저장소 루트에서 JupyterLab을 시작한 뒤 `experiment.ipynb`를 위에서 아래로 실행한다. seed별 학습을 별도 셀로 분리했으므로 진행 상황을 확인하면서 실행할 수 있다.

생성되는 CSV와 audit JSON은 `results/`에 저장되며 Git에 커밋하지 않는다.

## 파일

- `exact_event_pipeline.py`: 제공된 standalone 파이프라인의 저장소 내 사본
- `experiment.ipynb`: 사용자 실행용 셀 단위 워크플로
- `FIXED_CONSTANTS.md`: 제공된 고정 상수와 안전성 계약

## 주의

현재 제공된 standalone 파일은 full-train 제출 재현 경로다. 문서에 기재된 3-seed outer OOF 수치 자체를 다시 산출하는 별도 validation runner는 포함되어 있지 않으므로, 제출 재현과 OOF 독립 재검증을 구분해서 기록한다.

