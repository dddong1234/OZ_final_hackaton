# FE setting DB 대조와 exp_006 선정 근거

원본: [Notion FE setting DB](https://app.notion.com/p/3b67226115e7831fb20881a72793df16)

## SDH와 겹치는 항목

WT 이진화, gene/token burden, mutation type count, 빈도 필터, hotspot, 상수열
제거, token 표현과 class balancing은 exp_003~005에서 이미 수행했거나 공용 모델에
적용되어 있다.

## 아직 직접 검증하지 않은 주요 항목

| 후보 | DB 근거 | 판단 |
| --- | --- | --- |
| burden 3종 | 단일 gene burden 대비 3-seed +0.00456, 3/3 seed 양수 | exp_006 |
| cell token dedup on/off | 팀 표준이나 독립 ablation 없음 | 다음 후보 |
| recurrent missense 확장 | seed 42 Macro F1 0.408125, 단 1-seed | 후속 검증 |
| mutation type ratio | count 대비 +0.00061, 변동 범위 안 | 보류 |
| 계층 분류 | 평면 대비 -0.00545 | 기각 |
| 확률 bias multiplier | 검증 seed +0.00153로 미검출 | 기각 |

## 선정 이유

burden 3종은 fold 밖에서 학습할 vocabulary나 target 통계를 요구하지 않는 행 단위
피처이며, 현재 최고 FE에 숫자 한 개만 추가한다. 따라서 공용 모델 설정을 유지한
상태에서 순수 FE 증분을 가장 명확하게 측정할 수 있다.
