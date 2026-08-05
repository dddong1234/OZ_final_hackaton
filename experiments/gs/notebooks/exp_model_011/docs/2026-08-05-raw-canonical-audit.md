# Raw Mutation vs Canonical Profile Audit

## 목적

현재 H0는 mutation 문자열을 `gene × functional event type`으로 압축한다. 이 감사는 원문 mutation 문자열과 정확 이벤트 수준 표현이 이 압축 과정에서 사라지는 정보를 갖는지 train 데이터만으로 확인한다. 모델 학습, 앙상블, 제출 파일 생성은 하지 않는다.

## 비교 표현

| 표현 | 정의 | 확인하려는 정보 |
|---|---|---|
| raw | 각 gene 셀의 원문 문자열 | 표기·복수 event·정확 변이 정보 |
| canonical_event | 대소문자·`p.` 접두사·구분자를 정규화한 `gene=event` | 정확 변이 구조 |
| gene_type | H0 호환 `gene__functional_type` 집합 | H0가 직접 쓰는 coarse 구조 |

WT, 공백, NaN은 세 표현에서 event 0개다. 복수 event 셀은 구분자를 기준으로 각각 한 번씩 분리하며, 파싱하지 못한 event도 `OTHER`로 보존한다.

## 규정 계약

- `train.csv`만 읽는다. `test.csv`는 경로도 열지 않는다.
- profile vocabulary, 통계, 선택, 스케일링을 test에 fit하지 않는다.
- 암종명·유전자명·exact mutation을 고정 규칙으로 쓰지 않는다.
- 입력 label은 purity 집계에만 사용하며, 분류기나 threshold는 학습하지 않는다.

## 실행

노트북 `exp/exp-raw-canonical-audit-01.ipynb`에서 `RUN_EXPERIMENT=True`로 바꾼 뒤 실행한다. 먼저 `--smoke`가 아니라 전체 train-only 감사를 실행한다.

## 결과 해석

1. `raw`와 `canonical_event`의 차이는 표기 정규화가 합친 정보다.
2. `canonical_event`와 `gene_type`의 차이는 H0의 coarse functional-type 압축이 합친 정확 이벤트 정보다.
3. raw/canonical의 weighted purity가 gene_type보다 높고, canonical-to-gene_type 병합 행도 충분하면 raw-token 모델을 **후보**로 검토한다.
4. 이 감사만으로 성능 향상을 주장하지 않는다. 다음 단계는 별도 seed42 OOF raw-token 오류 다양성 감사이며, 여기서 기준을 통과해야만 3-seed로 확장한다.

## 생성 결과

- `*_summary.csv`: 표현별 unique profile, duplicate rows, weighted purity
- `*_transition_summary.csv`: 표현 전환에서 병합된 profile/행 수
- `*_profile_purity_hashed.csv`: 원문을 저장하지 않는 hashed profile-level purity 분포
- `*_event_type_counts.csv`: parser functional-type 분포
- `*_audit.json`: test 미열람, segment 보존, NaN 계약 및 다음 단계 후보 여부
