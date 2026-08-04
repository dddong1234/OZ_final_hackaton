# Selective EB gate: 새 seed 확정 검증

## 고정 규칙

- P1+Empirical-Bayes의 행별 Top-1 − Top-2 확률 margin이 `0.05` 미만이면 P1 non-EB 확률을 사용한다.
- margin이 `0.05` 이상이면 P1+EB 확률을 유지한다.
- `0.05`는 기존 seeds `42/777/2024`의 취약구간 감사에서 이미 정했으며 재탐색하지 않는다.

## 검증

- 새 seeds: `31415 / 52 / 62`.
- 모델·피처·EB 파라미터·blend는 변경하지 않는다.
- 모든 supervised 통계는 outer fold-train만 사용한다. test는 읽지 않고 제출파일도 생성하지 않는다.

## 채택 기준

- 세 seed 모두 P1+EB보다 상승
- 평균 개선 `+0.003` 이상, 15 folds 중 11개 이상 상승
- low-margin 구간 개선, 클래스별 치명적 하락 없음
- 수렴 경고 0, `leakage_check=True`, `nan_as_mutation_count=0`
