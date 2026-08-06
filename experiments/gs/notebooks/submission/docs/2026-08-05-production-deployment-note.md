# 운영 배포 경량화 메모

현재 대회 제출 모델은 3-seed bagging, 두 LR 분기, Empirical-Bayes 변환,
자동 LGBM specialist를 포함한다. 학습은 오프라인 배치에 적합하지만, 요청마다
전체 학습을 다시 수행하는 실시간 서비스 구조에는 적합하지 않다.

## 권장 운영 분리

- **대회/오프라인 배치:** 성능 우선 3-seed bagged Selective-EB LR + automatic
  LGBM specialist를 유지한다.
- **실시간 서비스:** 성능이 사전 검증된 단일 seed 또는 단일 LR+EB 모델을 별도로
  운영 후보로 둔다. 지연시간·메모리·운영 복잡도를 낮추기 위한 경량화 모델이며,
  대회 최고 성능 모델을 대체한다는 뜻은 아니다.

## 실시간 추론에 저장할 산출물

1. 유전자 열 순서와 event vocabulary
2. 파싱 규칙, Empirical-Bayes 통계, 표준화 파라미터
3. 경량 LR(+EB) 모델과 클래스 순서
4. Selective-EB를 운영할 때의 고정 margin 규칙
5. 모델·피처 버전 및 입력 schema 검증 규칙

새 운영 모델은 대회 OOF와 별도로 latency, 메모리, 단일/배치 추론 시간, 성능 저하를
검증한 뒤 채택한다.
