# Jupyter 개인 실험 공간 안내

`member_a`부터 `member_d`까지 각 팀원의 개인 작업 공간입니다.

## 필수 규칙

- 자신의 `member_x/`만 수정합니다.
- 원본 데이터는 저장소 루트의 `data/raw/`에서 읽고 수정하지 않습니다.
- JupyterLab을 저장소 루트에서 실행하고 자신의 `notebooks/`에서 실험합니다.
- 개인 폴더 내부 구조와 실행 방식은 자유입니다.
- 팀에 결과를 공유할 때만 루트 `README.md`의 결과 규격을 따릅니다.
- 공용 `common/`, `configs/`, 환경 파일 변경은 팀에 먼저 공유합니다.

## Quick Start

각 팀원의 `notebooks/00_quick_start.ipynb`를 열고 위에서 아래로 실행하면 데이터 확인,
전처리, 학습, Macro F1 평가와 submission 저장까지 체험할 수 있습니다.

## 선택형 CLI 베이스라인

각 팀원의 `exp_001_baseline/`은 선택적으로 사용할 수 있는 실행 예제입니다.

```bash
python -m experiments.member_a.notebooks.run_eda
python -m experiments.member_a.exp_001_baseline.training.run
python -m experiments.member_a.exp_001_baseline.inference
```

기본 사용 방식은 Jupyter Notebook입니다. CLI 베이스라인을 복사해서 사용해도 되고,
노트북이나 별도 스크립트, PyTorch 등 원하는 방식으로 구성해도 됩니다.

## 공유 결과

성능을 공유할 때는 최소한 실험 ID, owner, 모델, seed/fold, 검증 방식과 지표를
기록합니다.

확률 앙상블에 참여할 때는 `test_probabilities.parquet`, OOF stacking에 참여할 때는
`oof_probabilities.parquet`을 루트 README의 컬럼 규격에 맞춰 준비합니다.
