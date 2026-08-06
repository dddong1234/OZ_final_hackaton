# H0 versus OVR LR 오류 다양성 감사

H1 MoE가 기각된 뒤 진행하는 감사다. 최종 H0 Selective-EB 확률과, 동일 structured+Empirical-Bayes 입력을 사용한 OVR Logistic Regression의 오류가 실제로 보완적인지 확인한다.

- Outer CV: seed42 Stratified 5-fold
- H0, event vocabulary, EB score, 표준화는 각 fold-train으로만 fit
- OVR은 같은 fold-train의 H0+EB feature를 사용하며 `C=0.07`, `max_iter=2000`, `class_weight=balanced`, 순차 학습(`n_jobs=1`)으로 메모리를 제한한다.
- test는 읽지 않으며, blend weight와 제출 파일은 만들지 않는다.

저장하는 핵심 지표는 H0 오답을 OVR이 회복한 수, H0 정답을 OVR이 손상한 수, hard prediction 불일치율, diagnostic oracle Macro F1이다. 회복 신호가 충분할 때만 다음 단계에서 미리 고정한 제한 blend 하나를 검증한다.
