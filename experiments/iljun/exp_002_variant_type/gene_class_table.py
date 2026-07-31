"""
유전자 × 암종 연결표 — train 데이터에서 직접 만든다
====================================================

    .venv/bin/python3 experiments/iljun/exp_002_variant_type/gene_class_table.py

----------------------------------------------------------------------
왜 만드나
----------------------------------------------------------------------
"어느 유전자가 어느 암종에 몰려 있는가"를 알면 두 가지를 할 수 있다.

1. 우리 데이터가 알려진 생물학과 일치하는지 확인 (발표 근거)
2. 마커가 없는 암종을 찾아 성능 한계를 설명

**외부 유전자 목록을 쓰지 않는다.** 대회 규정상 외부 데이터는 금지이고,
경수님 실험에서도 driver/oncogene 목록 기반 피처는 전부 baseline 이하였다.
여기서는 train 안에서만 센다. test 는 열지 않는다.

----------------------------------------------------------------------
세 가지 지표
----------------------------------------------------------------------
유전자 g, 암종 c 에 대해:

  변이율   P(g|c)  = 그 암종 환자 중 이 유전자에 변이가 있는 비율
                    → "이 암종의 몇 %가 이 유전자를 갖고 있나" (민감도)

  집중도   P(c|g)  = 이 유전자에 변이가 있는 환자 중 그 암종의 비율
                    → "이 유전자를 보면 이 암종이라고 말할 수 있나" (특이도)

  lift     P(g|c) / P(g)
                    → "전체 평균보다 몇 배나 몰려 있나"

마커 판정 기준 (임의로 정한 것이며, 바꾸면 결과가 달라진다):
  · 변이율 >= 10%      해당 암종 환자의 10명 중 1명 이상이 가짐
  · lift   >= 3.0      전체 평균보다 3배 이상 몰림
  · 환자수 >= 3        우연 방지 (소수 클래스는 38명뿐이라 필요)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

MIN_PREV, MIN_LIFT, MIN_N = 0.10, 3.0, 3

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "configs" / "baseline.yaml").exists(): return p
    raise FileNotFoundError("레포 루트를 못 찾음")

ROOT = find_root(Path(__file__).resolve().parent)
train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
gene_cols = [c for c in train.columns if c not in ("ID","SUBCLASS")]
y = train["SUBCLASS"].values
classes = sorted(pd.unique(y))

M = (train[gene_cols].fillna("WT").values != "WT")          # 환자 × 유전자 (True/False)
n_total = len(train)
overall = M.sum(axis=0) / n_total                            # 유전자별 전체 변이율

rows = []
for c in classes:
    idx = (y == c); n_c = idx.sum()
    cnt = M[idx].sum(axis=0)                                 # 이 암종에서 각 유전자 변이 환자 수
    prev = cnt / n_c
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(overall > 0, prev / overall, 0.0)
    conc = np.where(cnt > 0, cnt / np.maximum(M.sum(axis=0), 1), 0.0)   # P(c|g)
    for j, g in enumerate(gene_cols):
        if cnt[j] >= MIN_N and prev[j] >= MIN_PREV and lift[j] >= MIN_LIFT:
            rows.append({"암종": c, "n_class": int(n_c), "유전자": g,
                         "환자수": int(cnt[j]), "변이율": round(float(prev[j]),4),
                         "전체변이율": round(float(overall[j]),4),
                         "lift": round(float(lift[j]),2),
                         "집중도": round(float(conc[j]),4)})

mk = pd.DataFrame(rows).sort_values(["암종","lift"], ascending=[True,False])

print("="*78); print("  유전자 × 암종 연결표 (train 6,201명만 사용)")
print(f"  마커 기준: 변이율>={MIN_PREV:.0%} · lift>={MIN_LIFT} · 환자수>={MIN_N}")
print("="*78)
print(f"\n조건을 만족한 (유전자,암종) 쌍: {len(mk):,}개\n")

print("── lift 상위 20 (가장 특이적인 조합) " + "─"*35)
print(mk.nlargest(20,"lift")[["암종","유전자","환자수","변이율","lift","집중도"]].to_string(index=False))

print("\n── 암종별 마커 개수 " + "─"*52)
cnt_by = mk.groupby("암종").size().reindex(classes, fill_value=0).sort_values()
size_by = pd.Series({c:(y==c).sum() for c in classes})
tab = pd.DataFrame({"환자수": size_by, "마커 수": cnt_by}).sort_values("마커 수")
print(tab.to_string())

zero = tab[tab["마커 수"]==0].index.tolist()
print(f"\n마커 0개 암종 {len(zero)}개: {zero}")

print("\n── 논문 대조 " + "─"*60)
known = [("BRAF","THCA",0.617),("APC","COAD",0.81),("VHL","KIRC",None),("IDH1","LGG",None),
         ("VHL","KIPAN",None),("IDH1","GBMLGG",None),("TP53","BRCA",None)]
for g,c,paper in known:
    if g in gene_cols:
        j = gene_cols.index(g); idx=(y==c)
        p = M[idx,j].sum()/idx.sum()
        pp = f"   논문 {paper:.1%}" if paper else ""
        print(f"  {g:6} ↔ {c:8} 우리 {p:6.1%}{pp}")
    else:
        print(f"  {g:6} ↔ {c:8} — 이 유전자는 우리 4,384개 컬럼에 없음")

print("\n── 신장암 마커 가용성 (논문에서 지목된 유전자) " + "─"*25)
for g in ["VHL","PBRM1","SETD2","KDM5C","BAP1","MTOR","TP53"]:
    print(f"  {g:8} {'있음' if g in gene_cols else '없음 ← 우리 데이터에 컬럼 자체가 없다'}")

art = ROOT/"experiments"/"iljun"/"exp_002_variant_type"/"artifacts"; art.mkdir(parents=True, exist_ok=True)
mk.to_csv(art/"gene_class_markers.csv", index=False, encoding="utf-8-sig")
tab.to_csv(art/"gene_class_marker_counts.csv", encoding="utf-8-sig")
print(f"\n저장: artifacts/gene_class_markers.csv · gene_class_marker_counts.csv")

# ----------------------------------------------------------------------
# 클래스별 F1 과 연결 (class_f1_delta.csv 가 있을 때만)
# ----------------------------------------------------------------------
f1_path = art / "class_f1_delta.csv"
if f1_path.exists():
    f1df = pd.read_csv(f1_path, index_col=0)
    f1col = [c for c in f1df.columns if "최고" in c or "GBVR" in c]
    if f1col:
        f1 = f1df[f1col[0]].rename("F1")
        best = mk.groupby("암종").agg(최고변이율=("변이율","max")).reindex(classes).fillna(0)
        d = pd.DataFrame({"환자수": size_by, "마커 수": cnt_by}).join(best).join(f1).dropna()

        print("\n" + "="*78)
        print("  마커와 모델 성능의 관계")
        print("="*78)
        for col in ["마커 수","최고변이율","환자수"]:
            print(f"  {col:8} vs F1 상관 {d[col].corr(d['F1']):+.3f}")
        print("  → 환자 수는 F1 과 사실상 무관하다. class_weight='balanced' 와 Macro F1 때문에")
        print("    크기 이점이 제거되기 때문이다. 대신 '구분되는 마커가 있는가'가 성능을 가른다.")

        OVERLAP = [c for c in ["KIPAN","KIRC","GBMLGG","LGG"] if c in d.index]
        rest = d.drop(index=OVERLAP)
        print(f"\n  전체 {len(d)}개            최고변이율 vs F1 {d['최고변이율'].corr(d['F1']):+.3f}")
        print(f"  포함관계 4개 제외 ({len(rest)}개)  최고변이율 vs F1 {rest['최고변이율'].corr(d['F1']):+.3f}")
        print(f"\n  포함관계 4개 평균 F1 {d.loc[OVERLAP,'F1'].mean():.4f}")
        print(f"  나머지     평균 F1 {rest['F1'].mean():.4f}"
              f"   차이 {rest['F1'].mean()-d.loc[OVERLAP,'F1'].mean():+.4f}")

        top = mk.sort_values("변이율",ascending=False).drop_duplicates("암종").set_index("암종")["유전자"]
        print("\n  왜 예외인가 — 대표 마커가 겹친다")
        for a,b in [("KIRC","KIPAN"),("LGG","GBMLGG")]:
            if a in top.index and b in top.index:
                same = "★ 같음" if top[a]==top[b] else ""
                print(f"    {a:7} {top[a]:6}  ↔  {b:7} {top[b]:6}  {same}")
        d.to_csv(art/"marker_vs_f1.csv", encoding="utf-8-sig")
        print(f"\n  저장: artifacts/marker_vs_f1.csv")
