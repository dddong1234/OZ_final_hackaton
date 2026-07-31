"""Create descriptive EDA artifacts from train.csv only.

The script accepts one explicit train file and never opens test.csv.  Its output
is descriptive evidence for feature hypotheses, not a fitted final model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.moon.exp_006_train_only_eda.variant_features import (
    EVENT_TYPES,
    TRUNCATING_TYPES,
    event_table,
    normalise_values,
    summary_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train = pd.read_csv(args.train_path, low_memory=False)
    target, identifier = "SUBCLASS", "ID"
    if {target, identifier}.difference(train.columns):
        raise ValueError("Expected ID and SUBCLASS columns in train.csv.")
    genes = [column for column in train.columns if column not in {target, identifier}]
    values = normalise_values(train[genes], genes)
    events = event_table(values)
    summary = summary_features(values, events, recurrent_pairs=[])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    class_distribution = train.groupby(target).size().rename("sample_count").reset_index()
    class_distribution["sample_fraction"] = class_distribution.sample_count / len(train)
    class_distribution.sort_values("sample_count", ascending=False).to_csv(output / "class_distribution.csv", index=False)

    event_counts = (
        events.event_type.value_counts()
        .reindex(EVENT_TYPES, fill_value=0)
        .rename_axis("event_type")
        .rename("event_count")
        .reset_index()
    )
    event_counts["event_fraction"] = event_counts.event_count / max(len(events), 1)
    event_counts.to_csv(output / "event_type_distribution.csv", index=False)

    burden_columns = [column for column in summary.columns if column.endswith("_count")]
    summary[burden_columns].describe().T.to_csv(output / "burden_overall.csv")
    by_class = pd.concat([train[[target]], summary[burden_columns]], axis=1).groupby(target).agg(["mean", "median", "min", "max"])
    by_class.to_csv(output / "burden_by_subclass.csv")

    mutated = values.ne("WT") & values.ne("MISSING")
    top_global = (
        mutated.sum()
        .sort_values(ascending=False)
        .rename_axis("gene")
        .rename("mutated_samples")
        .reset_index()
    )
    top_global["mutation_rate"] = top_global.mutated_samples / len(train)
    top_global.head(100).to_csv(output / "top_mutated_genes_overall.csv", index=False)
    long = mutated.copy()
    long[target] = train[target].to_numpy()
    by_class_gene = long.groupby(target).sum().T
    rows: list[dict[str, object]] = []
    for subclass in by_class_gene.columns:
        top = by_class_gene[subclass].sort_values(ascending=False).head(20)
        class_size = int((train[target] == subclass).sum())
        rows.extend(
            {target: subclass, "gene": gene, "mutated_samples": int(count), "mutation_rate": float(count / class_size)}
            for gene, count in top.items()
            if count > 0
        )
    pd.DataFrame(rows).to_csv(output / "top_mutated_genes_by_subclass.csv", index=False)

    metadata = {
        "scope": "train.csv only; test.csv was not opened",
        "train_rows": int(len(train)),
        "subclass_count": int(train[target].nunique()),
        "mutation_gene_columns": int(len(genes)),
        "missing_cells": int(values.eq("MISSING").sum().sum()),
        "non_wt_cells": int(mutated.sum().sum()),
        "event_count": int(len(events)),
        "truncating_definition": sorted(TRUNCATING_TYPES),
    }
    (output / "eda_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved EDA artifacts to {output}")


if __name__ == "__main__":
    main()
