"""Disease/target node tables and the 13 canonical therapeutic areas, used to build the candidate sets
(scripts/build_ratio_eval.py via src/evaluation/targetbench.py)."""
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Canonical therapeutic areas
# ---------------------------------------------------------------------------
CANONICAL_TA_IDS = {
    "MONDO_0045024",  # cancer or benign tumor
    "EFO_0000319",    # cardiovascular disease
    "EFO_0001379",    # endocrine system disease
    "EFO_0010282",    # gastrointestinal disease
    "OTAR_0000018",   # genetic, familial or congenital disease
    "EFO_0005803",    # hematologic disease
    "EFO_0000540",    # immune system disease
    "EFO_0010285",    # integumentary system disease
    "OTAR_0000006",   # musculoskeletal or connective tissue disease
    "EFO_0000618",    # nervous system disease
    "OTAR_0000017",   # reproductive system or breast disease
    "OTAR_0000010",   # respiratory or thoracic disease
    "EFO_0009690",    # urinary system disease
}



def load_nodes(nodes_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the rich evidenceDated node tables. `nodes_dir` is the parent of
    `diseases/` and `targets/`. pyarrow auto-globs the part files.

    Adds two derived columns used by the scoring prompt:
      - targets.protein_class       — L1 label from `targetClass`, else "Unknown"
      - diseases.therapeutic_area   — human-readable TA name resolved against
                                      the disease table itself; prefers the
                                      first canonical TA from CANONICAL_TA_IDS,
                                      else first listed, else "Unknown"
    """
    diseases_path = os.path.join(nodes_dir, "diseases")
    targets_path = os.path.join(nodes_dir, "targets")

    diseases = pd.read_parquet(
        diseases_path,
        columns=["id", "name", "description", "therapeuticAreas"],
    )
    targets = pd.read_parquet(
        targets_path,
        columns=["id", "approvedSymbol", "approvedName",
                 "functionDescriptions", "targetClass"],
    )

    def _join_descs(x):
        if x is None:
            return ""
        try:
            return " ".join(s for s in x if s)
        except TypeError:
            return ""

    targets["target_description"] = targets["functionDescriptions"].apply(_join_descs)

    def _l1_protein_class(x) -> str:
        if x is None:
            return "Unknown"
        try:
            for entry in x:
                if entry is not None and entry.get("level") == "l1":
                    label = entry.get("label")
                    if label:
                        return str(label)
            # No L1 hit; fall back to the first labelled entry of any level.
            for entry in x:
                if entry is not None and entry.get("label"):
                    return str(entry["label"])
        except (TypeError, AttributeError):
            pass
        return "Unknown"

    targets["protein_class"] = targets["targetClass"].apply(_l1_protein_class)
    targets = targets.drop(columns=["functionDescriptions", "targetClass"])

    # Disease therapeutic area → human-readable string listing ALL TAs the
    # disease belongs to (joined with " | "). A disease often belongs to
    # multiple TAs (e.g. HIV-1 = "infectious disease | reproductive system");
    # the prompt should show every one. Negative sampling in `sample_negatives`
    # already pools candidate targets across all of a disease's TAs, so this
    # is purely the display path.
    id_to_name = dict(zip(diseases["id"], diseases["name"]))

    def _ta_names(tas) -> str:
        if tas is None:
            return "Unknown"
        try:
            ta_list = list(tas)
        except TypeError:
            return "Unknown"
        names = [str(id_to_name[ta]) for ta in ta_list if id_to_name.get(ta)]
        return " | ".join(names) if names else "Unknown"

    diseases["therapeutic_area"] = diseases["therapeuticAreas"].apply(_ta_names)

    print(f"Targets: {len(targets):,} | Diseases: {len(diseases):,}")
    n_pc = (targets["protein_class"] != "Unknown").sum()
    n_ta = (diseases["therapeutic_area"] != "Unknown").sum()
    print(f"  targets with protein_class: {n_pc:,} ({100*n_pc/len(targets):.1f}%)")
    print(f"  diseases with therapeutic_area: {n_ta:,} ({100*n_ta/len(diseases):.1f}%)")
    return targets, diseases


# ---------------------------------------------------------------------------
# Helpers


def attach_names(df: pd.DataFrame, targets: pd.DataFrame, diseases: pd.DataFrame) -> pd.DataFrame:
    t = (
        targets.rename(columns={"id": "targetId", "approvedSymbol": "gene_name"})
        [["targetId", "gene_name", "protein_class", "target_description"]]
    )
    d = (
        diseases.rename(columns={
            "id": "diseaseId",
            "name": "disease_name",
            "description": "disease_description",
        })
        [["diseaseId", "disease_name", "therapeutic_area", "disease_description"]]
    )
    return df.merge(t, on="targetId", how="left").merge(d, on="diseaseId", how="left")
