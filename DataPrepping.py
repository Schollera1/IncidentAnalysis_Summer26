# Data Prepping File, do this second. Will review/clean data and turn old labels into new L1/L2 labels
import pandas as pd
import numpy as np
from SetUp import (
    excel_file_name,
    sheet_name,
    ticket_taxonomy_final,
    ticket_month_year,
    ticket_priority,
    restrict_to_priority,
    l2_to_l1,
    label_renames,
    legacy_parent_splits,
    exclude_labels,
    abstain_labels,
    min_support_for_l2,
    human_confirmed_start,
    human_confirmed_end,
    combine_ticket_text,
    combine_ticket_text_masked,
)


# Remove stray whitespaces/typos from an old label
def normalize_label(label):
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    cleaned = str(label).strip()
    typo_fixes = {
        "code / Config Deployment Regression": "Code / Config Deployment Regression",
    }
    return typo_fixes.get(cleaned, cleaned)


# Determines label statuses for the model (old to new label, already new L2 labels, broad labels that
# need splitting and thus review, or those which must simply be human-reviewed). Returns (status, new_l1, new_l2)
def resolve_label(old_label):
    if old_label is None:
        return ("unknown_label", None, None)
    if old_label in exclude_labels:  # not a root cause, drop it
        return ("exclude", None, None)
    if (
        old_label in abstain_labels
    ):  # "Other", let the model suggest but do not train on it
        return ("needs_classifier", "Review/Not a Root Cause", None)
    if old_label in label_renames:  # old label that was just renamed
        new_l2 = label_renames[old_label]
        return ("labeled", l2_to_l1[new_l2], new_l2)
    if old_label in l2_to_l1:  # already a clean new L2 label
        return ("labeled", l2_to_l1[old_label], old_label)
    if (
        old_label in legacy_parent_splits
    ):  # broad old label, good at L1 but flag it for an L2 split
        return ("labeled_l1", old_label, None)
    return (
        "unknown_label",
        None,
        None,
    )  # something we did not expect, let the model suggest


# Add the new label columns, the review flags, the model text, and the training target
def transform(tickets_df):
    tickets_df = tickets_df.copy()
    tickets_df["old_label"] = tickets_df[ticket_taxonomy_final].map(normalize_label)
    resolved = tickets_df["old_label"].apply(resolve_label)
    tickets_df["resolve_status"] = [r[0] for r in resolved]
    tickets_df["new_l1"] = [r[1] for r in resolved]
    tickets_df["new_l2"] = [r[2] for r in resolved]
    # tickets created Oct 2025 to May 2026 were all human-confirmed
    created = pd.to_datetime(tickets_df[ticket_month_year], errors="coerce")
    in_window = (created >= human_confirmed_start) & (created <= human_confirmed_end)
    tickets_df["label_status"] = np.where(in_window, "human_confirmed", "unverified")
    tickets_df["text"] = tickets_df.apply(combine_ticket_text, axis=1)
    tickets_df["text_masked_merchant"] = tickets_df.apply(
        combine_ticket_text_masked, axis=1
    )
    # pick how specific the training label can be: keep an L2 only if it has enough tickets,
    # otherwise fall back to its L1 (broad old labels already sit at L1)
    is_trainable = tickets_df["resolve_status"].isin(["labeled", "labeled_l1"])
    l2_support = tickets_df.loc[
        tickets_df["new_l2"].notna() & is_trainable, "new_l2"
    ].value_counts()

    def training_label(row):
        if row["resolve_status"] == "labeled_l1":
            return row["new_l1"]  # Use L1
        if row["resolve_status"] == "labeled":
            new_l2 = row["new_l2"]
            if l2_support.get(new_l2, 0) >= min_support_for_l2:
                return new_l2  # enough tickets, keep the L2
            return row["new_l1"]  # too few tickets, fall back to the L1
        return None

    tickets_df["train_label"] = tickets_df.apply(training_label, axis=1)

    # which pile each ticket goes in for a human
    def review_bucket(row):
        if row["resolve_status"] == "exclude":
            return "exclude"
        if len(str(row["text"]).strip()) == 0:
            return "P0_missing_or_short_text"
        if row["resolve_status"] == "labeled_l1":
            return "P1_needs_l2_human_split"
        if row["resolve_status"] == "needs_classifier":
            return "P2_other_review"
        if row["resolve_status"] == "unknown_label":
            return "P2_unknown_label_review"
        if row["resolve_status"] == "labeled":
            return "train_candidate"
        return "review"

    tickets_df["review_bucket"] = tickets_df.apply(review_bucket, axis=1)

    # what model reads for training and preiction
    def prepped_status(row):
        if row["resolve_status"] == "exclude":
            return "exclude"
        if row["resolve_status"] in ("labeled", "labeled_l1"):
            return "labeled"
        return "needs_classifier"

    tickets_df["prepped_status"] = tickets_df.apply(prepped_status, axis=1)
    tickets_df["prepped_new_label"] = tickets_df["train_label"]
    return tickets_df


# restricts priority, cleans, and saves data for mode. Also prints some summary stats
def main():
    tickets_df = pd.read_excel(excel_file_name, sheet_name=sheet_name)
    if restrict_to_priority:
        before = len(tickets_df)
        tickets_df = tickets_df[
            tickets_df[ticket_priority] == restrict_to_priority
        ].copy()
        print(
            f"Kept only {restrict_to_priority}: {len(tickets_df)} of {before} tickets\n"
        )
    tickets_df = transform(tickets_df)
    print("label statuses:")
    print(tickets_df["resolve_status"].value_counts().to_string())
    print("\nreview piles:")
    print(tickets_df["review_bucket"].value_counts().to_string())
    print("\ntraining classes and how many tickets each has:")
    print(
        tickets_df.loc[tickets_df["prepped_status"] == "labeled", "train_label"]
        .value_counts()
        .to_string()
    )
    print(
        "\nhuman-confirmed tickets:",
        int((tickets_df["label_status"] == "human_confirmed").sum()),
    )
    tickets_df.to_excel("prepped.xlsx", index=False)
    print("\nSaved prepped.xlsx")


if __name__ == "__main__":
    main()
