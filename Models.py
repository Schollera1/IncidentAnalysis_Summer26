# Models File, do this third. Uses a BERT text model (sentence embeddings) plus logistic regression to learn the categories, then scores the tickets that still need a label. For each one it gives the top 3 guesses, a confidence, a recommended action, and for broad old labels a guess limited to that label's allowed L2 children.
import pandas as pd
import numpy as np
from SetUp import legacy_parent_splits

bert_model_name = "BAAI/bge-base-en-v1.5"
tickets_df = pd.read_excel("prepped.xlsx")
training_df = tickets_df[
    (tickets_df["prepped_status"] == "labeled") & (tickets_df["text"].str.len() > 0)
]
training_text = training_df["text"].tolist()
training_labels = training_df["prepped_new_label"].tolist()
review_buckets = [
    "P1_needs_l2_human_split",
    "P2_other_review",
    "P2_unknown_label_review",
    "P0_missing_or_short_text",
]
review_df = tickets_df[tickets_df["review_bucket"].isin(review_buckets)].copy()
auto_threshold = 0.85  # at or above this, safe to auto-label
med_threshold = 0.60  # at or above this, medium confidence
close_margin = 0.10  # top 2 guesses closer than this, and need human review


# turn text into BERT sentence embeddings
def embed_texts(texts):
    from sentence_transformers import SentenceTransformer

    bert_model = SentenceTransformer(bert_model_name)
    return bert_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


# decide what to tell human for one ticket, returns (restricted_l2, restricted_prob, action, reason)
def recommend_action(row, ranked, class_labels, probabilities):
    if row["review_bucket"] == "P0_missing_or_short_text":
        return (None, None, "review_missing_text", "no usable text")
    top_prob = ranked[0][1]
    second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    if row["review_bucket"] == "P1_needs_l2_human_split":
        # only let the model pick from this old label's allowed children that it actually learneddrf
        allowed = [
            c for c in legacy_parent_splits.get(row["new_l1"], ()) if c in class_labels
        ]
        if not allowed:
            return (
                None,
                None,
                "review_split_children_not_trained",
                "the L2 children have too few tickets to train",
            )
        best = max(allowed, key=lambda c: probabilities[list(class_labels).index(c)])
        best_prob = float(probabilities[list(class_labels).index(best)])
        return (
            best,
            best_prob,
            "review_split_suggestion",
            f"best child of {row['new_l1']}",
        )
    # open guess for "Other" and unknown rows
    if top_prob - second_prob < close_margin:
        return (None, None, "review_close_call", f"top 2 within {close_margin}")
    if top_prob >= auto_threshold:
        return (None, None, "auto_label_candidate", f"pretty sure ({top_prob:.2f})")
    if top_prob >= med_threshold:
        return (None, None, "review_medium_confidence", f"{top_prob:.2f}")
    return (None, None, "review_low_confidence", f"{top_prob:.2f}")


def predict_and_save():
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(np.array(embed_texts(training_text)), training_labels)
    class_labels = classifier.classes_
    probabilities = classifier.predict_proba(
        np.array(embed_texts(review_df["text"].tolist()))
    )

    rows = []
    for i, (_, row) in enumerate(review_df.iterrows()):
        order = np.argsort(probabilities[i])[::-1][:3]
        ranked = [(class_labels[j], float(probabilities[i][j])) for j in order]
        restricted_l2, restricted_prob, action, reason = recommend_action(
            row, ranked, class_labels, probabilities[i]
        )
        rows.append(
            {
                "pred_l2_1": ranked[0][0],
                "pred_prob_1": round(ranked[0][1], 3),
                "pred_l2_2": ranked[1][0] if len(ranked) > 1 else None,
                "pred_prob_2": round(ranked[1][1], 3) if len(ranked) > 1 else None,
                "pred_l2_3": ranked[2][0] if len(ranked) > 2 else None,
                "pred_prob_3": round(ranked[2][1], 3) if len(ranked) > 2 else None,
                "top2_margin": round(
                    ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0), 3
                ),
                "restricted_l2": restricted_l2,
                "restricted_prob": round(restricted_prob, 3)
                if restricted_prob is not None
                else None,
                "recommended_action": action,
                "recommendation_reason": reason,
            }
        )
    scored = pd.concat([review_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    # sort so the rows that need the most attention show up first
    action_order = [
        "review_split_children_not_trained",
        "review_missing_text",
        "review_low_confidence",
        "review_close_call",
        "review_medium_confidence",
        "review_split_suggestion",
        "auto_label_candidate",
    ]
    scored["_order"] = (
        scored["recommended_action"]
        .map({a: i for i, a in enumerate(action_order)})
        .fillna(99)
    )
    scored.sort_values(["_order", "pred_prob_1"]).drop(columns="_order").to_excel(
        "predictions_review_queue.xlsx", index=False
    )

    print("how many tickets got each recommended action:")
    print(scored["recommended_action"].value_counts().to_string())
    print("\nSaved predictions_review_queue.xlsx")


if __name__ == "__main__":
    predict_and_save()
