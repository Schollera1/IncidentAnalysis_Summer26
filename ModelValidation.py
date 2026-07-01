# Model Validation File, do this fourth, gives a per-category report and confusion matrix, plus top 3 accuracy, a confidence check, and a time-based holdout (train on older tickets, test on the newest ones).
import pandas as pd
import numpy as np
from collections import Counter
from SetUp import ticket_month_year

bert_model_name = "BAAI/bge-base-en-v1.5"
tickets_df = pd.read_excel("prepped.xlsx")
labeled_df = tickets_df[
    (tickets_df["prepped_status"] == "labeled") & (tickets_df["text"].str.len() > 0)
].copy()


# turn text into BERT sentence embeddings
def embed_texts(texts):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(bert_model_name).encode(
        texts, show_progress_bar=True, normalize_embeddings=True
    )


# set aside classes with too few tickets to test fairly
def drop_rare_classes(text, labels, min_count=3):
    counts = Counter(labels)
    keep = [i for i, label in enumerate(labels) if counts[label] >= min_count]
    dropped = sorted({label for label in labels if counts[label] < min_count})
    if dropped:
        print(f"Set aside rare classes (fewer than {min_count} tickets): {dropped}")
    return [text[i] for i in keep], [labels[i] for i in keep]


# train on one set of tickets and predict another (used by the time-based holdout)
def fit_predict(train_text, train_labels, test_text):
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(np.array(embed_texts(train_text)), train_labels)
    return classifier.predict(np.array(embed_texts(test_text)))


# train on older months, test on the newest ones, to see if the model still works over time
def temporal_holdout(holdout_months=6):
    from sklearn.metrics import f1_score

    frame = labeled_df.copy()
    frame["created"] = pd.to_datetime(frame[ticket_month_year], errors="coerce")
    cutoff = frame["created"].max() - pd.DateOffset(months=holdout_months)
    train = frame[frame["created"] <= cutoff]
    test = frame[frame["created"] > cutoff]
    if len(test) < 10 or train["prepped_new_label"].nunique() < 2:
        print("\ntime-based holdout: not enough data to test")
        return
    predicted = fit_predict(
        train["text"].tolist(),
        train["prepped_new_label"].tolist(),
        test["text"].tolist(),
    )
    macro = f1_score(test["prepped_new_label"], predicted, average="macro")
    print(
        f"\ntime-based holdout (last {holdout_months} months as the test): {len(train)} train / {len(test)} test, macro-F1 {macro:.3f}"
    )


def validate(n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        accuracy_score,
    )

    text = labeled_df["text"].tolist()
    labels = labeled_df["prepped_new_label"].tolist()
    text, labels = drop_rare_classes(text, labels, min_count=n_splits)
    labels = np.array(labels)
    features = np.array(
        embed_texts(text)
    )  # embeddings are frozen, so no leakage across folds
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    probabilities = cross_val_predict(
        classifier, features, labels, cv=kfold, method="predict_proba"
    )
    class_labels = np.unique(labels)
    predicted = class_labels[probabilities.argmax(axis=1)]
    print("=== BERT model, cross-validated ===")
    print(
        f"accuracy {accuracy_score(labels, predicted):.3f}   "
        f"macro-F1 {f1_score(labels, predicted, average='macro'):.3f}   "
        f"weighted-F1 {f1_score(labels, predicted, average='weighted'):.3f}"
    )
    top3 = np.argsort(probabilities, axis=1)[:, -3:]
    top3_hit = np.mean([labels[i] in class_labels[top3[i]] for i in range(len(labels))])
    print(f"top 3 accuracy {top3_hit:.3f}")
    print("\nper-category report:")
    print(classification_report(labels, predicted, zero_division=0))

    confusion = confusion_matrix(labels, predicted, labels=list(class_labels))
    pd.DataFrame(confusion, index=class_labels, columns=class_labels).to_excel(
        "confusion_matrix.xlsx"
    )
    # confidence check
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    calibration = pd.DataFrame(
        {
            "correct": correct,
            "confidence": confidence,
            "bin": pd.cut(confidence, [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        }
    )
    calibration = calibration.groupby("bin", observed=False).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
    )
    print("\nconfidence check:")
    print(calibration.round(3).to_string())
    calibration.round(3).to_csv("calibration.csv")

    temporal_holdout()
    print("\nSaved confusion_matrix.xlsx and calibration.csv")


if __name__ == "__main__":
    validate()
