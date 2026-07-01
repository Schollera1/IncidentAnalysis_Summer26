# Grouping File. Clusters a slice of tickets (by default the broad ones that need an L2 split) to see if hidden sub-groups exist. Uses BERTopic (same BERT text model as the model classifier). It also profiles each group by merchant and month, so you do not mistake a group that is really just one merchant or one template for a real new root cause.
import pandas as pd
import numpy as np
from SetUp import ticket_merchant, ticket_month_year

bert_model_name = "BAAI/bge-base-en-v1.5"
filter_col = "review_bucket"
filter_values = ["P1_needs_l2_human_split"]
text_col = "text_masked_merchant"
tickets_df = pd.read_excel("prepped.xlsx")
slice_df = tickets_df[tickets_df[filter_col].isin(filter_values)].copy()
slice_df = slice_df[slice_df[text_col].str.len() > 0]
texts = slice_df[text_col].tolist()
print(f"Grouping {len(texts)} tickets where {filter_col} is one of {filter_values}")


# BERTopic will group the tickets by meaning and names each group by its words
def cluster_bertopic(texts):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    bert_model = SentenceTransformer(bert_model_name)
    topic_vectorizer = CountVectorizer(
        stop_words="english", min_df=2
    )  # keeps the group names readable
    dimension_reducer = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=0
    )
    topic_model = BERTopic(
        embedding_model=bert_model,
        vectorizer_model=topic_vectorizer,
        umap_model=dimension_reducer,
        min_topic_size=8,
        verbose=False,
    )
    labels, _ = topic_model.fit_transform(texts)
    names = {
        row["Topic"]: row["Name"] for _, row in topic_model.get_topic_info().iterrows()
    }
    return np.array(labels), names


# write a summary of each group plus who and when it came from
def profile_and_save(slice_df, labels, names):
    slice_df = slice_df.copy()
    slice_df["cluster"] = labels
    slice_df["month"] = (
        pd.to_datetime(slice_df[ticket_month_year], errors="coerce")
        .dt.to_period("M")
        .astype(str)
    )
    summary = []
    for cluster_id, group in slice_df.groupby("cluster"):
        merchant_counts = group[ticket_merchant].value_counts()
        top_merchant = merchant_counts.index[0] if len(merchant_counts) else "n/a"
        top_merchant_share = (
            round(100 * merchant_counts.iloc[0] / len(group), 0)
            if len(merchant_counts)
            else 0
        )
        summary.append(
            {
                "cluster": cluster_id,
                "count": len(group),
                "top_terms": names.get(cluster_id, ""),
                "top_merchant": top_merchant,
                "top_merchant_share_%": top_merchant_share,  # a high share means one merchant, not a real root cause
                "distinct_merchants": group[ticket_merchant].nunique(),
                "distinct_months": group["month"].nunique(),
                "top_old_label": group["old_label"].value_counts().index[0]
                if "old_label" in group
                else "",
            }
        )
    summary_df = pd.DataFrame(summary).sort_values("count", ascending=False)
    print("\n=== groups ===")
    print(summary_df.to_string(index=False))
    with pd.ExcelWriter("grouping_results.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="group_summary", index=False)
        slice_df.to_excel(writer, sheet_name="ticket_groups", index=False)
    print("\nSaved grouping_results.xlsx")
    print(
        "Note: a group with a high top_merchant_share is really just one merchant or template, not a new root cause."
    )


if __name__ == "__main__":
    labels, names = cluster_bertopic(texts)
    profile_and_save(slice_df, labels, names)
