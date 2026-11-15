"""Compare TF-IDF and BM25 over a fixed grid of top-k CAPEC candidates.

For each NIS2 measure, rank 559 CAPEC descriptions and retain at most k
positive-score candidates. Evaluate using WISARD's Case A and Case B.
"""

import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Fixed in advance; no k is selected using the expert reference.
K_VALUES = [5, 10, 20, 50, 100, 200, 300, 400, 559]
K1, B = 1.5, 0.75  # Conventional BM25 parameters.
OUTPUT = Path("lexical_topk_results")

# WISARD: SS (extended with Mistral) - RMax3 (2 models).
# Values taken from the corrected WISARD evaluation (P, R).
WISARD = {
    (7, "A"): (0.20877277085330775, 0.788235294117647),
    (7, "B"): (0.20076726342711, 0.5683257918552036),
    (21, "A"): (0.2019840176357123, 0.8348519362186788),
    (21, "B"): (0.20103822557810289, 0.48519362186788156),
}


def bm25_scores(query, documents, lengths, idf, average_length):
    """Return one BM25 score per CAPEC description."""
    scores = []

    for document, length in zip(documents, lengths):
        normalization = K1 * (1 - B + B * length / average_length)
        score = 0.0

        for term in query:
            frequency = document.get(term, 0)

            if frequency:
                score += (
                    idf[term]
                    * frequency
                    * (K1 + 1)
                    / (frequency + normalization)
                )

        scores.append(score)

    return scores


def expand_children(pairs, children):
    """Add only the direct CAPEC children of each predicted attack pattern."""
    result = set(pairs)

    for measure, attack in pairs:
        result.update(
            (measure, child)
            for child in children.get(attack, set())
        )

    return result


def evaluate(predictions, reference):
    tp = len(predictions & reference)
    fp = len(predictions - reference)
    fn = len(reference - predictions)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        Precision=precision,
        Recall=recall,
        F1=f1,
    )


def main():
    # ---------------------------------------------------------
    # 1. Load the input datasets
    # ---------------------------------------------------------

    measures = pd.read_csv("measures.csv")

    capec = (
        pd.read_csv("Capec-2_filtered.csv")
        .sort_values("ID")
    )

    expert = pd.read_csv(
        "expert_reference_expanded_all_articles.csv"
    )

    assert len(capec) == 559
    assert capec["ID"].nunique() == 559

    assert measures.groupby("article").size().to_dict() == {
        7: 10,
        21: 10,
    }

    assert expert.groupby("Article").size().to_dict() == {
        7: 1105,
        21: 878,
    }

    OUTPUT.mkdir(exist_ok=True)

    capec_ids = capec["ID"].astype(int).tolist()

    descriptions = (
        capec["Description"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # ---------------------------------------------------------
    # 2. Prepare TF-IDF and BM25
    # ---------------------------------------------------------

    # Both methods use the same lowercase,
    # English-stopword-filtered tokens.

    tfidf = TfidfVectorizer(stop_words="english")

    document_vectors = tfidf.fit_transform(descriptions)

    tokenize = tfidf.build_analyzer()

    documents = [
        Counter(tokenize(text))
        for text in descriptions
    ]

    lengths = [
        sum(document.values())
        for document in documents
    ]

    average_length = sum(lengths) / len(lengths)

    document_frequency = Counter()

    for document in documents:
        document_frequency.update(document.keys())

    n_documents = len(documents)

    idf = {
        term: math.log(
            1 + (n_documents - count + 0.5) / (count + 0.5)
        )
        for term, count in document_frequency.items()
    }

    # ---------------------------------------------------------
    # 3. Prepare the CAPEC ChildOf relationships
    # ---------------------------------------------------------

    children = {}

    for _, row in capec.iterrows():
        for parent in re.findall(
            r"ChildOf:\s*CAPEC ID:\s*(\d+)",
            str(row["Related Attack Patterns"]),
            flags=re.I,
        ):
            children.setdefault(
                int(parent), set()
            ).add(int(row["ID"]))

    # ---------------------------------------------------------
    # 4. Rank CAPEC patterns for every NIS2 measure
    # ---------------------------------------------------------

    rankings = []
    results = []

    for article in [7, 21]:

        reference_rows = expert[
            expert["Article"] == article
        ]

        reference = set(
            zip(
                reference_rows["Measure"].astype(int),
                reference_rows["Attack Pattern"].astype(int),
            )
        )

        candidates = {
            "TF-IDF": {},
            "BM25": {},
        }

        for measure in measures[
            measures["article"] == article
        ].itertuples(index=False):

            measure_id = int(measure.measure)
            text = str(measure.measure_text)

            # TF-IDF cosine similarity
            query_vector = tfidf.transform([text])

            tfidf_scores = (
                query_vector @ document_vectors.T
            ).toarray().ravel()

            # BM25 scores
            bm25 = bm25_scores(
                tokenize(text),
                documents,
                lengths,
                idf,
                average_length,
            )

            for method, scores in [
                ("TF-IDF", tfidf_scores),
                ("BM25", bm25),
            ]:

                # Rank by descending score.
                # CAPEC ID is used only to resolve equal scores.
                ranked = sorted(
                    zip(capec_ids, scores),
                    key=lambda item: (-item[1], item[0]),
                )

                # IMPORTANT:
                # Keep only strictly positive scores.
                # Never fill top-k with zero-score candidates.
                positive = [
                    (attack, float(score))
                    for attack, score in ranked
                    if score > 0
                ]

                candidates[method][measure_id] = positive

                rankings.extend(
                    (
                        article,
                        method,
                        measure_id,
                        attack,
                        rank,
                        score,
                    )
                    for rank, (attack, score) in enumerate(
                        positive, start=1
                    )
                )

        # -----------------------------------------------------
        # 5. Evaluate different top-k values
        # -----------------------------------------------------

        for method in ["TF-IDF", "BM25"]:

            for k in K_VALUES:

                # Select at most k positive-score CAPEC patterns
                # for each NIS2 measure.
                predictions = {
                    (measure, attack)
                    for measure, ranked
                    in candidates[method].items()
                    for attack, score in ranked[:k]
                }

                # Case A: expand predictions and reference.
                # Case B: expand reference only.
                #
                # The expert reference is already expanded
                # in the input CSV.

                for case, tested in [
                    (
                        "A",
                        expand_children(predictions, children),
                    ),
                    (
                        "B",
                        predictions,
                    ),
                ]:

                    results.append(
                        dict(
                            Article=article,
                            Method=method,
                            K=k,
                            Case=case,
                            N_Corr=len(predictions),
                            EvaluatedPredictions=len(tested),
                            ReferenceCount=len(reference),
                            **evaluate(tested, reference),
                        )
                    )

    # ---------------------------------------------------------
    # 6. Save the evaluation results
    # ---------------------------------------------------------

    metrics = pd.DataFrame(results)

    metrics.to_csv(
        OUTPUT / "metrics.csv",
        index=False,
    )

    pd.DataFrame(
        rankings,
        columns=[
            "Article",
            "Method",
            "Measure",
            "Attack Pattern",
            "Rank",
            "Score",
        ],
    ).to_csv(
        OUTPUT / "positive_rankings.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 7. Generate precision-recall plots
    # ---------------------------------------------------------

    for article in [7, 21]:

        for case in ["A", "B"]:

            fig, ax = plt.subplots(figsize=(6.3, 4.3))

            for method in ["TF-IDF", "BM25"]:

                rows = metrics[
                    (metrics.Article == article)
                    & (metrics.Case == case)
                    & (metrics.Method == method)
                ]

                ax.plot(
                    rows.Recall,
                    rows.Precision,
                    "o-",
                    label=method,
                )

            # WISARD reference point
            p_w, r_w = WISARD[(article, case)]

            ax.scatter(
                [r_w],
                [p_w],
                marker="*",
                s=160,
                label="WISARD",
            )

            ax.set(
                title=f"Article {article} – Case {case}",
                xlabel="Recall",
                ylabel="Precision",
                xlim=(0, 0.95),
                ylim=(0, 0.35),
            )

            ax.grid(alpha=0.25)
            ax.legend()

            fig.tight_layout()

            fig.savefig(
                OUTPUT
                / f"precision_recall_article{article}_case{case}.png",
                dpi=180,
            )

            fig.savefig(
                OUTPUT
                / f"precision_recall_article{article}_case{case}.pdf"
            )

            plt.close(fig)

    # ---------------------------------------------------------
    # 8. Print results
    # ---------------------------------------------------------

    print(metrics.round(4).to_string(index=False))

    print(
        f"\nSaved results to: {OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()
