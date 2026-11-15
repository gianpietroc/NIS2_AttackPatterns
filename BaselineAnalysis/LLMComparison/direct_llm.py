
import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

PAPER_MODEL = "gpt-4o-mini-2024-07-18"
MODERN_MODEL = "gpt-5.6-luna"
BATCH_SIZE = 40
CHILDOF = re.compile(r"ChildOf:\s*CAPEC ID:\s*(\d+)", re.IGNORECASE)

SYSTEM_PROMPT = """You are evaluating whether a security measure from the
NIS2 Directive can PREVENT or MITIGATE candidate CAPEC attack patterns.

Treat EACH candidate independently.

Select a CAPEC attack pattern only if, based on the provided texts, implementing
the security measure could reasonably:
1) PREVENT the attack pattern, i.e., help stop the attack from occurring; OR
2) MITIGATE the attack pattern, i.e., help reduce its likelihood, effectiveness,
   consequences, or impact.

Do NOT select a candidate merely because the measure could detect the attack,
support response or recovery after the attack, improve cybersecurity in general,
or because non-compliance might broadly increase exposure. Such a relationship
qualifies only when it also provides a reasonable prevention or mitigation link.

Do not select a candidate merely because both texts concern cybersecurity.
Do not impose a quota. It is valid to select none, one, several, or all
candidates if justified by the prevention-or-mitigation criterion.

Use only the provided security measure and CAPEC descriptions. Do not use
external knowledge, tools, web search, ontology information, CAPEC hierarchy,
or any additional context.

Return only the selected CAPEC IDs through the required structured output.
Do not provide explanations."""

SCHEMA = {
    "type": "json_schema",
    "name": "capec_selection",
    "description": "CAPEC IDs that the NIS2 measure could reasonably prevent or mitigate.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "selected_capec_ids": {"type": "array", "items": {"type": "integer"}}
        },
        "required": ["selected_capec_ids"],
        "additionalProperties": False,
    },
}


def build_prompt(article, point, measure_id, measure_text, batch):
    candidates = []
    for row in batch.itertuples(index=False):
        candidates.append(
            f"CAPEC-{int(row.ID)}\n"
            f"Description: {row.Description}"
        )
    candidate_text = "\n\n---\n\n".join(candidates)
    return f"""NIS2 Article {article}(2), point ({point})
Measure index: {measure_id}

SECURITY MEASURE:
{measure_text}

CANDIDATE CAPEC ATTACK PATTERNS:
{candidate_text}

Evaluate every candidate independently. Return the IDs of all candidates for
which the security measure could reasonably PREVENT or MITIGATE the attack
pattern according to the criterion above."""


def load_data(measures_file, capec_file, reference_file):
    measures = pd.read_csv(measures_file)
    capec = pd.read_csv(capec_file)
    reference = pd.read_csv(reference_file)

    for df, columns, filename in [
        (measures, ("article", "measure", "point", "measure_text"), measures_file),
        (capec, ("ID", "Description", "Related Attack Patterns"), capec_file),
        (reference, ("Article", "Measure", "Attack Pattern"), reference_file),
    ]:
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"{filename}: missing columns: {sorted(missing)}")

    measures["article"] = measures["article"].astype(int)
    measures["measure"] = measures["measure"].astype(int)
    measures = measures.sort_values(["article", "measure"]).reset_index(drop=True)
    capec["ID"] = capec["ID"].astype(int)
    capec["Description"] = capec["Description"].fillna("").astype(str)
    capec = capec.sort_values("ID").reset_index(drop=True)

    if len(measures) != 20 or any(sum(measures["article"] == a) != 10 for a in (7, 21)):
        raise ValueError("Expected 20 measures: 10 for Article 7 and 10 for Article 21")
    if measures.duplicated(["article", "measure"]).any():
        raise ValueError("Duplicate measures")
    if len(capec) != 559 or capec["ID"].duplicated().any():
        raise ValueError("Expected 559 CAPEC patterns with unique IDs")

    children = {}
    for _, row in capec.iterrows():
        related = row["Related Attack Patterns"]
        for parent in CHILDOF.findall("" if pd.isna(related) else str(related)):
            children.setdefault(int(parent), set()).add(int(row["ID"]))

    refs = {
        article: set(zip(
            reference.loc[reference["Article"] == article, "Measure"].astype(int),
            reference.loc[reference["Article"] == article, "Attack Pattern"].astype(int),
        )) for article in (7, 21)
    }
    return measures, capec, children, refs


def ask_llm(client, model_id, prompt, batch_ids):
    request = {
        "model": model_id,
        "instructions": SYSTEM_PROMPT,
        "input": prompt,
        "text": {"format": SCHEMA},
        "max_output_tokens": 2048,
    }
    if model_id.startswith("gpt-4o"):
        request["temperature"] = 0
    elif model_id.startswith("gpt-5.6"):
        request["reasoning"] = {"effort": "none"}

    for attempt in range(5):
        try:
            response = client.responses.create(**request)
            if getattr(response, "status", None) not in (None, "completed"):
                raise ValueError(f"Response status: {response.status}")
            ids = sorted(set(json.loads(response.output_text)["selected_capec_ids"]))
            if not all(type(i) is int and i in batch_ids for i in ids):
                raise ValueError("The model returned IDs not present in the batch")
            return ids
        except Exception as exc:
            if attempt == 4:
                raise
            delay = 2 * (2 ** attempt)
            print(f"Request failed ({attempt + 1}/5): {exc}; retrying in {delay}s")
            time.sleep(delay)


def predict_model(client, label, model_id, measures, capec, out):
    rows = []
    cache_dir = out / "cache" / label
    cache_dir.mkdir(parents=True, exist_ok=True)

    for measure in measures.itertuples(index=False):
        article, mid = int(measure.article), int(measure.measure)
        for batch_no, start in enumerate(range(0, len(capec), BATCH_SIZE), start=1):
            batch = capec.iloc[start:start + BATCH_SIZE]
            valid = set(batch["ID"].astype(int))
            prompt = build_prompt(article, str(measure.point), mid,
                                  str(measure.measure_text), batch)
            cache_file = cache_dir / f"article{article}_measure{mid}_batch{batch_no:02d}.json"
            saved = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

            if (saved.get("model_id") == model_id and
                    saved.get("system_prompt") == SYSTEM_PROMPT and
                    saved.get("user_prompt") == prompt and
                    set(saved.get("selected_capec_ids", [])) <= valid):
                selected = saved["selected_capec_ids"]
            else:
                selected = ask_llm(client, model_id, prompt, valid)
                cache_file.write_text(json.dumps({
                    "model_id": model_id,
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": prompt,
                    "selected_capec_ids": selected,
                }), encoding="utf-8")

            for capec_id in selected:
                rows.append({"ModelLabel": label, "ModelID": model_id,
                             "Article": article, "Measure": mid,
                             "Attack Pattern": int(capec_id)})
            print(f"{label}: Article {article}, measure {mid}, "
                  f"batch {batch_no}: {len(selected)}/{len(batch)}")
    return rows


def expand(pairs, children):
    result = set(pairs)
    for mid, capec_id in pairs:
        result.update((mid, child) for child in children.get(capec_id, set()))
    return result


def metrics(pred, ref):
    tp = len(pred & ref)
    fp = len(pred - ref)
    fn = len(ref - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1


def evaluate(predictions, refs, children):
    results = []
    for label, model_id in (("paper_gpt4o_mini", PAPER_MODEL), ("modern_gpt56", MODERN_MODEL)):
        for article in (7, 21):
            df = predictions[(predictions["ModelLabel"] == label) &
                             (predictions["Article"] == article)]
            original = set(zip(df["Measure"].astype(int), df["Attack Pattern"].astype(int)))
            expanded = expand(original, children)
            a = metrics(expanded, refs[article])
            b = metrics(original, refs[article])
            results.append({
                "ModelLabel": label, "ModelID": model_id, "Article": article,
                "N.Corr.": len(original),
                "EvalPredCount_A": len(expanded),
                "ReferenceCount": len(refs[article]),
                **{f"{col}_A": val for col, val in zip(("TP", "FP", "FN", "P", "R", "F1"), a)},
                **{f"{col}_B": val for col, val in zip(("TP", "FP", "FN", "P", "R", "F1"), b)},
            })
    return pd.DataFrame(results)


def paper_table(metrics_df):
    rows = []
    for _, r in metrics_df.iterrows():
        rows.append({
            "ModelLabel": r["ModelLabel"], "ModelID": r["ModelID"],
            "Article": int(r["Article"]), "N.Corr.": int(r["N.Corr."]),
            "TP_A/B": f'{r["TP_A"]:.0f}/{r["TP_B"]:.0f}',
            "P_A/B": f'{r["P_A"]:.3f}/{r["P_B"]:.3f}',
            "R_A/B": f'{r["R_A"]:.3f}/{r["R_B"]:.3f}',
            "F1_A/B": f'{r["F1_A"]:.3f}/{r["F1_B"]:.3f}',
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Minimal direct LLM baseline")
    parser.add_argument("--measures", type=Path, required=True)
    parser.add_argument("--capec", type=Path, required=True)
    parser.add_argument("--expert-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("direct_llm_results_simple"))
    parser.add_argument("--predictions", type=Path,
                        help="Re-evaluate an existing predictions CSV WITHOUT API calls")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    measures, capec, children, refs = load_data(
        args.measures, args.capec, args.expert_reference)
    columns = ["ModelLabel", "ModelID", "Article", "Measure", "Attack Pattern"]

    if args.predictions:
        pred_df = pd.read_csv(args.predictions)
        if set(columns) - set(pred_df.columns):
            raise ValueError("The predictions file is missing required columns")
        pred_df = pred_df[columns].drop_duplicates()
        expected_models = {"paper_gpt4o_mini": PAPER_MODEL, "modern_gpt56": MODERN_MODEL}
        for label, model_id in expected_models.items():
            rows = pred_df[pred_df["ModelLabel"] == label]
            if rows.empty or set(rows["ModelID"]) != {model_id}:
                raise ValueError(f"Missing predictions or different model ID for {label}")
        if not set(pred_df["ModelLabel"]).issubset(expected_models):
            raise ValueError("Unknown ModelLabel in predictions")
        allowed_pairs = set(zip(measures["article"], measures["measure"]))
        if not set(zip(pred_df["Article"], pred_df["Measure"])) <= allowed_pairs:
            raise ValueError("Predictions contain unknown measures")
        if not set(pred_df["Attack Pattern"].astype(int)) <= set(capec["ID"]):
            raise ValueError("Predictions contain unknown CAPEC patterns")
        print("Using existing predictions: no API calls.")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("Set OPENAI_API_KEY before running the script")
        client = OpenAI()
        rows = []
        for label, model in (("paper_gpt4o_mini", PAPER_MODEL), ("modern_gpt56", MODERN_MODEL)):
            rows.extend(predict_model(client, label, model, measures, capec, args.out))
        pred_df = pd.DataFrame(rows, columns=columns).drop_duplicates()

    pred_df.to_csv(args.out / "direct_llm_predictions.csv", index=False)
    metrics_df = evaluate(pred_df, refs, children)
    metrics_df.to_csv(args.out / "direct_llm_metrics.csv", index=False)
    paper = paper_table(metrics_df)
    paper.to_csv(args.out / "direct_llm_paper_table.csv", index=False)
    print("\n", paper.to_string(index=False))


if __name__ == "__main__":
    main()
