
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from src.retrieve import retrieve


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "evaluation" / "questions.json"
OUTPUT_DIR = BASE_DIR / "evaluation" / "results"

DETAIL_PATH = OUTPUT_DIR / "benchmark_results.json"
CSV_PATH = OUTPUT_DIR / "benchmark_summary.csv"

TOP_K = 4
CANDIDATE_K = 12

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
API_KEY = os.getenv("GEMINI_API_KEY")


# --------------------------------------------------
# Gemini client: initialize only in full mode
# --------------------------------------------------

def create_gemini_client():
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from your .env file."
        )

    from google import genai
    return genai.Client(api_key=API_KEY)


# --------------------------------------------------
# Dataset and source helpers
# --------------------------------------------------

def normalize_source(source):
    return Path(str(source).replace("\\", "/")).name.lower()


def is_relevant(result, relevant_sources):
    source = normalize_source(result.get("source", ""))
    page = result.get("page")

    for gold in relevant_sources:
        if (
            normalize_source(gold["source"]) == source
            and gold["page"] == page
        ):
            return True

    return False


# --------------------------------------------------
# Retrieval metrics
# --------------------------------------------------

def recall_at_k(results, relevant_sources):
    if not relevant_sources:
        return None

    retrieved = {
        (
            normalize_source(result.get("source", "")),
            result.get("page"),
        )
        for result in results
    }

    gold = {
        (
            normalize_source(item["source"]),
            item["page"],
        )
        for item in relevant_sources
    }

    return len(retrieved & gold) / len(gold)


def reciprocal_rank(results, relevant_sources):
    if not relevant_sources:
        return None

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, relevant_sources):
            return 1.0 / rank

    return 0.0


def safe_retrieve(question, use_reranker):
    start = time.perf_counter()

    results = retrieve(
        question,
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        use_reranker=use_reranker,
    )

    elapsed = time.perf_counter() - start

    return results[:TOP_K], elapsed


# --------------------------------------------------
# Gemini generation
# --------------------------------------------------

def format_context(results):
    blocks = []

    for index, result in enumerate(results, start=1):
        blocks.append(
            f"[S{index}]\n"
            f"Source: {result.get('source', 'unknown')}\n"
            f"Page: {result.get('page', 'unknown')}\n"
            f"Content:\n{result.get('text', '')}"
        )

    return "\n\n".join(blocks)


def generate_answer(client, prompt):
    start = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return (
        response.text or "",
        time.perf_counter() - start,
    )


def answer_with_rag(client, question, results):
    prompt = f"""
Answer the question using ONLY the document context.
Do not invent facts or sources.
If evidence is insufficient, say so.
Cite factual claims using exact identifiers such as [S1].
Only use source identifiers present in the supplied context.

DOCUMENT CONTEXT:
{format_context(results)}

QUESTION:
{question}

Give a clear, evidence-grounded answer.
"""
    return generate_answer(client, prompt)


def answer_without_rag(client, question):
    prompt = f"""
Answer using your general knowledge, not local documents.
Do not claim to have read local files.
If uncertain, say so.

QUESTION:
{question}
"""
    return generate_answer(client, prompt)


def citation_metrics(answer, number_of_sources):
    found = set(re.findall(r"\[S(\d+)\]", answer))
    valid = {str(i) for i in range(1, number_of_sources + 1)}

    invalid = sorted(found - valid)

    return {
        "has_citation": bool(found),
        "invalid_citation_ids": invalid,
        "citation_ids_valid": not invalid,
    }


# --------------------------------------------------
# Checkpoint persistence
# --------------------------------------------------

def load_checkpoint():
    if not DETAIL_PATH.exists():
        return {}

    try:
        with open(DETAIL_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {item["id"]: item for item in data}

    except (json.JSONDecodeError, KeyError, TypeError):
        print("Existing checkpoint is invalid; starting fresh.")
        return {}


def save_checkpoint(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ordered = list(results.values())

    with open(DETAIL_PATH, "w", encoding="utf-8") as file:
        json.dump(ordered, file, indent=2, ensure_ascii=False)

    rows = []

    for item in ordered:
        row = {
            "id": item.get("id"),
            "question": item.get("question"),
            "dense_recall_at_4": item.get("dense_recall_at_4"),
            "reranked_recall_at_4": item.get("reranked_recall_at_4"),
            "dense_mrr": item.get("dense_mrr"),
            "reranked_mrr": item.get("reranked_mrr"),
            "dense_retrieval_seconds": item.get("dense_retrieval_seconds"),
            "reranked_retrieval_seconds": item.get("reranked_retrieval_seconds"),
            "dense_answer_status": item.get("dense_answer_status"),
            "reranked_answer_status": item.get("reranked_answer_status"),
            "plain_gemini_status": item.get("plain_gemini_status"),
        }
        rows.append(row)

    if rows:
        with open(
            CSV_PATH, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)


def summarize(results):
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for mode in ("dense", "reranked"):
        recall_key = f"{mode}_recall_at_4"
        mrr_key = f"{mode}_mrr"

        recall_values = [
            item[recall_key]
            for item in results.values()
            if item.get(recall_key) is not None
        ]
        mrr_values = [
            item[mrr_key]
            for item in results.values()
            if item.get(mrr_key) is not None
        ]

        if recall_values:
            print(
                f"{mode} mean Recall@4: "
                f"{sum(recall_values) / len(recall_values):.3f}"
            )

        if mrr_values:
            print(
                f"{mode} mean MRR: "
                f"{sum(mrr_values) / len(mrr_values):.3f}"
            )

    print(f"\nDetailed JSON: {DETAIL_PATH}")
    print(f"Summary CSV:   {CSV_PATH}")


# --------------------------------------------------
# Main evaluator
# --------------------------------------------------

def evaluate(mode, resume):
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Create the benchmark dataset first: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_by_id = load_checkpoint() if resume else {}

    client = None
    if mode == "full":
        client = create_gemini_client()

    for index, item in enumerate(questions, start=1):
        question_id = item["id"]
        question = item["question"]
        gold_sources = item.get("relevant_sources", [])

        existing = results_by_id.get(question_id, {})

        if (
            resume
            and existing.get("retrieval_status") == "completed"
            and (
                mode == "retrieval"
                or existing.get("full_status") == "completed"
            )
        ):
            print(f"\n[{index}/{len(questions)}] {question_id}: already completed; skipping.")
            continue

        record = existing or {
            "id": question_id,
            "question": question,
        }

        record["id"] = question_id
        record["question"] = question

        print(f"\n[{index}/{len(questions)}] {question_id}: {question}")

        try:
            dense_results, dense_time = safe_retrieve(
                question, use_reranker=False
            )
            reranked_results, reranked_time = safe_retrieve(
                question, use_reranker=True
            )

            record.update({
                "retrieval_status": "completed",
                "dense_recall_at_4": recall_at_k(
                    dense_results, gold_sources
                ),
                "reranked_recall_at_4": recall_at_k(
                    reranked_results, gold_sources
                ),
                "dense_mrr": reciprocal_rank(
                    dense_results, gold_sources
                ),
                "reranked_mrr": reciprocal_rank(
                    reranked_results, gold_sources
                ),
                "dense_retrieval_seconds": round(dense_time, 4),
                "reranked_retrieval_seconds": round(reranked_time, 4),
                "dense_sources": [
                    {
                        "source": r.get("source"),
                        "page": r.get("page"),
                        "distance": r.get("distance"),
                    }
                    for r in dense_results
                ],
                "reranked_sources": [
                    {
                        "source": r.get("source"),
                        "page": r.get("page"),
                        "distance": r.get("distance"),
                        "rerank_score": r.get("rerank_score"),
                    }
                    for r in reranked_results
                ],
            })

            print(
                f"  dense: Recall@4={record['dense_recall_at_4']}, "
                f"MRR={record['dense_mrr']}"
            )
            print(
                f"  reranked: Recall@4={record['reranked_recall_at_4']}, "
                f"MRR={record['reranked_mrr']}"
            )

        except Exception as exc:
            record["retrieval_status"] = "failed"
            record["retrieval_error"] = str(exc)
            print(f"  Retrieval failed: {exc}")

        # Persist retrieval metrics immediately.
        results_by_id[question_id] = record
        save_checkpoint(results_by_id)

        if mode != "full":
            continue

        if record.get("retrieval_status") != "completed":
            record["full_status"] = "retrieval_failed"
            results_by_id[question_id] = record
            save_checkpoint(results_by_id)
            continue

        # Generation failures are isolated from retrieval results.
        for method, retrieved in (
            ("dense", dense_results),
            ("reranked", reranked_results),
        ):
            try:
                answer, elapsed = answer_with_rag(
                    client, question, retrieved
                )

                record[f"{method}_answer"] = answer
                record[f"{method}_generation_seconds"] = round(elapsed, 4)
                record[f"{method}_citations"] = citation_metrics(
                    answer, len(retrieved)
                )
                record[f"{method}_answer_status"] = "completed"

            except Exception as exc:
                record[f"{method}_answer_status"] = "failed"
                record[f"{method}_answer_error"] = str(exc)
                print(f"  {method} Gemini call failed: {exc}")

            results_by_id[question_id] = record
            save_checkpoint(results_by_id)

        try:
            answer, elapsed = answer_without_rag(client, question)

            record["plain_gemini_answer"] = answer
            record["plain_gemini_generation_seconds"] = round(elapsed, 4)
            record["plain_gemini_status"] = "completed"

        except Exception as exc:
            record["plain_gemini_status"] = "failed"
            record["plain_gemini_error"] = str(exc)
            print(f"  Plain Gemini call failed: {exc}")

        record["full_status"] = (
            "completed"
            if (
                record.get("dense_answer_status") == "completed"
                and record.get("reranked_answer_status") == "completed"
                and record.get("plain_gemini_status") == "completed"
            )
            else "partial"
        )

        results_by_id[question_id] = record
        save_checkpoint(results_by_id)

    summarize(results_by_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate dense and reranked document RAG."
    )

    parser.add_argument(
        "--mode",
        choices=["retrieval", "full"],
        default="retrieval",
        help="retrieval uses no Gemini calls; full also generates answers.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing evaluation results.",
    )

    args = parser.parse_args()

    evaluate(mode=args.mode, resume=args.resume)
