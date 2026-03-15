import json
import asyncio
import os
import httpx
import warnings
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# MLE FIX: Suppress noisy tracebacks to keep the terminal output clean and professional
# Placed correctly after all imports to satisfy Ruff E402 strict linting rules.
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ragas").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8080/query")


async def run_pipeline(client: httpx.AsyncClient, query: str) -> dict:
    payload = {
        "remediation": {
            "user_query": query,
            "equipment_id": "system",
            "failure_mode": "general_evaluation",
        }
    }

    try:
        response = await client.post(ORCHESTRATOR_URL, json=payload, timeout=300.0)

        if response.status_code == 400:
            error_detail = response.json().get("detail", "Guardrail Blocked")
            print(f"🛡️ Guardrail Intercepted '{query}'")
            return {
                "answer": f"System refused to answer. Reason: {error_detail}",
                "contexts": ["No context retrieved due to guardrail block."],
            }
        elif response.status_code == 422:
            print(f"❌ API Error 422 for '{query}'")
            return {"answer": "Error", "contexts": ["Error"]}
        elif response.status_code != 200:
            print(f"❌ API Error {response.status_code} for '{query}'")
            return {"answer": "Error", "contexts": ["Error"]}

        data = response.json()

        contexts = []
        result_data = data.get("result", {})
        if isinstance(result_data, dict) and "sources" in result_data:
            contexts = [str(src) for src in result_data["sources"]]
        elif isinstance(result_data, dict) and "retrieved_contexts" in result_data:
            contexts = [str(ctx) for ctx in result_data["retrieved_contexts"]]

        return {
            "answer": data.get("answer", "No answer generated"),
            "contexts": contexts if contexts else ["No context retrieved"],
        }
    except httpx.ReadTimeout:
        print(f"❌ Timeout Error for '{query}'")
        return {"answer": "Error", "contexts": ["Error"]}
    except Exception:
        print(f"❌ Request Failed for '{query}'")
        return {"answer": "Error", "contexts": ["Error"]}


async def main():
    with open("data/golden_test_set.json", "r") as f:
        test_cases = json.load(f)

    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    print(f"Running pipeline for {len(test_cases)} test cases against {ORCHESTRATOR_URL}...")

    async with httpx.AsyncClient() as client:
        health_url = ORCHESTRATOR_URL.replace("/query", "/health")
        is_healthy = False

        for attempt in range(1, 6):
            try:
                res = await client.get(health_url, timeout=30.0)
                if res.status_code == 200:
                    print("✅ Orchestrator Health Check Passed.")
                    is_healthy = True
                    break
            except httpx.RequestError:
                print("⚠️ Health check dropped. Retrying in 10 seconds...")
                await asyncio.sleep(10)

        if not is_healthy:
            print("CRITICAL: Cannot establish stable connection to LLM Orchestrator.")
            return

        for case in test_cases:
            result = await run_pipeline(client, case["query"])

            dataset_dict["question"].append(case["query"])
            dataset_dict["user_input"].append(case["query"])

            dataset_dict["answer"].append(result["answer"])
            dataset_dict["response"].append(result["answer"])

            dataset_dict["contexts"].append(result["contexts"])
            dataset_dict["retrieved_contexts"].append(result["contexts"])

            dataset_dict["ground_truth"].append(case["ground_truth"])
            dataset_dict["reference"].append(case["ground_truth"])

    dataset = Dataset.from_dict(dataset_dict)

    print("Initializing Ragas Evaluation Models (8GB RAM Optimized)...")

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    # MLE FIX: Removed format="json" to prevent Schema Regurgitation
    judge_llm = ChatOllama(
        model="llama3.2:3b", temperature=0.0, base_url=ollama_url, client_kwargs={"timeout": 600.0}
    )

    judge_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    print("Running Ragas metrics with LLM self-correction. This will take a few minutes...\n")

    # MLE FIX: Added max_retries=10 so Ragas automatically asks the 3B model
    # to fix its own syntax errors behind the scenes without crashing.
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=1, timeout=1200, max_retries=10),
    )

    os.makedirs("data/evaluation_results", exist_ok=True)
    df = evaluation_result.to_pandas()
    df.to_csv("data/evaluation_results/latest_run.csv", index=False)

    summary_dict = {}
    if hasattr(evaluation_result, "items"):
        summary_dict = {metric: score for metric, score in evaluation_result.items()}
    elif hasattr(evaluation_result, "scores"):
        summary_dict = evaluation_result.scores

    with open("data/evaluation_results/summary.json", "w") as f:
        json.dump(summary_dict, f, indent=4)

    print("\nEvaluation Complete. Aggregate Scores:")
    print(json.dumps(summary_dict, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
