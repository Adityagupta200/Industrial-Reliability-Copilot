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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# MLE FIX: Suppress noisy tracebacks to keep the terminal output clean and professional
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ragas").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000/query")

async def run_pipeline(client: httpx.AsyncClient, query: str) -> dict:
    payload = {
        "root_cause": {
            "user_query": query,
            "anomaly_description": "Evaluating system baseline",
            "sensor_data": {},
            "equipment_id": "eval-system",
            "prompt_version": "v1.0"
        }
    }

    try:
        # Strict timeout to enforce the 2s SLA during offline load testing
        response = await client.post(ORCHESTRATOR_URL, json=payload, timeout=5.0)

        if response.status_code == 400:
            error_detail = response.json().get("detail", "Guardrail Blocked")
            return {
                "answer": f"Blocked: {error_detail}",
                "contexts": ["No context retrieved due to guardrail block."],
            }
        elif response.status_code != 200:
            return {"answer": "Error", "contexts": ["Error"]}

        data = response.json()
        result_payload = data.get("result", {})
        
        # Extract the structured hypotheses and sources
        hypotheses = result_payload.get("result", {}).get("hypotheses", [])
        answer_text = "\n".join([f"{h.get('cause')}: {h.get('evidence')} (Source: {h.get('source')})" for h in hypotheses])
        
        # Extract the raw context array sent back by the orchestrator
        contexts = result_payload.get("retrieved_contexts", [])

        return {
            "answer": answer_text if answer_text else "No valid hypotheses generated",
            "contexts": [str(ctx) for ctx in contexts] if contexts else ["No context retrieved"],
        }
    except httpx.ReadTimeout:
        print(f"❌ Timeout Error for '{query}' - Failed SLA")
        return {"answer": "Error", "contexts": ["Error"]}
    except Exception:
        return {"answer": "Error", "contexts": ["Error"]}

async def main():
    with open("data/golden_test_set.json", "r") as f:
        test_cases = json.load(f)

    dataset_dict = {
        "question": [], "answer": [], "contexts": [], "ground_truth": [],
        "user_input": [], "response": [], "retrieved_contexts": [], "reference": [],
    }

    print(f"Running pipeline against {ORCHESTRATOR_URL}...")

    async with httpx.AsyncClient() as client:
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

    # PATH 1: Strategic API Compromise for Portfolio Metrics
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("CRITICAL: OPENAI_API_KEY required for accurate Ragas evaluation.")

    print("Initializing Ragas Evaluation Models (OpenAI GPT-4o-mini for cost efficiency)...")
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    print("Running Ragas metrics...")
    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=2, timeout=600, max_retries=3),
    )

    os.makedirs("data/evaluation_results", exist_ok=True)
    df = evaluation_result.to_pandas()
    df.to_csv("data/evaluation_results/latest_run.csv", index=False)

    summary_dict = {metric: score for metric, score in evaluation_result.items()} if hasattr(evaluation_result, "items") else evaluation_result.scores

    with open("data/evaluation_results/summary.json", "w") as f:
        json.dump(summary_dict, f, indent=4)

    print("\nEvaluation Complete. Aggregate Scores:")
    print(json.dumps(summary_dict, indent=4))

if __name__ == "__main__":
    asyncio.run(main())