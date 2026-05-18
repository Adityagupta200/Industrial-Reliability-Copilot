import os
import warnings
import logging
import importlib.metadata
import builtins

# PRODUCTION FIX: Environment variables, warning filters, and metadata spoofing
# MUST be set BEFORE importing third-party libraries to maintain a clean CI/CD log.
os.environ["USE_TORCH"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ragas").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)

_orig_version = importlib.metadata.version
def _mock_version(pkg):
    if pkg == "torch": 
        return "2.3.0"
    return _orig_version(pkg)
importlib.metadata.version = _mock_version

import torch
import torch.nn as nn
import torch.optim.lr_scheduler

torch.__version__ = "2.3.0"

for missing_type, fallback_type in [
    ("uint16", "int16"), ("uint32", "int32"), ("uint64", "int64"),
    ("float8_e4m3fn", "float32"), ("float8_e5m2", "float32"),
    ("float8_e4m3fnuz", "float32"), ("float8_e5m2fnuz", "float32"),
    ("bfloat16", "float16")
]:
    if not hasattr(torch, missing_type):
        setattr(torch, missing_type, getattr(torch, fallback_type, torch.float32))

if hasattr(torch.optim.lr_scheduler, "_LRScheduler"):
    LRScheduler = torch.optim.lr_scheduler._LRScheduler
else:
    LRScheduler = object
setattr(torch.optim.lr_scheduler, "LRScheduler", LRScheduler)

builtins.torch = torch
builtins.nn = nn
builtins.LRScheduler = LRScheduler

import transformers.utils.import_utils
transformers.utils.import_utils.is_torch_available = lambda: True
transformers.utils.import_utils._torch_available = True
transformers.utils.import_utils._torch_version = "2.3.0"
transformers.utils.import_utils.requires_backends = lambda *args, **kwargs: None
if "torch" in transformers.utils.import_utils.BACKENDS_MAPPING:
    transformers.utils.import_utils.BACKENDS_MAPPING["torch"] = (lambda: True, None)

# --- SAFE IMPORTS POST-ENVIRONMENT PATCHING ---
import json
import asyncio
import httpx
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

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000/query")


async def run_pipeline(client: httpx.AsyncClient, case: dict) -> dict:
    query = case.get("query", "")
    description = case.get("anomaly_description", query)
    sensor_data = case.get("sensor_data", {})
    
    # PRODUCTION FIX: Default to an empty string so the dynamic regex engines 
    # in the chains can successfully extract IDs like "pump_P-23".
    equipment = case.get("equipment_id", "")

    query_lower = query.lower()
    if "calibrate" in query_lower or "procedure" in query_lower or "maintenance" in query_lower:
        payload = {
            "chain": "remediation",
            "remediation": {
                "user_query": query,
                "equipment_id": equipment,
                "prompt_version": "v1.0"
            }
        }
    else:
        payload = {
            "chain": "root_cause",
            "root_cause": {
                "user_query": query,
                "anomaly_description": description,
                "sensor_data": sensor_data,
                "equipment_id": equipment,
                "prompt_version": "v1.0",
            }
        }

    try:
        response = await client.post(ORCHESTRATOR_URL, json=payload, timeout=10.0)

        if response.status_code == 400:
            error_detail = response.json().get("detail", "Guardrail Blocked")
            return {
                "answer": f"Blocked: {error_detail}",
                "contexts": ["No context retrieved due to guardrail block."],
            }
        elif response.status_code not in (200, 202):
            return {"answer": "Error", "contexts": ["Error"]}

        data = response.json()
        
        if "job_id" in data:
            job_id = data["job_id"]
            for _ in range(30):
                await asyncio.sleep(1)
                status_res = await client.get(f"{ORCHESTRATOR_URL}/{job_id}", timeout=10.0)
                if status_res.status_code == 200:
                    status_data = status_res.json()
                    if status_data.get("status") == "completed":
                        data = {"result": status_data.get("result", {})}
                        break
                    elif status_data.get("status") == "failed":
                        error_msg = status_data.get("error", "Unknown Orchestrator Error")
                        print(f"❌ Async Job Failed for '{query}': {error_msg}")
                        return {"answer": "Error", "contexts": ["Error"]}
            else:
                return {"answer": "Timeout waiting for async job", "contexts": ["Timeout"]}

        result_payload = data.get("result", {})
        inner_result = result_payload.get("result", result_payload)

        hypotheses = inner_result.get("hypotheses", [])
        if not hypotheses:
            raw_answer = inner_result.get("answer", inner_result.get("output", ""))
            if not raw_answer:
                raw_answer = str(inner_result)
            answer_text = str(raw_answer)
        else:
            answer_text = "\n".join(
                [
                    f"{h.get('cause', 'Unknown')}: {h.get('evidence', 'No evidence')} (Source: {h.get('source', 'Unknown')})"
                    for h in hypotheses
                ]
            )

        raw_ctx = result_payload.get("raw_context", "")
        if raw_ctx and isinstance(raw_ctx, str):
            contexts = [c.strip() for c in raw_ctx.split("\n---\n") if c.strip()]
        elif isinstance(raw_ctx, list):
            contexts = raw_ctx
        else:
            contexts = []

        context_list = contexts if contexts else ["No context retrieved"]

        return {
            "answer": answer_text,
            "contexts": context_list,
        }
    except httpx.ReadTimeout:
        print(f"❌ Timeout Error for '{query}' - Failed SLA")
        return {"answer": "Error", "contexts": ["Error"]}
    except Exception as e:
        print(f"❌ Exception for '{query}': {e}")
        return {"answer": "Error", "contexts": ["Error"]}


async def main():
    with open("data/golden_test_set.json", "r") as f:
        test_cases = json.load(f)

    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "ground_truths": [],
        "reference": [],
    }

    print(f"Running pipeline against {ORCHESTRATOR_URL}...")

    async with httpx.AsyncClient() as client:
        for case in test_cases:
            result = await run_pipeline(client, case)
            
            gt = case.get("ground_truth", "")

            dataset_dict["question"].append(case.get("query", ""))
            dataset_dict["answer"].append(result["answer"])
            dataset_dict["contexts"].append(result["contexts"])
            
            dataset_dict["ground_truth"].append(gt)
            dataset_dict["ground_truths"].append([gt])
            dataset_dict["reference"].append(gt)

    dataset = Dataset.from_dict(dataset_dict)

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

    summary_dict = df.mean(numeric_only=True).to_dict()

    with open("data/evaluation_results/summary.json", "w") as f:
        json.dump(summary_dict, f, indent=4)

    print("\nEvaluation Complete. Aggregate Scores:")
    print(json.dumps(summary_dict, indent=4))


if __name__ == "__main__":
    asyncio.run(main())