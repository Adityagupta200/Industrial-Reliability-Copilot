from llm_orchestrator.clients.rag_client import RAGClient


def test_rag_client_repairs_mojibake_text_before_validation() -> None:
    client = RAGClient(
        base_url="http://rag-service",
        hybrid_path="/retrieve/hybrid",
        procedures_path="/retrieve/procedures",
        semantic_path="/retrieve/semantic",
    )

    docs = client._extract_docs(
        {
            "documents": [
                {
                    "id": "doc-1",
                    "text": "7\u00c3\u00b71.2 QBEP \u00e2\u20ac\u00a2 50\u00c2\u00baC",
                    "metadata": {"source_file": "pump_manual.pdf"},
                    "score": 0.8,
                    "source": "semantic",
                }
            ]
        }
    )

    assert docs[0]["text"] == "7/1.2 QBEP - 50 degrees C"


def test_rag_client_defaults_direct_procedure_path() -> None:
    client = RAGClient(
        base_url="http://rag-service",
        hybrid_path="/retrieve/hybrid",
        procedures_path="/retrieve/procedures",
        semantic_path="/retrieve/semantic",
    )

    assert client.procedures_direct_path == "/retrieve/procedures/direct"
