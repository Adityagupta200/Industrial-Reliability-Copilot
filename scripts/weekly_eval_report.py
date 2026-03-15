import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from src.llm_orchestrator.llm_config import load_settings


def generate_report():
    settings = load_settings()
    engine = create_engine(settings.services.incidents_db_dsn)

    one_week_ago = datetime.utcnow() - timedelta(days=7)
    query = f"""
        SELECT query_id, user_query, answer, user_feedback_score, latency_ms 
        FROM query_logs 
        WHERE timestamp > '{one_week_ago.isoformat()}'
    """

    df = pd.read_sql(query, engine)

    print("=== Weekly Production Report ===")
    print(f"Total Queries Processed: {len(df)}")
    if not df.empty:
        print(f"Average Latency: {df['latency_ms'].mean():.2f} ms")

        downvotes = df[df["user_feedback_score"] == -1]
        print(f"\nTotal Downvotes: {len(downvotes)}")
        if not downvotes.empty:
            print("\nTop Failing Queries (Downvoted):")
            for _, row in downvotes.head(10).iterrows():
                print(f"- Query: {row['user_query']}")
                print(f"  Answer: {row['answer'][:100]}...\n")


if __name__ == "__main__":
    generate_report()
