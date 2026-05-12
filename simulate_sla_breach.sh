#!/bin/bash
echo "🚨 Simulating SLA Breach and SRE Monitoring..."

for i in {1..5}; do
  echo "--- Request $i ---"
  
  # 1. Client POSTs the job (Orchestrator survives and accepts it)
  JOB_ID=$(curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"root_cause": {"user_query": "Diagnose failure", "anomaly_description": "Unknown"}}' \
    | grep -o '"job_id":"[^"]*' | cut -d'"' -f4)

  echo "   ✅ POST /query returned 202 Accepted (Job ID: $JOB_ID)"

  # 2. Client checks the job status (Discovers the background worker failed)
  sleep 1
  STATUS_PAYLOAD=$(curl -s -X GET http://localhost:8000/query/$JOB_ID)
  STATUS=$(echo $STATUS_PAYLOAD | grep -o '"status":"[^"]*' | cut -d'"' -f4)
  ERROR=$(echo $STATUS_PAYLOAD | grep -o '"error":"[^"]*' | cut -d'"' -f4)
  
  echo "   ❌ Job Status: $STATUS | Reason: $ERROR"

  # 3. Load Balancer / SRE checks Readiness (Discovers the 503 SLA Breach)
  HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X GET http://localhost:8080/health/ready)
  echo "   🏥 Orchestrator /health/ready returned HTTP $HEALTH_STATUS (Service Degraded)"
  echo ""
  sleep 1
done

echo "🎉 SLA Breach simulation complete! Check Grafana to see the 503 errors spiking."