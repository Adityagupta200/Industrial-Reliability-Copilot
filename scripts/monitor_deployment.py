import argparse
import time
import sys
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_health(gateway_url):
    try:
        response = requests.get(f"{gateway_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def monitor(duration, namespace):
    # In a real environment, this would hit your Prometheus service
    # e.g., PROMETHEUS_URL = "http://prometheus-service.monitoring.svc.cluster.local:9090"
    logger.info(f"Starting post-deployment monitoring for {duration} seconds in {namespace}...")

    # Simulating the internal cluster URL for the API gateway
    gateway_url = f"http://api-gateway.{namespace}.svc.cluster.local:8000"

    end_time = time.time() + duration
    error_count = 0
    max_allowed_errors = 3

    while time.time() < end_time:
        is_healthy = check_health(gateway_url)

        if not is_healthy:
            error_count += 1
            logger.warning(f"Health check failed! Error count: {error_count}/{max_allowed_errors}")

            if error_count >= max_allowed_errors:
                logger.error("SLA Breach detected! High error rate threshold exceeded.")
                sys.exit(1)  # Exiting with 1 triggers the rollback step in GitHub Actions
        else:
            logger.info("Health check passed. Services operating nominally.")
            # Reset error count on success to only catch sustained outages
            error_count = 0

        time.sleep(30)  # Poll every 30 seconds

    logger.info("Monitoring window complete. No SLA breaches detected.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=600, help="Monitoring duration in seconds")
    parser.add_argument(
        "--namespace", type=str, default="production", help="K8s namespace to monitor"
    )
    args = parser.parse_args()

    monitor(args.duration, args.namespace)
