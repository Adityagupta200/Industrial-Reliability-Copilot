from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutTarget:
    name: str
    timeout_seconds: int


def run_kubectl(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["kubectl", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stdout)
    return result


def kubectl_json(args: list[str]) -> dict:
    result = run_kubectl([*args, "-o", "json"], check=True)
    return json.loads(result.stdout)


def deployment_is_serving_new_revision(namespace: str, deployment: str) -> bool:
    doc = kubectl_json(["get", "deployment", deployment, "-n", namespace])
    spec = doc.get("spec", {})
    status = doc.get("status", {})

    desired = int(spec.get("replicas") or 1)
    generation = int(doc["metadata"].get("generation") or 0)
    observed_generation = int(status.get("observedGeneration") or 0)
    updated = int(status.get("updatedReplicas") or 0)
    available = int(status.get("availableReplicas") or 0)
    unavailable = int(status.get("unavailableReplicas") or 0)

    print(
        "deployment_status",
        json.dumps(
            {
                "deployment": deployment,
                "desired": desired,
                "observed_generation": observed_generation,
                "generation": generation,
                "updated": updated,
                "available": available,
                "unavailable": unavailable,
            },
            sort_keys=True,
        ),
    )

    return (
        observed_generation >= generation
        and updated >= desired
        and available >= desired
        and unavailable == 0
    )


def force_delete_stuck_terminating_pods(namespace: str, deployment: str) -> int:
    pods = kubectl_json(["get", "pods", "-n", namespace, "-l", f"app={deployment}"])
    deleted = 0

    for item in pods.get("items", []):
        metadata = item.get("metadata", {})
        pod_name = metadata.get("name")
        deletion_timestamp = metadata.get("deletionTimestamp")
        if not pod_name or not deletion_timestamp:
            continue

        print(
            f"forcing deletion of terminating pod {namespace}/{pod_name} "
            f"from deployment {deployment}"
        )
        result = run_kubectl(
            [
                "delete",
                "pod",
                pod_name,
                "-n",
                namespace,
                "--force",
                "--grace-period=0",
            ]
        )
        print(result.stdout)
        if result.returncode == 0:
            deleted += 1

    return deleted


def wait_for_rollout(namespace: str, target: RolloutTarget) -> None:
    print(
        f"waiting for deployment/{target.name} in namespace {namespace} "
        f"for {target.timeout_seconds}s"
    )
    result = run_kubectl(
        [
            "rollout",
            "status",
            f"deployment/{target.name}",
            "-n",
            namespace,
            f"--timeout={target.timeout_seconds}s",
        ]
    )
    print(result.stdout)
    if result.returncode == 0:
        return

    if not deployment_is_serving_new_revision(namespace, target.name):
        raise RuntimeError(
            f"deployment/{target.name} did not become healthy. "
            "Leaving rollout failed for diagnostics."
        )

    deleted = force_delete_stuck_terminating_pods(namespace, target.name)
    if deleted == 0:
        raise RuntimeError(
            f"deployment/{target.name} is serving the new revision, but no terminating "
            "old pods were safe to force-delete."
        )

    retry = run_kubectl(
        [
            "rollout",
            "status",
            f"deployment/{target.name}",
            "-n",
            namespace,
            "--timeout=180s",
        ]
    )
    print(retry.stdout)
    if retry.returncode != 0:
        raise RuntimeError(f"deployment/{target.name} still did not finish rollout.")


def parse_target(value: str) -> RolloutTarget:
    if ":" not in value:
        raise argparse.ArgumentTypeError("deployment must be formatted as name:timeout_seconds")
    name, timeout = value.split(":", 1)
    try:
        timeout_seconds = int(timeout)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid timeout {timeout!r}") from exc
    return RolloutTarget(name=name, timeout_seconds=timeout_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for Kubernetes deployments and safely clear stuck terminating pods."
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--deployment", action="append", type=parse_target, required=True)
    args = parser.parse_args()

    try:
        for target in args.deployment:
            wait_for_rollout(args.namespace, target)
    except Exception as exc:
        print(f"::error::{exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
