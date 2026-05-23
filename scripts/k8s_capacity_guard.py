from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


CPU_MILLI = 1000
MEMORY_MULTIPLIERS = {
    "Ki": 1 / 1024,
    "Mi": 1,
    "Gi": 1024,
    "Ti": 1024 * 1024,
    "K": 1 / 1000,
    "M": 1,
    "G": 1000,
    "T": 1000 * 1000,
}


@dataclass(frozen=True)
class Capacity:
    cpu_millicores: int
    memory_mib: int

    def __add__(self, other: "Capacity") -> "Capacity":
        return Capacity(
            cpu_millicores=self.cpu_millicores + other.cpu_millicores,
            memory_mib=self.memory_mib + other.memory_mib,
        )

    def __sub__(self, other: "Capacity") -> "Capacity":
        return Capacity(
            cpu_millicores=self.cpu_millicores - other.cpu_millicores,
            memory_mib=self.memory_mib - other.memory_mib,
        )


ZERO_CAPACITY = Capacity(cpu_millicores=0, memory_mib=0)


def run_kubectl(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", *args, "-o", "json"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return json.loads(result.stdout)


def parse_cpu(value: str | None) -> int:
    if not value:
        return 0
    if value.endswith("m"):
        return int(value[:-1])
    return int(float(value) * CPU_MILLI)


def parse_memory(value: str | None) -> int:
    if not value:
        return 0
    match = re.fullmatch(r"([0-9.]+)([A-Za-z]+)?", value)
    if not match:
        raise ValueError(f"Unsupported Kubernetes memory quantity: {value!r}")
    amount = float(match.group(1))
    suffix = match.group(2) or ""
    if not suffix:
        return int(amount / (1024 * 1024))
    multiplier = MEMORY_MULTIPLIERS.get(suffix)
    if multiplier is None:
        raise ValueError(f"Unsupported Kubernetes memory suffix: {suffix!r}")
    return int(amount * multiplier)


def resource_request(container: dict[str, Any]) -> Capacity:
    requests = container.get("resources", {}).get("requests", {})
    return Capacity(
        cpu_millicores=parse_cpu(requests.get("cpu")),
        memory_mib=parse_memory(requests.get("memory")),
    )


def pod_request(pod: dict[str, Any]) -> Capacity:
    spec = pod.get("spec", {})
    app_request = ZERO_CAPACITY
    for container in spec.get("containers", []):
        app_request += resource_request(container)

    max_init_cpu = 0
    max_init_memory = 0
    for container in spec.get("initContainers", []):
        request = resource_request(container)
        max_init_cpu = max(max_init_cpu, request.cpu_millicores)
        max_init_memory = max(max_init_memory, request.memory_mib)

    return Capacity(
        cpu_millicores=max(app_request.cpu_millicores, max_init_cpu),
        memory_mib=max(app_request.memory_mib, max_init_memory),
    )


def node_allocatable(node: dict[str, Any]) -> Capacity:
    allocatable = node.get("status", {}).get("allocatable", {})
    return Capacity(
        cpu_millicores=parse_cpu(allocatable.get("cpu")),
        memory_mib=parse_memory(allocatable.get("memory")),
    )


def node_is_ready(node: dict[str, Any]) -> bool:
    if node.get("spec", {}).get("unschedulable"):
        return False
    conditions = node.get("status", {}).get("conditions", [])
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
    )


def non_terminal_pods(pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal = {"Succeeded", "Failed"}
    return [pod for pod in pods if pod.get("status", {}).get("phase") not in terminal]


def check_capacity(args: argparse.Namespace) -> int:
    nodes = run_kubectl(["get", "nodes"]).get("items", [])
    pods = run_kubectl(["get", "pods", "--all-namespaces"]).get("items", [])
    ready_nodes = [node for node in nodes if node_is_ready(node)]
    node_names = {node["metadata"]["name"] for node in ready_nodes}

    node_free: dict[str, Capacity] = {
        node["metadata"]["name"]: node_allocatable(node) for node in ready_nodes
    }

    for pod in non_terminal_pods(pods):
        node_name = pod.get("spec", {}).get("nodeName")
        if node_name not in node_names:
            continue
        node_free[node_name] = node_free[node_name] - pod_request(pod)

    total_free = ZERO_CAPACITY
    max_node_free = ZERO_CAPACITY
    for free in node_free.values():
        total_free += free
        max_node_free = Capacity(
            cpu_millicores=max(max_node_free.cpu_millicores, free.cpu_millicores),
            memory_mib=max(max_node_free.memory_mib, free.memory_mib),
        )

    summary = {
        "ready_nodes": len(ready_nodes),
        "required_ready_nodes": args.min_ready_nodes,
        "total_free_cpu_millicores": total_free.cpu_millicores,
        "required_total_free_cpu_millicores": args.min_free_cpu_millicores,
        "total_free_memory_mib": total_free.memory_mib,
        "required_total_free_memory_mib": args.min_free_memory_mib,
        "max_node_free_cpu_millicores": max_node_free.cpu_millicores,
        "required_node_free_cpu_millicores": args.min_node_free_cpu_millicores,
        "max_node_free_memory_mib": max_node_free.memory_mib,
        "required_node_free_memory_mib": args.min_node_free_memory_mib,
    }
    print("cluster_capacity", json.dumps(summary, sort_keys=True))

    failures = []
    if len(ready_nodes) < args.min_ready_nodes:
        failures.append(f"ready nodes {len(ready_nodes)} < {args.min_ready_nodes}")
    if total_free.cpu_millicores < args.min_free_cpu_millicores:
        failures.append(
            f"free CPU {total_free.cpu_millicores}m < {args.min_free_cpu_millicores}m"
        )
    if total_free.memory_mib < args.min_free_memory_mib:
        failures.append(
            f"free memory {total_free.memory_mib}Mi < {args.min_free_memory_mib}Mi"
        )
    if max_node_free.cpu_millicores < args.min_node_free_cpu_millicores:
        failures.append(
            "largest schedulable node free CPU "
            f"{max_node_free.cpu_millicores}m < {args.min_node_free_cpu_millicores}m"
        )
    if max_node_free.memory_mib < args.min_node_free_memory_mib:
        failures.append(
            "largest schedulable node free memory "
            f"{max_node_free.memory_mib}Mi < {args.min_node_free_memory_mib}Mi"
        )

    if failures:
        print(
            "::error::Insufficient EKS rollout headroom for Phase 9 zero-downtime "
            f"deployment: {'; '.join(failures)}"
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail fast when the cluster lacks schedulable headroom for rollout surge pods."
    )
    parser.add_argument("--min-ready-nodes", type=int, default=3)
    parser.add_argument("--min-free-cpu-millicores", type=int, default=2000)
    parser.add_argument("--min-free-memory-mib", type=int, default=2048)
    parser.add_argument("--min-node-free-cpu-millicores", type=int, default=500)
    parser.add_argument("--min-node-free-memory-mib", type=int, default=1536)
    args = parser.parse_args()

    try:
        return check_capacity(args)
    except Exception as exc:
        print(f"::error::Unable to evaluate cluster capacity: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
