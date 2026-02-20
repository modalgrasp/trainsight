"""
trainsight.integrations.k8s
==============================
Kubernetes GPU pod monitoring for TrainSight.

Emits events:
* ``k8s.gpu_pods``    – list of pods with GPU resource requests
* ``k8s.cost_update`` – estimated cost per pod

Usage::

    from trainsight import Dashboard
    from trainsight.integrations.k8s import KubernetesNodeMonitor

    dashboard = Dashboard()
    monitor = KubernetesNodeMonitor(dashboard, namespace="ml-training")
    monitor.poll()  # call periodically
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from kubernetes import client, config as k8s_config
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kubernetes is required: pip install kubernetes"
    ) from exc

from trainsight.core.event import Event

logger = logging.getLogger("trainsight.k8s")

# Default hourly cost per GPU (USD) – rough estimates
_DEFAULT_GPU_COST_PER_HOUR = 2.50


class KubernetesNodeMonitor:
    """
    Discover and monitor GPU pods in a Kubernetes namespace.

    Parameters
    ----------
    monitor:
        A TrainSight ``Dashboard`` instance.
    namespace:
        Kubernetes namespace to watch.
    label_selector:
        Optional label selector string (e.g. ``"app=training"``).
    gpu_cost_per_hour:
        Estimated USD cost per GPU per hour for cost estimation.
    """

    def __init__(
        self,
        monitor: Any,
        namespace: str = "default",
        label_selector: Optional[str] = None,
        gpu_cost_per_hour: float = _DEFAULT_GPU_COST_PER_HOUR,
    ) -> None:
        self.monitor = monitor
        self.namespace = namespace
        self.label_selector = label_selector
        self.gpu_cost_per_hour = float(gpu_cost_per_hour)

        # Load kubeconfig (in-cluster first, then local)
        try:
            k8s_config.load_incluster_config()
        except Exception:
            try:
                k8s_config.load_kube_config()
            except Exception as exc:
                logger.warning("Could not load kubeconfig: %s", exc)

        self._v1 = client.CoreV1Api()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def poll(self) -> list[dict[str, Any]]:
        """
        Query the cluster for GPU pods and emit events.

        Returns the list of GPU pod descriptors.
        """
        pods = self._get_gpu_pods()
        self._emit(
            "k8s.gpu_pods",
            {
                "namespace": self.namespace,
                "total_pods": len(pods),
                "pods": pods,
            },
        )

        # Emit cost estimates
        for pod in pods:
            cost = self._estimate_cost(pod)
            self._emit(
                "k8s.cost_update",
                {
                    "pod_name": pod["name"],
                    "namespace": pod["namespace"],
                    "gpu_count": pod["gpu_limit"],
                    "estimated_hourly_cost_usd": cost,
                },
            )

        return pods

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_gpu_pods(self) -> list[dict[str, Any]]:
        gpu_pods: list[dict[str, Any]] = []
        try:
            kwargs: dict[str, Any] = {"namespace": self.namespace}
            if self.label_selector:
                kwargs["label_selector"] = self.label_selector
            pod_list = self._v1.list_namespaced_pod(**kwargs)
        except Exception as exc:
            logger.warning("Failed to list pods: %s", exc)
            return gpu_pods

        for pod in pod_list.items:
            for container in pod.spec.containers:
                limits = (
                    container.resources.limits
                    if container.resources and container.resources.limits
                    else {}
                )
                gpu_limit = limits.get("nvidia.com/gpu")
                if gpu_limit:
                    gpu_pods.append(
                        {
                            "name": pod.metadata.name,
                            "namespace": pod.metadata.namespace,
                            "gpu_limit": int(gpu_limit),
                            "phase": pod.status.phase,
                            "node": pod.spec.node_name,
                            "container": container.name,
                        }
                    )
        return gpu_pods

    def _estimate_cost(self, pod: dict[str, Any]) -> float:
        return float(pod.get("gpu_limit", 1)) * self.gpu_cost_per_hour

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = Event(
            type=event_type,
            payload=payload,
            timestamp=datetime.now(timezone.utc),
        )
        bus = getattr(self.monitor, "event_bus", None)
        if bus is None:
            return
        emit_fn = getattr(bus, "emit_async", None) or bus.emit
        try:
            emit_fn(event)
        except Exception:
            pass
