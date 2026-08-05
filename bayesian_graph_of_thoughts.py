"""
bayesian_graph_of_thoughts.py

A minimal research prototype for Idea 7:
Bayesian Graph of Thoughts (BGoT) for LLM agents.

Core idea
---------
Each reasoning node contains both:

    1. semantic content (a thought)
    2. a probabilistic belief about its usefulness or correctness

The graph supports:

- Bayesian belief updates from new evidence
- Message passing between connected thoughts
- Graph-level entropy measurement
- Information-gain-based node selection
- Entropy-based stopping

This is a lightweight research scaffold rather than a production agent.
An LLM, tool system, or external verifier can later be connected to the
`observe(...)` and `expand(...)` interfaces.

Example
-------
    python bayesian_graph_of_thoughts.py

Dependencies
------------
    numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


EPS = 1e-9


def clip_probability(value: float) -> float:
    """Keep probabilities away from exact 0 and 1."""
    return float(np.clip(value, EPS, 1.0 - EPS))


def logit(probability: float) -> float:
    """Convert probability to log-odds."""
    p = clip_probability(probability)
    return log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def bernoulli_entropy(probability: float) -> float:
    """Entropy of a Bernoulli belief in nats."""
    p = clip_probability(probability)
    return float(-(p * log(p) + (1.0 - p) * log(1.0 - p)))


@dataclass
class Belief:
    """
    Bernoulli belief over whether a thought is useful/correct.

    probability:
        Posterior probability that the thought is useful.

    evidence_strength:
        A soft confidence count. Larger values make the belief more resistant
        to weak messages from neighboring nodes.
    """

    probability: float = 0.5
    evidence_strength: float = 1.0

    def entropy(self) -> float:
        return bernoulli_entropy(self.probability)

    def update_with_likelihood_ratio(
        self,
        likelihood_ratio: float,
        evidence_weight: float = 1.0,
    ) -> None:
        """
        Bayesian update in log-odds space.

        posterior odds = prior odds * likelihood ratio

        evidence_weight allows weak or strong observations.
        """
        if likelihood_ratio <= 0.0:
            raise ValueError("likelihood_ratio must be positive.")
        if evidence_weight < 0.0:
            raise ValueError("evidence_weight must be non-negative.")

        posterior_log_odds = (
            logit(self.probability)
            + evidence_weight * log(likelihood_ratio)
        )
        self.probability = sigmoid(posterior_log_odds)
        self.evidence_strength += evidence_weight

    def blend_message(
        self,
        incoming_probability: float,
        message_weight: float,
    ) -> None:
        """
        Blend a neighboring belief in log-odds space.

        Strong prior evidence reduces the impact of a graph message.
        """
        if not 0.0 <= message_weight <= 1.0:
            raise ValueError("message_weight must be in [0, 1].")

        adaptive_weight = message_weight / max(self.evidence_strength, 1.0)
        mixed_log_odds = (
            (1.0 - adaptive_weight) * logit(self.probability)
            + adaptive_weight * logit(incoming_probability)
        )
        self.probability = sigmoid(mixed_log_odds)


@dataclass
class ThoughtNode:
    """A node in the Bayesian thought graph."""

    node_id: str
    text: str
    belief: Belief = field(default_factory=Belief)
    utility: float = 0.0
    expanded: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class ThoughtEdge:
    """
    Directed relationship between two reasoning nodes.

    relation_strength:
        Magnitude of influence in [0, 1].

    polarity:
        +1 means support.
        -1 means contradiction.
    """

    source: str
    target: str
    relation_strength: float = 0.5
    polarity: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.relation_strength <= 1.0:
            raise ValueError("relation_strength must be in [0, 1].")
        if self.polarity not in (-1, 1):
            raise ValueError("polarity must be either -1 or +1.")


@dataclass
class SelectionResult:
    """Result of selecting the next graph node to expand."""

    node_id: str
    expected_information_gain: float
    acquisition_score: float


class BayesianGraphOfThoughts:
    """Minimal Bayesian Graph-of-Thoughts implementation."""

    def __init__(
        self,
        entropy_tolerance: float = 1e-3,
        utility_weight: float = 0.3,
        uncertainty_weight: float = 1.0,
    ) -> None:
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.entropy_tolerance = entropy_tolerance
        self.utility_weight = utility_weight
        self.uncertainty_weight = uncertainty_weight
        self._previous_entropy: Optional[float] = None

    def add_node(
        self,
        node_id: str,
        text: str,
        prior_probability: float = 0.5,
        utility: float = 0.0,
        metadata: Optional[Dict[str, float]] = None,
    ) -> None:
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id!r} already exists.")

        self.nodes[node_id] = ThoughtNode(
            node_id=node_id,
            text=text,
            belief=Belief(probability=clip_probability(prior_probability)),
            utility=float(utility),
            metadata=metadata or {},
        )

    def add_edge(
        self,
        source: str,
        target: str,
        relation_strength: float = 0.5,
        polarity: int = 1,
    ) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("Both source and target nodes must exist.")

        self.edges.append(
            ThoughtEdge(
                source=source,
                target=target,
                relation_strength=relation_strength,
                polarity=polarity,
            )
        )

    def observe(
        self,
        node_id: str,
        likelihood_ratio: float,
        evidence_weight: float = 1.0,
    ) -> None:
        """
        Update a node with external evidence.

        Example:
            likelihood_ratio > 1 supports the thought.
            likelihood_ratio < 1 contradicts the thought.
        """
        self.nodes[node_id].belief.update_with_likelihood_ratio(
            likelihood_ratio=likelihood_ratio,
            evidence_weight=evidence_weight,
        )

    def graph_entropy(self) -> float:
        """Total uncertainty in the graph."""
        return float(sum(node.belief.entropy() for node in self.nodes.values()))

    def normalized_graph_entropy(self) -> float:
        """Average node entropy, normalized to [0, 1]."""
        if not self.nodes:
            return 0.0
        max_entropy = log(2.0)
        return self.graph_entropy() / (len(self.nodes) * max_entropy)

    def propagate_beliefs(
        self,
        iterations: int = 1,
        damping: float = 0.5,
    ) -> None:
        """
        Propagate support and contradiction through the graph.

        Updates are synchronous: messages are computed from the current graph
        and applied after each iteration.
        """
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        if not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be in [0, 1].")

        for _ in range(iterations):
            incoming: Dict[str, List[Tuple[float, float]]] = {
                node_id: [] for node_id in self.nodes
            }

            for edge in self.edges:
                source_probability = self.nodes[edge.source].belief.probability

                if edge.polarity == 1:
                    message_probability = source_probability
                else:
                    message_probability = 1.0 - source_probability

                message_weight = damping * edge.relation_strength
                incoming[edge.target].append(
                    (clip_probability(message_probability), message_weight)
                )

            for target, messages in incoming.items():
                for probability, weight in messages:
                    self.nodes[target].belief.blend_message(
                        incoming_probability=probability,
                        message_weight=weight,
                    )

    def expected_information_gain(
        self,
        node_id: str,
        observation_accuracy: float = 0.8,
    ) -> float:
        """
        Estimate the expected entropy reduction from observing a node.

        We model a future binary observation with known reliability:

            P(observation = true | hypothesis = true) = accuracy
            P(observation = false | hypothesis = false) = accuracy

        This is a simple value-of-information approximation.
        """
        if not 0.5 < observation_accuracy < 1.0:
            raise ValueError("observation_accuracy must be in (0.5, 1).")

        prior = self.nodes[node_id].belief.probability
        prior_entropy = bernoulli_entropy(prior)

        p_positive = (
            prior * observation_accuracy
            + (1.0 - prior) * (1.0 - observation_accuracy)
        )
        p_negative = 1.0 - p_positive

        positive_lr = observation_accuracy / (1.0 - observation_accuracy)
        negative_lr = (1.0 - observation_accuracy) / observation_accuracy

        posterior_positive = sigmoid(logit(prior) + log(positive_lr))
        posterior_negative = sigmoid(logit(prior) + log(negative_lr))

        expected_posterior_entropy = (
            p_positive * bernoulli_entropy(posterior_positive)
            + p_negative * bernoulli_entropy(posterior_negative)
        )

        return max(0.0, prior_entropy - expected_posterior_entropy)

    def select_next_node(
        self,
        observation_accuracy: float = 0.8,
    ) -> SelectionResult:
        """
        Select the unexpanded node with the best acquisition score.

        score =
            uncertainty_weight * expected_information_gain
            + utility_weight * expected_utility
        """
        candidates = [node for node in self.nodes.values() if not node.expanded]
        if not candidates:
            raise RuntimeError("No unexpanded nodes remain.")

        best: Optional[SelectionResult] = None

        for node in candidates:
            info_gain = self.expected_information_gain(
                node.node_id,
                observation_accuracy=observation_accuracy,
            )
            expected_utility = node.belief.probability * node.utility

            score = (
                self.uncertainty_weight * info_gain
                + self.utility_weight * expected_utility
            )

            result = SelectionResult(
                node_id=node.node_id,
                expected_information_gain=info_gain,
                acquisition_score=score,
            )

            if best is None or result.acquisition_score > best.acquisition_score:
                best = result

        assert best is not None
        return best

    def mark_expanded(self, node_id: str) -> None:
        self.nodes[node_id].expanded = True

    def should_stop(self) -> bool:
        """
        Stop when graph entropy has converged.

        The first call initializes the entropy reference and returns False.
        """
        current_entropy = self.graph_entropy()

        if self._previous_entropy is None:
            self._previous_entropy = current_entropy
            return False

        improvement = self._previous_entropy - current_entropy
        self._previous_entropy = current_entropy

        return improvement >= 0.0 and improvement < self.entropy_tolerance

    def summary(self) -> str:
        lines = [
            "Bayesian Graph of Thoughts",
            "=" * 40,
            f"Nodes: {len(self.nodes)}",
            f"Edges: {len(self.edges)}",
            f"Graph entropy: {self.graph_entropy():.4f}",
            f"Normalized entropy: {self.normalized_graph_entropy():.4f}",
            "",
        ]

        for node in self.nodes.values():
            lines.append(
                f"[{node.node_id}] p={node.belief.probability:.3f} "
                f"H={node.belief.entropy():.3f} "
                f"utility={node.utility:.2f} "
                f"expanded={node.expanded}"
            )
            lines.append(f"    {node.text}")

        return "\n".join(lines)


def build_demo_graph() -> BayesianGraphOfThoughts:
    """Construct a small reasoning graph for demonstration."""
    graph = BayesianGraphOfThoughts(
        entropy_tolerance=0.005,
        utility_weight=0.4,
        uncertainty_weight=1.0,
    )

    graph.add_node(
        "A",
        "The retrieved evidence directly supports hypothesis H.",
        prior_probability=0.65,
        utility=0.90,
    )
    graph.add_node(
        "B",
        "The source may be outdated and should be verified.",
        prior_probability=0.55,
        utility=0.75,
    )
    graph.add_node(
        "C",
        "A second independent source confirms the claim.",
        prior_probability=0.50,
        utility=0.95,
    )
    graph.add_node(
        "D",
        "The original conclusion should be rejected.",
        prior_probability=0.35,
        utility=0.80,
    )

    graph.add_edge("A", "C", relation_strength=0.75, polarity=1)
    graph.add_edge("B", "A", relation_strength=0.55, polarity=-1)
    graph.add_edge("C", "D", relation_strength=0.80, polarity=-1)
    graph.add_edge("A", "D", relation_strength=0.60, polarity=-1)

    return graph


def main() -> None:
    graph = build_demo_graph()
    rng = np.random.default_rng(11)

    print("Initial graph")
    print(graph.summary())
    print()

    for step in range(6):
        selection = graph.select_next_node(observation_accuracy=0.82)
        node_id = selection.node_id

        # Synthetic external verification result.
        true_support_probability = graph.nodes[node_id].belief.probability
        observation_supports = rng.random() < true_support_probability

        likelihood_ratio = 4.0 if observation_supports else 0.25

        graph.observe(
            node_id=node_id,
            likelihood_ratio=likelihood_ratio,
            evidence_weight=1.0,
        )
        graph.mark_expanded(node_id)
        graph.propagate_beliefs(iterations=2, damping=0.45)

        print(
            f"Step {step + 1}: selected {node_id}, "
            f"EIG={selection.expected_information_gain:.4f}, "
            f"score={selection.acquisition_score:.4f}, "
            f"observation={'support' if observation_supports else 'contradict'}"
        )
        print(
            f"Graph entropy={graph.graph_entropy():.4f}, "
            f"normalized={graph.normalized_graph_entropy():.4f}"
        )
        print()

        if graph.should_stop():
            print("Stopping: graph entropy has converged.")
            break

        if all(node.expanded for node in graph.nodes.values()):
            print("Stopping: all nodes have been expanded.")
            break

    print()
    print("Final graph")
    print(graph.summary())


if __name__ == "__main__":
    main()
