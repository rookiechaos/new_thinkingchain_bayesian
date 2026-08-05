"""
bayesian_world_model.py

A minimal research prototype for Idea 4:
Bayesian World Model for LLM Agents.

Core idea
---------
An agent should not call tools blindly. Before taking an action, it can use a
probabilistic world model to estimate:

    P(success | state, action)
    E[next_state_value | state, action]
    uncertainty(state, action)

The controller can then choose actions by balancing expected utility,
information gain, and action cost.

This implementation uses random-forest ensembles as lightweight approximations
to posterior uncertainty. The interfaces are intentionally modular so that the
estimators can later be replaced by SBART, Gaussian processes, Bayesian neural
networks, or calibrated ensembles.

Example
-------
    python bayesian_world_model.py

Dependencies
------------
    numpy
    scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


FloatArray = NDArray[np.float64]
ActionName = Literal["search", "retrieve", "verify", "calculate", "answer"]


ACTION_TO_INDEX: Dict[ActionName, int] = {
    "search": 0,
    "retrieve": 1,
    "verify": 2,
    "calculate": 3,
    "answer": 4,
}


@dataclass(frozen=True)
class AgentState:
    """Compact representation of an LLM agent's current state."""

    step: int
    confidence: float
    evidence_quality: float
    evidence_coverage: float
    contradiction_level: float
    problem_difficulty: float
    remaining_budget: int

    def to_array(self) -> FloatArray:
        return np.asarray(
            [
                float(self.step),
                self.confidence,
                self.evidence_quality,
                self.evidence_coverage,
                self.contradiction_level,
                self.problem_difficulty,
                float(self.remaining_budget),
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Transition:
    """One observed state-action transition."""

    state: AgentState
    action: ActionName
    success: bool
    next_state: AgentState
    reward: float
    action_cost: float


@dataclass(frozen=True)
class WorldModelPrediction:
    """Probabilistic prediction for a candidate action."""

    action: ActionName
    success_probability: float
    success_uncertainty: float
    expected_next_value: float
    value_uncertainty: float
    expected_reward: float
    acquisition_score: float


@dataclass
class ControllerConfig:
    """Weights used by the model-based action controller."""

    success_weight: float = 0.35
    value_weight: float = 0.65
    uncertainty_weight: float = 0.20
    cost_weight: float = 1.0
    optimistic_exploration: bool = True


class ProbabilisticClassifier(Protocol):
    def fit(self, x: FloatArray, y: NDArray[np.int64]) -> "ProbabilisticClassifier":
        ...

    def predict_distribution(self, x: FloatArray) -> Tuple[FloatArray, FloatArray]:
        ...


class ProbabilisticRegressor(Protocol):
    def fit(self, x: FloatArray, y: FloatArray) -> "ProbabilisticRegressor":
        ...

    def predict_distribution(self, x: FloatArray) -> Tuple[FloatArray, FloatArray]:
        ...


def encode_state_action(state: AgentState, action: ActionName) -> FloatArray:
    """Concatenate state features with a one-hot action vector."""
    action_vector = np.zeros(len(ACTION_TO_INDEX), dtype=np.float64)
    action_vector[ACTION_TO_INDEX[action]] = 1.0
    return np.concatenate([state.to_array(), action_vector])


def state_value(state: AgentState) -> float:
    """
    Heuristic scalar value used for training and demonstration.

    In a real system, replace this with task reward, verifier score,
    answer correctness probability, or learned utility.
    """
    value = (
        0.38 * state.confidence
        + 0.27 * state.evidence_quality
        + 0.25 * state.evidence_coverage
        - 0.22 * state.contradiction_level
    )
    return float(np.clip(value, 0.0, 1.0))


class TreePosteriorClassifier:
    """Approximate classification uncertainty from individual trees."""

    def __init__(
        self,
        n_estimators: int = 250,
        min_samples_leaf: int = 6,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(
        self,
        x: FloatArray,
        y: NDArray[np.int64],
    ) -> "TreePosteriorClassifier":
        self.model.fit(x, y)
        return self

    def predict_distribution(self, x: FloatArray) -> Tuple[FloatArray, FloatArray]:
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("Classifier must be fitted before prediction.")

        tree_probabilities = []
        for tree in self.model.estimators_:
            probabilities = tree.predict_proba(x)
            classes = list(tree.classes_)

            if 1 in classes:
                positive_index = classes.index(1)
                positive_probability = probabilities[:, positive_index]
            else:
                positive_probability = np.zeros(x.shape[0], dtype=np.float64)

            tree_probabilities.append(positive_probability)

        samples = np.stack(tree_probabilities, axis=0)
        return samples.mean(axis=0), samples.std(axis=0, ddof=1)


class TreePosteriorRegressor:
    """Approximate regression uncertainty from individual trees."""

    def __init__(
        self,
        n_estimators: int = 250,
        min_samples_leaf: int = 6,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(self, x: FloatArray, y: FloatArray) -> "TreePosteriorRegressor":
        self.model.fit(x, y)
        return self

    def predict_distribution(self, x: FloatArray) -> Tuple[FloatArray, FloatArray]:
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("Regressor must be fitted before prediction.")

        samples = np.stack(
            [tree.predict(x) for tree in self.model.estimators_],
            axis=0,
        )
        return samples.mean(axis=0), samples.std(axis=0, ddof=1)


class BayesianWorldModel:
    """
    Learns probabilistic transition outcomes for state-action pairs.

    The model predicts:
      1. tool/action success probability
      2. next-state value
      3. immediate reward
    """

    def __init__(
        self,
        success_model: Optional[ProbabilisticClassifier] = None,
        value_model: Optional[ProbabilisticRegressor] = None,
        reward_model: Optional[ProbabilisticRegressor] = None,
    ) -> None:
        self.success_model = success_model or TreePosteriorClassifier(
            random_state=10
        )
        self.value_model = value_model or TreePosteriorRegressor(
            random_state=11
        )
        self.reward_model = reward_model or TreePosteriorRegressor(
            random_state=12
        )

    def fit(self, transitions: Sequence[Transition]) -> "BayesianWorldModel":
        if not transitions:
            raise ValueError("At least one transition is required.")

        x = np.vstack(
            [encode_state_action(t.state, t.action) for t in transitions]
        )
        success_y = np.asarray(
            [int(t.success) for t in transitions],
            dtype=np.int64,
        )
        next_value_y = np.asarray(
            [state_value(t.next_state) for t in transitions],
            dtype=np.float64,
        )
        reward_y = np.asarray(
            [t.reward for t in transitions],
            dtype=np.float64,
        )

        self.success_model.fit(x, success_y)
        self.value_model.fit(x, next_value_y)
        self.reward_model.fit(x, reward_y)
        return self

    def predict(
        self,
        state: AgentState,
        action: ActionName,
        action_cost: float,
        config: ControllerConfig,
    ) -> WorldModelPrediction:
        x = encode_state_action(state, action)[None, :]

        success_mean, success_std = self.success_model.predict_distribution(x)
        value_mean, value_std = self.value_model.predict_distribution(x)
        reward_mean, _ = self.reward_model.predict_distribution(x)

        uncertainty_sign = 1.0 if config.optimistic_exploration else -1.0
        total_uncertainty = float(success_std[0] + value_std[0])

        acquisition_score = (
            config.success_weight * float(success_mean[0])
            + config.value_weight * float(value_mean[0])
            + uncertainty_sign * config.uncertainty_weight * total_uncertainty
            + float(reward_mean[0])
            - config.cost_weight * action_cost
        )

        return WorldModelPrediction(
            action=action,
            success_probability=float(success_mean[0]),
            success_uncertainty=float(success_std[0]),
            expected_next_value=float(value_mean[0]),
            value_uncertainty=float(value_std[0]),
            expected_reward=float(reward_mean[0]),
            acquisition_score=float(acquisition_score),
        )


class ModelBasedAgentController:
    """Chooses actions using the learned Bayesian world model."""

    def __init__(
        self,
        world_model: BayesianWorldModel,
        action_costs: Dict[ActionName, float],
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.world_model = world_model
        self.action_costs = action_costs
        self.config = config or ControllerConfig()

    def rank_actions(
        self,
        state: AgentState,
        actions: Iterable[ActionName],
    ) -> List[WorldModelPrediction]:
        predictions = [
            self.world_model.predict(
                state=state,
                action=action,
                action_cost=self.action_costs[action],
                config=self.config,
            )
            for action in actions
        ]
        return sorted(
            predictions,
            key=lambda prediction: prediction.acquisition_score,
            reverse=True,
        )

    def select_action(
        self,
        state: AgentState,
        actions: Iterable[ActionName],
    ) -> WorldModelPrediction:
        ranked = self.rank_actions(state, actions)
        if not ranked:
            raise ValueError("At least one candidate action is required.")
        return ranked[0]


def simulate_transition(
    state: AgentState,
    action: ActionName,
    rng: np.random.Generator,
    action_costs: Dict[ActionName, float],
) -> Transition:
    """Synthetic environment used only for the runnable demonstration."""

    difficulty = state.problem_difficulty
    success_probability = {
        "search": 0.82 - 0.25 * difficulty,
        "retrieve": 0.78 - 0.18 * difficulty,
        "verify": 0.72 + 0.18 * state.contradiction_level,
        "calculate": 0.88 - 0.10 * difficulty,
        "answer": 0.30 + 0.60 * state.confidence,
    }[action]
    success_probability = float(np.clip(success_probability, 0.05, 0.98))
    success = bool(rng.random() < success_probability)

    confidence = state.confidence
    quality = state.evidence_quality
    coverage = state.evidence_coverage
    contradiction = state.contradiction_level

    if action == "search":
        coverage += 0.18 if success else 0.02
        quality += 0.08 if success else -0.01
    elif action == "retrieve":
        coverage += 0.12 if success else 0.01
        quality += 0.13 if success else -0.02
    elif action == "verify":
        contradiction -= 0.18 if success else 0.02
        confidence += 0.10 if success else -0.03
    elif action == "calculate":
        confidence += 0.16 if success else -0.05
        contradiction -= 0.08 if success else -0.01
    elif action == "answer":
        confidence += 0.05 if success else -0.08

    noise = rng.normal(0.0, 0.025, size=4)
    confidence = float(np.clip(confidence + noise[0], 0.0, 1.0))
    quality = float(np.clip(quality + noise[1], 0.0, 1.0))
    coverage = float(np.clip(coverage + noise[2], 0.0, 1.0))
    contradiction = float(np.clip(contradiction + noise[3], 0.0, 1.0))

    next_state = AgentState(
        step=state.step + 1,
        confidence=confidence,
        evidence_quality=quality,
        evidence_coverage=coverage,
        contradiction_level=contradiction,
        problem_difficulty=difficulty,
        remaining_budget=max(0, state.remaining_budget - 1),
    )

    improvement = state_value(next_state) - state_value(state)
    action_cost = action_costs[action]
    reward = float(improvement - action_cost)

    return Transition(
        state=state,
        action=action,
        success=success,
        next_state=next_state,
        reward=reward,
        action_cost=action_cost,
    )


def generate_dataset(
    n_transitions: int,
    rng: np.random.Generator,
    action_costs: Dict[ActionName, float],
) -> List[Transition]:
    actions: List[ActionName] = list(ACTION_TO_INDEX)
    transitions: List[Transition] = []

    for _ in range(n_transitions):
        state = AgentState(
            step=int(rng.integers(0, 8)),
            confidence=float(rng.uniform(0.05, 0.95)),
            evidence_quality=float(rng.uniform(0.0, 1.0)),
            evidence_coverage=float(rng.uniform(0.0, 1.0)),
            contradiction_level=float(rng.uniform(0.0, 0.8)),
            problem_difficulty=float(rng.uniform(0.0, 1.0)),
            remaining_budget=int(rng.integers(1, 10)),
        )
        action = actions[int(rng.integers(0, len(actions)))]
        transitions.append(
            simulate_transition(state, action, rng, action_costs)
        )

    return transitions


def evaluate_world_model(
    model: BayesianWorldModel,
    transitions: Sequence[Transition],
) -> Tuple[float, float]:
    x = np.vstack(
        [encode_state_action(t.state, t.action) for t in transitions]
    )
    success_true = np.asarray(
        [int(t.success) for t in transitions],
        dtype=np.int64,
    )
    value_true = np.asarray(
        [state_value(t.next_state) for t in transitions],
        dtype=np.float64,
    )

    success_mean, _ = model.success_model.predict_distribution(x)
    value_mean, _ = model.value_model.predict_distribution(x)

    success_pred = (success_mean >= 0.5).astype(np.int64)
    success_accuracy = accuracy_score(success_true, success_pred)
    value_rmse = mean_squared_error(value_true, value_mean) ** 0.5

    return float(success_accuracy), float(value_rmse)


def main() -> None:
    rng = np.random.default_rng(21)

    action_costs: Dict[ActionName, float] = {
        "search": 0.045,
        "retrieve": 0.035,
        "verify": 0.055,
        "calculate": 0.030,
        "answer": 0.010,
    }

    train_data = generate_dataset(
        n_transitions=5000,
        rng=rng,
        action_costs=action_costs,
    )
    test_data = generate_dataset(
        n_transitions=1000,
        rng=rng,
        action_costs=action_costs,
    )

    world_model = BayesianWorldModel()
    world_model.fit(train_data)

    success_accuracy, value_rmse = evaluate_world_model(
        world_model,
        test_data,
    )

    controller = ModelBasedAgentController(
        world_model=world_model,
        action_costs=action_costs,
        config=ControllerConfig(
            success_weight=0.30,
            value_weight=0.70,
            uncertainty_weight=0.18,
            cost_weight=1.0,
            optimistic_exploration=True,
        ),
    )

    current_state = AgentState(
        step=2,
        confidence=0.46,
        evidence_quality=0.58,
        evidence_coverage=0.40,
        contradiction_level=0.32,
        problem_difficulty=0.70,
        remaining_budget=5,
    )

    ranked_actions = controller.rank_actions(
        current_state,
        actions=ACTION_TO_INDEX.keys(),
    )

    print("Bayesian World Model Demo")
    print("=" * 48)
    print(f"Success prediction accuracy: {success_accuracy:.4f}")
    print(f"Next-state value RMSE:       {value_rmse:.4f}")
    print(f"Current state value:         {state_value(current_state):.4f}")
    print()
    print("Candidate action ranking")
    print("-" * 48)

    for prediction in ranked_actions:
        print(
            f"{prediction.action:10s} "
            f"score={prediction.acquisition_score:+.4f} "
            f"P(success)={prediction.success_probability:.3f}"
            f"±{prediction.success_uncertainty:.3f} "
            f"E[V_next]={prediction.expected_next_value:.3f}"
            f"±{prediction.value_uncertainty:.3f} "
            f"E[reward]={prediction.expected_reward:+.3f}"
        )

    selected = ranked_actions[0]
    print()
    print(f"Selected action: {selected.action}")


if __name__ == "__main__":
    main()
