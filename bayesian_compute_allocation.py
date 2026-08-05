"""
bayesian_compute_allocation.py

Minimal research prototype for Bayesian Compute Allocation (Idea 5).

The controller estimates the Expected Value of Thinking (EVT):

    EVT(s_t) = E[V(s_{t+1}) - V(s_t) | s_t]

It continues reasoning only when the uncertainty-adjusted expected gain exceeds
another step's compute cost.

The included tree ensemble is only an approximate Bayesian uncertainty model.
It can later be replaced with SBART, a Gaussian process, a Bayesian neural
network, or another calibrated posterior estimator.

Run:
    python bayesian_compute_allocation.py

Dependencies:
    numpy
    scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

FloatArray = NDArray[np.float64]


class ProbabilisticRegressor(Protocol):
    """Interface for a model that predicts a mean and epistemic uncertainty."""

    def fit(self, x: FloatArray, y: FloatArray) -> "ProbabilisticRegressor":
        ...

    def predict_distribution(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return predictive mean and standard deviation."""
        ...


@dataclass(frozen=True)
class ReasoningState:
    """Compact features describing the current reasoning state."""

    step: int
    current_value: float
    confidence: float
    disagreement: float
    problem_difficulty: float
    remaining_budget: int

    def to_array(self) -> FloatArray:
        return np.asarray(
            [
                float(self.step),
                self.current_value,
                self.confidence,
                self.disagreement,
                self.problem_difficulty,
                float(self.remaining_budget),
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class AllocationDecision:
    """Decision returned by the compute allocator."""

    continue_reasoning: bool
    expected_gain: float
    uncertainty: float
    acquisition_value: float
    compute_cost: float


@dataclass
class ControllerConfig:
    """Hyperparameters for Bayesian compute allocation."""

    compute_cost: float = 0.03
    exploration_weight: float = 0.5
    min_remaining_budget: int = 1
    max_steps: int = 20
    optimistic_exploration: bool = True


class TreeEnsemblePosterior:
    """
    Approximate posterior uncertainty using individual tree predictions.

    This is not full Bayesian inference. It is a practical placeholder for an
    SBART-style model during early experiments.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(self, x: FloatArray, y: FloatArray) -> "TreeEnsemblePosterior":
        self.model.fit(x, y)
        return self

    def predict_distribution(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("The model must be fitted before prediction.")

        samples = np.stack(
            [tree.predict(x) for tree in self.model.estimators_],
            axis=0,
        )
        mean = samples.mean(axis=0)
        std = samples.std(axis=0, ddof=1)
        return mean.astype(np.float64), std.astype(np.float64)


class BayesianComputeAllocator:
    """Uncertainty-aware optimal-stopping controller."""

    def __init__(
        self,
        estimator: ProbabilisticRegressor,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.estimator = estimator
        self.config = config or ControllerConfig()

    def fit(
        self,
        states: Sequence[ReasoningState],
        observed_gains: Sequence[float],
    ) -> "BayesianComputeAllocator":
        if len(states) != len(observed_gains):
            raise ValueError("states and observed_gains must have equal length.")
        if not states:
            raise ValueError("At least one training sample is required.")

        x = np.vstack([state.to_array() for state in states])
        y = np.asarray(observed_gains, dtype=np.float64)
        self.estimator.fit(x, y)
        return self

    def decide(self, state: ReasoningState) -> AllocationDecision:
        if state.remaining_budget < self.config.min_remaining_budget:
            return AllocationDecision(False, 0.0, 0.0, -np.inf, self.config.compute_cost)

        if state.step >= self.config.max_steps:
            return AllocationDecision(False, 0.0, 0.0, -np.inf, self.config.compute_cost)

        x = state.to_array()[None, :]
        mean, std = self.estimator.predict_distribution(x)

        expected_gain = float(mean[0])
        uncertainty = float(std[0])
        sign = 1.0 if self.config.optimistic_exploration else -1.0

        acquisition_value = (
            expected_gain
            + sign * self.config.exploration_weight * uncertainty
            - self.config.compute_cost
        )

        return AllocationDecision(
            continue_reasoning=acquisition_value > 0.0,
            expected_gain=expected_gain,
            uncertainty=uncertainty,
            acquisition_value=acquisition_value,
            compute_cost=self.config.compute_cost,
        )


def build_evt_dataset(
    trajectories: Iterable[Sequence[ReasoningState]],
    values: Iterable[Sequence[float]],
) -> tuple[list[ReasoningState], list[float]]:
    """Construct one-step EVT labels from completed reasoning trajectories."""

    state_rows: list[ReasoningState] = []
    gain_rows: list[float] = []

    for trajectory, trajectory_values in zip(trajectories, values):
        if len(trajectory) != len(trajectory_values):
            raise ValueError("Each trajectory must align with its values.")

        for t in range(len(trajectory) - 1):
            state_rows.append(trajectory[t])
            gain_rows.append(float(trajectory_values[t + 1] - trajectory_values[t]))

    return state_rows, gain_rows


def simulate_reasoning_trajectory(
    difficulty: float,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[list[ReasoningState], list[float]]:
    """Generate a synthetic reasoning trajectory for a runnable demonstration."""

    states: list[ReasoningState] = []
    values: list[float] = []
    current_value = float(rng.uniform(0.05, 0.25))
    confidence = float(rng.uniform(0.1, 0.3))

    for step in range(max_steps):
        disagreement = float(
            np.clip(
                difficulty * (1.0 - confidence) + rng.normal(0.0, 0.05),
                0.0,
                1.0,
            )
        )

        states.append(
            ReasoningState(
                step=step,
                current_value=current_value,
                confidence=confidence,
                disagreement=disagreement,
                problem_difficulty=difficulty,
                remaining_budget=max_steps - step,
            )
        )
        values.append(current_value)

        saturation = max(0.0, 1.0 - current_value)
        base_gain = (
            0.20
            * saturation
            * np.exp(-0.20 * step)
            * (0.65 + 0.70 * difficulty)
        )
        noise = rng.normal(0.0, 0.025 + 0.025 * difficulty)
        gain = float(np.clip(base_gain + noise, -0.08, 0.25))

        current_value = float(np.clip(current_value + gain, 0.0, 1.0))
        confidence = float(
            np.clip(
                confidence + 0.12 * gain + 0.03 + rng.normal(0.0, 0.015),
                0.0,
                1.0,
            )
        )

    return states, values


def run_controlled_reasoning(
    trajectory: Sequence[ReasoningState],
    values: Sequence[float],
    allocator: BayesianComputeAllocator,
) -> tuple[int, float, list[AllocationDecision]]:
    """Apply the learned stopping policy to a pre-generated trajectory."""

    if len(trajectory) != len(values):
        raise ValueError("trajectory and values must have equal length.")

    decisions: list[AllocationDecision] = []

    for idx, state in enumerate(trajectory[:-1]):
        decision = allocator.decide(state)
        decisions.append(decision)
        if not decision.continue_reasoning:
            return idx + 1, float(values[idx]), decisions

    return len(trajectory), float(values[-1]), decisions


def main() -> None:
    rng = np.random.default_rng(7)

    train_trajectories: list[list[ReasoningState]] = []
    train_values: list[list[float]] = []

    for _ in range(600):
        states, values = simulate_reasoning_trajectory(
            difficulty=float(rng.uniform(0.0, 1.0)),
            max_steps=12,
            rng=rng,
        )
        train_trajectories.append(states)
        train_values.append(values)

    train_states, train_gains = build_evt_dataset(train_trajectories, train_values)

    estimator = TreeEnsemblePosterior(
        n_estimators=250,
        min_samples_leaf=8,
        random_state=7,
    )
    allocator = BayesianComputeAllocator(
        estimator=estimator,
        config=ControllerConfig(
            compute_cost=0.035,
            exploration_weight=0.35,
            max_steps=12,
            optimistic_exploration=True,
        ),
    )
    allocator.fit(train_states, train_gains)

    test_trajectories: list[list[ReasoningState]] = []
    test_values: list[list[float]] = []

    for _ in range(150):
        states, values = simulate_reasoning_trajectory(
            difficulty=float(rng.uniform(0.0, 1.0)),
            max_steps=12,
            rng=rng,
        )
        test_trajectories.append(states)
        test_values.append(values)

    test_states, test_gains = build_evt_dataset(test_trajectories, test_values)
    x_test = np.vstack([state.to_array() for state in test_states])
    mean, std = estimator.predict_distribution(x_test)
    rmse = mean_squared_error(np.asarray(test_gains), mean) ** 0.5

    adaptive_steps: list[int] = []
    adaptive_values: list[float] = []
    fixed_values: list[float] = []
    fixed_depth = 6

    for trajectory, values in zip(test_trajectories, test_values):
        steps_used, final_value, _ = run_controlled_reasoning(
            trajectory,
            values,
            allocator,
        )
        adaptive_steps.append(steps_used)
        adaptive_values.append(final_value)
        fixed_values.append(values[min(fixed_depth - 1, len(values) - 1)])

    print("Bayesian Compute Allocation Demo")
    print("=" * 40)
    print(f"Training samples: {len(train_states)}")
    print(f"Test EVT RMSE:     {rmse:.4f}")
    print(f"Mean uncertainty:  {float(std.mean()):.4f}")
    print()
    print("Adaptive policy")
    print(f"  Mean steps:      {float(np.mean(adaptive_steps)):.2f}")
    print(f"  Mean value:      {float(np.mean(adaptive_values)):.4f}")
    print()
    print(f"Fixed-depth policy (depth={fixed_depth})")
    print(f"  Mean steps:      {fixed_depth:.2f}")
    print(f"  Mean value:      {float(np.mean(fixed_values)):.4f}")


if __name__ == "__main__":
    main()
