"""
bayesian_tree_of_thoughts.py

A minimal research prototype for Idea 2: Bayesian Tree of Thoughts (BToT).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence
import numpy as np

SearchPolicy = Literal["ucb", "thompson", "mean"]

@dataclass
class PosteriorValue:
    mean: float
    std: float
    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mean, max(self.std, 1e-8)))

@dataclass
class ThoughtNode:
    node_id: str
    text: str
    depth: int
    parent_id: Optional[str]
    posterior: PosteriorValue
    terminal: bool = False
    expanded: bool = False
    pruned: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

@dataclass
class SearchConfig:
    branching_factor: int = 3
    beam_width: int = 4
    max_depth: int = 5
    exploration_weight: float = 1.0
    pruning_threshold: float = 0.20
    terminal_threshold: float = 0.92
    policy: SearchPolicy = "ucb"
    random_seed: int = 7

@dataclass
class SearchStep:
    selected_node_id: str
    selected_score: float
    created_children: List[str]
    pruned_nodes: List[str]

class ThoughtGenerator:
    def generate(self, node: ThoughtNode, branching_factor: int, rng: np.random.Generator) -> Sequence[str]:
        return [f"{node.text} -> candidate thought {i+1} at depth {node.depth+1}" for i in range(branching_factor)]

class BayesianThoughtEvaluator:
    def __init__(self, random_seed: int = 7) -> None:
        self.rng = np.random.default_rng(random_seed)
    def evaluate(self, parent: ThoughtNode, candidate_text: str, child_index: int) -> PosteriorValue:
        diminishing_returns = 0.14 * np.exp(-0.35 * parent.depth)
        branch_effect = 0.05 * np.cos(child_index + parent.depth)
        innovation = self.rng.normal(0.0, 0.06)
        mean = float(np.clip(parent.posterior.mean + diminishing_returns + branch_effect + innovation, 0.0, 1.0))
        std = float(np.clip(0.22 - 0.025 * parent.depth + abs(self.rng.normal(0.0, 0.03)), 0.03, 0.30))
        return PosteriorValue(mean, std)

class BayesianTreeOfThoughts:
    def __init__(self, root_text: str, generator: Optional[ThoughtGenerator]=None,
                 evaluator: Optional[BayesianThoughtEvaluator]=None,
                 config: Optional[SearchConfig]=None) -> None:
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.generator = generator or ThoughtGenerator()
        self.evaluator = evaluator or BayesianThoughtEvaluator(self.config.random_seed)
        self.nodes: Dict[str, ThoughtNode] = {
            "root": ThoughtNode("root", root_text, 0, None, PosteriorValue(0.35, 0.20))
        }
        self.frontier = ["root"]
        self.trace: List[SearchStep] = []
        self._node_counter = 0

    def acquisition_score(self, node: ThoughtNode) -> float:
        if self.config.policy == "mean": return node.posterior.mean
        if self.config.policy == "thompson": return node.posterior.sample(self.rng)
        return node.posterior.mean + self.config.exploration_weight * node.posterior.std

    def select_node(self) -> ThoughtNode:
        candidates = [self.nodes[i] for i in self.frontier if not self.nodes[i].expanded and not self.nodes[i].pruned and not self.nodes[i].terminal]
        if not candidates: raise RuntimeError("No expandable nodes remain.")
        return max(candidates, key=self.acquisition_score)

    def _new_node_id(self) -> str:
        self._node_counter += 1
        return f"n{self._node_counter}"

    def expand_node(self, node: ThoughtNode) -> List[str]:
        if node.depth >= self.config.max_depth:
            node.terminal = True
            return []
        child_ids = []
        for idx, text in enumerate(self.generator.generate(node, self.config.branching_factor, self.rng)):
            posterior = self.evaluator.evaluate(node, text, idx)
            child_id = self._new_node_id()
            terminal = posterior.mean >= self.config.terminal_threshold or node.depth + 1 >= self.config.max_depth
            self.nodes[child_id] = ThoughtNode(child_id, text, node.depth+1, node.node_id, posterior, terminal=terminal)
            node.children.append(child_id)
            child_ids.append(child_id)
        node.expanded = True
        self.frontier.extend(child_ids)
        return child_ids

    def prune_frontier(self) -> List[str]:
        pruned = []
        live = [self.nodes[i] for i in self.frontier if not self.nodes[i].expanded and not self.nodes[i].terminal and not self.nodes[i].pruned]
        for node in live:
            if node.posterior.mean < self.config.pruning_threshold:
                node.pruned = True; pruned.append(node.node_id)
        survivors = [n for n in live if not n.pruned]
        survivors.sort(key=self.acquisition_score, reverse=True)
        for node in survivors[self.config.beam_width:]:
            node.pruned = True; pruned.append(node.node_id)
        return pruned

    def best_terminal_node(self) -> Optional[ThoughtNode]:
        nodes = [n for n in self.nodes.values() if n.terminal and not n.pruned]
        return max(nodes, key=lambda n: n.posterior.mean) if nodes else None

    def reconstruct_path(self, node_id: str) -> List[ThoughtNode]:
        path=[]; cur=self.nodes[node_id]
        while True:
            path.append(cur)
            if cur.parent_id is None: break
            cur=self.nodes[cur.parent_id]
        return list(reversed(path))

    def search(self, max_expansions: int = 20) -> ThoughtNode:
        for _ in range(max_expansions):
            terminal=self.best_terminal_node()
            if terminal is not None and terminal.posterior.mean >= self.config.terminal_threshold:
                return terminal
            try: selected=self.select_node()
            except RuntimeError: break
            score=self.acquisition_score(selected)
            children=self.expand_node(selected)
            pruned=self.prune_frontier()
            self.trace.append(SearchStep(selected.node_id, float(score), children, pruned))
        terminal=self.best_terminal_node()
        if terminal is not None: return terminal
        active=[n for n in self.nodes.values() if not n.pruned]
        return max(active, key=lambda n:n.posterior.mean)

    def summary(self) -> str:
        lines=["Bayesian Tree of Thoughts", "="*40, f"Policy: {self.config.policy}", f"Nodes: {len(self.nodes)}", f"Expansions: {len(self.trace)}", ""]
        for n in sorted(self.nodes.values(), key=lambda x:(x.depth,x.node_id)):
            lines += [f"[{n.node_id}] depth={n.depth} mean={n.posterior.mean:.3f} std={n.posterior.std:.3f} terminal={n.terminal} pruned={n.pruned}", f"    {n.text}"]
        return "\n".join(lines)

def main() -> None:
    tree=BayesianTreeOfThoughts(
        root_text="Solve the problem using structured reasoning",
        config=SearchConfig(policy="ucb", exploration_weight=0.9)
    )
    best=tree.search(max_expansions=15)
    print(tree.summary())
    print("\nBest path\n" + "-"*40)
    for n in tree.reconstruct_path(best.node_id):
        print(f"{n.node_id}: mean={n.posterior.mean:.3f}, std={n.posterior.std:.3f}\n  {n.text}")

if __name__ == "__main__":
    main()
