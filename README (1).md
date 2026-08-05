# Bayesian Reasoning Control for LLM Agents

## Overview

This project explores a new perspective on inference-time reasoning for
Large Language Model (LLM) agents.

Instead of asking:

> **How can an LLM think longer?**

we ask:

> **How should an LLM allocate its reasoning budget under uncertainty?**

Our central hypothesis is that reasoning is fundamentally a **Bayesian
decision-making process** rather than a fixed sequence of generated
thoughts. We propose **Bayesian Reasoning Control (BRC)**, a framework
that treats reasoning as resource allocation over an uncertain thought
graph.

------------------------------------------------------------------------

# Motivation

Current inference-time scaling methods (Chain-of-Thought,
Tree-of-Thoughts, Graph-of-Thoughts, beam search, etc.) generally assume
that more reasoning leads to better performance.

However, two important questions remain largely unanswered:

1.  **When should an agent stop thinking?**
2.  **Which reasoning branch deserves more computation?**

Most existing methods use fixed reasoning depth, beam width, or token
budgets. This often wastes computation on easy problems while
under-exploring difficult ones.

We instead propose a Bayesian controller that dynamically decides
**where** and **how much** computation should be allocated.

------------------------------------------------------------------------

# Research Vision

Reasoning is viewed as sequential Bayesian decision making.

At every reasoning step, the agent decides whether additional
computation is worthwhile by balancing:

-   Expected improvement
-   Information gain
-   Computational cost

This transforms inference-time reasoning into a resource allocation
problem instead of a fixed decoding process.

------------------------------------------------------------------------

# Research Direction 1: Bayesian Compute Allocation (BCA)

## Core Question

> Can an agent learn whether additional reasoning is worth its
> computational cost?

Rather than always thinking longer, the agent estimates the **Expected
Value of Thinking (EVT)**.

### Expected Value of Thinking

For reasoning state:

\[ s_t \]

the expected benefit of one additional reasoning step is

\[ EVT(s_t) = `\mathbb{E}`{=tex} \[V(s\_{t+1})-V(s_t)\] \]

where

-   V(s) denotes the expected utility (or answer quality)
-   EVT estimates the marginal value of additional computation.

Reasoning continues only if

\[ EVT(s_t) \> C \]

where C represents the computational cost.

This naturally formulates inference-time reasoning as an **Optimal
Stopping Problem**.

### Expected Contributions

-   Adaptive reasoning depth
-   Adaptive token budget
-   Adaptive beam expansion
-   Better accuracy-compute Pareto frontier
-   Principled stopping criterion

------------------------------------------------------------------------

# Research Direction 2: Bayesian Graph of Thoughts (BGoT)

## Core Question

> Can an agent reason over uncertainty instead of only reasoning over
> thoughts?

Instead of representing each thought as plain text,

each node becomes

\[ v_i=(text_i, belief_i) \]

where

belief represents a posterior distribution over node usefulness.

Example:

-   Thought: "Search for additional evidence."

-   Belief: 0.82 ± 0.05

Thus every reasoning node carries both semantic content and uncertainty.

------------------------------------------------------------------------

## Belief Update

After new observations (tool calls, retrieval, execution, verification),

beliefs are updated using Bayesian inference

\[ P(H\|O) = `\frac{P(O|H)P(H)}{P(O)}`{=tex} \]

Reasoning therefore becomes an iterative belief refinement process.

------------------------------------------------------------------------

## Belief Propagation

Reasoning nodes are not independent.

Updating one node should influence connected nodes.

The graph performs message passing

\[ m\_{ij}=f(b_i,e\_{ij}) \]

and updates

\[ b_j=Update(b_j,m\_{ij}) \]

This produces a dynamic belief graph rather than a static reasoning
graph.

------------------------------------------------------------------------

## Graph Entropy

We introduce

\[ H(G) = `\sum`{=tex}\_i Entropy(b_i) \]

to quantify uncertainty over the entire reasoning graph.

Reasoning terminates when

\[ H_t-H\_{t+1}\<`\epsilon`{=tex} \]

instead of using a fixed reasoning budget.

------------------------------------------------------------------------

# Unified Framework

The two research directions naturally combine into one framework.

Reasoning consists of

-   an uncertain thought graph
-   Bayesian belief estimation
-   adaptive compute allocation

At every reasoning step, the controller chooses the next action by
maximizing

\[ a\^\* = `\arg`{=tex}`\max`{=tex}\_a `\left[
\mathbb{E}[\Delta U(a)]`{=tex}+
`\beta`{=tex}`\mathbb{E}`{=tex}\[`\Delta `{=tex}I(a)\] -
`\lambda `{=tex}C(a) `\right`{=tex}\] \]

where

-   ΔU : expected utility improvement
-   ΔI : expected information gain
-   C : computation cost

This provides a unified formulation for inference-time reasoning.

------------------------------------------------------------------------

# Possible Bayesian Implementations

The framework is model-agnostic.

Possible uncertainty estimators include

-   Soft Bayesian Additive Regression Trees (SBART)
-   Gaussian Processes
-   Bayesian Neural Networks
-   Deep Ensembles
-   Laplace Approximation

SBART is an attractive starting point because it naturally provides
posterior uncertainty while remaining lightweight and interpretable.

------------------------------------------------------------------------

# Experimental Plan

## Benchmarks

-   GSM8K
-   Game of 24
-   Mini Crosswords
-   HotpotQA
-   AgentBench

## Baselines

-   Chain-of-Thought
-   Tree-of-Thoughts
-   Graph-of-Thoughts
-   Beam Search
-   Monte Carlo Tree Search

## Evaluation Metrics

-   Accuracy
-   Compute cost
-   Token usage
-   Number of reasoning steps
-   Wall-clock latency
-   Compute--accuracy Pareto frontier

------------------------------------------------------------------------

# Long-Term Vision

Current LLM reasoning focuses on generating better thoughts.

This project argues that the next generation of reasoning systems should
focus on **controlling** reasoning.

Instead of asking

> "How can an LLM think longer?"

we ask

> "How can an LLM think intelligently under limited computation?"

Bayesian Reasoning Control aims to provide a principled foundation for
uncertainty-aware, computation-efficient inference-time reasoning in
future LLM agents.
