# Virtual Kernel — Executable Surrogate of OS Kernel Dynamics

Virtual Kernel is an executable, branchable surrogate model of OS kernel dynamics under configuration interventions. Rather than acting as a static point predictor, it provides an interactive simulation environment learned from intervention–response telemetry, enabling counterfactual rollouts, uncertainty estimation, and grounded planning without live kernel disruption.

## Core Capabilities

- **Latent Dynamics & Ensembles**: State-transition models `F(z, a) = z′` learned from intervention–response evidence with quadratic action skips to capture non-linear latency curves, backed by deep ensembles for calibrated uncertainty.
- **Branchable Environment API**: Fork counterfactual worlds with copy-on-write semantics (`observe`, `intervene`, `branch`, `rollout`, `measure`, `uncertainty`, `compare`).
- **Budgeted Search & Planning**: Beam screening for rapid breadth exploration and uncertainty-penalized UCT MCTS for robust sequential policy optimization under fixed oracle budgets.
- **Multi-Layer Trust & Doubt**: Continuous validation using conservation invariants, manifold OOD novelty detection, task-dependent fidelity metrics, and adversarial agent debate.
- **Strict Safety Authority**: The deterministic `PolicyValidator` is the sole actuation authority. The surrogate proposes and explores counterfactual branches, but never writes unverified parameters to a live kernel.

## Repository Structure

| Directory / File | Description |
|---|---|
| `kernel/` | OS policy layer: KIR (Kernel Intermediate Representation) schemas, Policy ABI, deterministic `PolicyValidator`, reference `KernelSimulator`, offline RL trainer, and runtime actuator. |
| `virtual_kernel/` | Surrogate model layer: datasets, latent transition dynamics, ensemble models, branchable environment, beam/MCTS planning, truth oracle, metrology, and active probe loops. |
| `benchmarks/` | Rigorous evaluation suites: certification (`run_levels`), oracle ladder comparisons (`run_ladder`), degradation tests, zero-shot splits, and compositional tasks. |
| `experiments/` | Step-by-step experimental pipeline (`phase1_predictor` through `phase6_closed_loop`). |
| `tests/` | Comprehensive test suite covering kernel mechanics, schemas, dynamics, environment branching, and trust metrics. |
| `demo_virtual_kernel.py` | Self-contained end-to-end demonstration of learning, branching, screening, debate, and validation. |

## Quickstart

### 1. Installation

```bash
pip install pydantic numpy pytest
```

### 2. Run Tests

```bash
python -m pytest tests -q
```

### 3. Run Demo

```bash
python demo_virtual_kernel.py
```

### 4. Run Benchmarks & Experiments

```bash
# Evaluate certified surrogate levels (L1–L5)
python benchmarks/run_levels.py

# Run closed-loop active learning
python experiments/phase6_closed_loop.py

# Run oracle ladder comparison
python benchmarks/run_ladder.py
```
*(Append `--full` to any benchmark or experiment script for exhaustive evaluations).*
