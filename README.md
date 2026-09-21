# Multi-Turn AI Misuse Detection Framework (SentinelLoop)

[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenRouter](https://img.shields.io/badge/Gateway-OpenRouter-6366f1.svg)](https://openrouter.ai)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](http://localhost:8501)

**Multi-Turn AI Misuse Detection Framework (SentinelLoop)** is a real-time, production-grade middleware architecture that intercepts multi-turn conversations between users and LLMs. By continuously evaluating dialogue histories with cumulative 5-dimensional (5D) behavioral vector scoring, it detects behavioral drift and dynamically applies defensive prompt patches or safe overrides to enforce safety alignment mid-session.

---

## Key Features

- **Cumulative 5D Behavioral Scoring**: Tracks dialogue histories across five dimensions: Compliance, Hedging, Sycophancy, Deception Markers, and Verbosity.
- **Drift Velocity Analysis**: Computes turn-over-turn compliance acceleration ($\Delta = \text{compliance}_t - \text{compliance}_{t-1}$) to detect sudden drift spikes before safety failures occur.
- **Dynamic Prompt Patching (Strategy A)**: Automatically injects targeted safety boundary directives into the active system context for subsequent turns when drift is detected ($\ge 0.50$).
- **Safe Refusal Overrides (Strategy B)**: Automatically intercepts and replaces critical policy violations ($\ge 0.85$) with standardized safe refusals.
- **Comprehensive Evaluation Harness**: Quantifies defense efficacy using Intervention Efficacy Rate (IER), False Positive Disruption (FPD), and Alignment Recovery Latency.
- **Live Operations Dashboard**: Streamlit-powered console featuring turn-by-turn chat inspection, Plotly compliance trajectory charts with threshold zones, and an interactive guardrail sandbox.

---

## Architecture Overview

```
User Prompt ──▶ [ GuardrailInterceptor Proxy ] ──▶ Target LLM
                       │                              │
                       │◀── Target Model Response ────┘
                       ▼
          [ Stateful Behavioral Monitor ]
               - 5D Vector Scoring
               - Drift Velocity (Δ)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[ Compliance ≥ 0.50 ]        [ Compliance ≥ 0.85 ]
   Strategy A:                  Strategy B:
Dynamic Prompt Patch          Safe Refusal Override
 (Injected into Context)      (Response Replaced)
       │                               │
       └───────────────┬───────────────┘
                       ▼
          Client Receives Safe Output
```

---

## Directory Structure

```
SentinelLoop/
├── data/
│   └── seed_conversations.csv               # Multi-turn benign & adversarial scenarios
├── src/
│   ├── __init__.py                          # Package metadata
│   ├── config.py                            # OpenRouter API client, model settings, & thresholds
│   ├── monitor/
│   │   ├── __init__.py                      # Monitor package exports
│   │   └── classifier.py                    # 5D BehavioralVector scoring & drift velocity
│   ├── interceptor/
│   │   ├── __init__.py                      # Interceptor package exports
│   │   └── middleware.py                    # GuardrailInterceptor (Strategy A & Strategy B)
│   └── eval/
│       ├── __init__.py                      # Evaluation harness exports
│       └── simulator.py                     # MultiTurnSimulator (Baseline vs. Defended)
├── app/
│   └── dashboard.py                         # Streamlit real-time operations dashboard
├── outputs/
│   └── defense_evaluation_results.json      # Benchmark evaluation results & telemetry
├── run_defense_evals.py                     # CLI evaluation orchestrator
└── requirements.txt                         # Dependency specifications
```

---

## Core Metrics

### 1. Per-Turn 5D Behavioral Vector
- **Compliance (0.0–1.0)**: Degree of adherence to unauthorized or potentially harmful user intent.
- **Hedging (0.0–1.0)**: Presence of moralizing qualifiers, preachy caveats, or disclaimers.
- **Sycophancy (0.0–1.0)**: Deference to unauthorized user authority claims or validation of risky assumptions.
- **Deception Markers (0.0–1.0)**: Evasive half-truths, omissions, or unverified instructions.
- **Verbosity Tokens (int)**: Token length of target response.

### 2. Aggregate Defense Metrics
- **Intervention Efficacy Rate (IER)**: Percentage of drifting conversations successfully halted or recovered ($100\%$ on benchmark dataset).
- **False Positive Disruption (FPD)**: Rate of benign conversations unnecessarily intervened ($0\%$ on benchmark dataset).
- **Alignment Recovery Latency**: Average number of turns required to restore safe compliance post-intervention ($1.0$ turns on benchmark dataset).

---

## Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Set up your OpenRouter API key in an environment variable or `.env` file:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
TARGET_MODEL=openai/gpt-4o-mini
JUDGE_MODEL=anthropic/claude-haiku-4.5
```

### 3. Run Defense Evaluations
Run the simulation harness comparing undefended baselines against SentinelLoop:
```bash
# High-speed offline mock mode
python run_defense_evals.py --mock

# Live API evaluation across target models
python run_defense_evals.py --models openai/gpt-4o-mini anthropic/claude-haiku-4.5
```

### 4. Launch Operations Dashboard
Launch the interactive Streamlit dashboard:
```bash
streamlit run app/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
