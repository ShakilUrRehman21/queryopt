# QueryOpt Lab

**An experimental research platform and interactive visualizer for studying, executing, benchmarking, and evaluating relational query optimization strategies — classical, learning, and AI-guided — on one controlled engine.**

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-0C3B2E?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-0C3B2E?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/frontend-React%20%7C%20TypeScript-6D9773?style=flat&logo=react&logoColor=white)](https://react.dev)
[![React Flow](https://img.shields.io/badge/visualizer-React%20Flow-BB8A52?style=flat)](https://reactflow.dev)
[![Tests](https://img.shields.io/badge/tests-115%20passed-6D9773?style=flat&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/license-Apache--2.0-0C3B2E?style=flat)](LICENSE)

---

## The product: paste SQL, get a faster query

Open the web app, paste a query, pick your database (PostgreSQL, MySQL, SQLite, SQL Server, Oracle, BigQuery, Snowflake, DuckDB) and press **Optimize**. You get:

- the **optimized query**, ready to copy;
- **What we changed**: rewrites that return the same rows on any data (applied automatically);
- **Worth checking**: improvements that depend on your data or intent (explained, never applied silently);
- **AI ideas** (only if an LLM key is configured): extra rewrites, clearly marked as unverified.

Optionally click **Add schema** and paste your `CREATE TABLE` / `CREATE INDEX` statements: existing indexes and primary keys are then skipped in the index advice, and `SELECT *` is expanded to your real columns.

**Windows quick start:** double-click `setup.bat` once, then `start.bat` (or run `backend.bat` and `frontend.bat` separately).

Nothing is executed; impact levels are estimates calibrated against MySQL 8 measurements. Every automatic rewrite is proven equivalent on real MySQL and DuckDB by `tests/semantic` (see `docs/AUDIT_REPORT.md`):

```bash
py -m pytest tests                                    # 357 tests, no database needed
set QUERYOPT_MYSQL_SOCKET=... (or QUERYOPT_MYSQL_HOST) # + 29 equivalence tests on your MySQL
py scripts/mysql_benchmark.py --sizes 20000 200000    # EXPLAIN / timing report
``` Code: `packages/sqlopt/`, endpoint `POST /api/optimize`.

The research engine described below (pluggable optimizers, experiments RQ1–RQ5) is still in the repository and usable from the CLI and API; it is no longer part of the web UI.

## What is QueryOpt Lab?

Production relational database management systems (PostgreSQL, DuckDB, SQLite) are designed for throughput and ACID correctness. However, their query optimizers are tightly coupled, monolithic black boxes:
- You cannot swap optimizer algorithms on the same execution engine to isolate their effects.
- You cannot easily trace *why* a particular join order or pushdown was chosen over alternatives.
- You cannot systematically measure operator-level cardinality estimation errors ($Q\text{-error}$) alongside physical I/O and CPU execution times under controlled data skew.

**QueryOpt Lab is a modular research instrument** built from first principles. It provides an end-to-end relational database stack connected to pluggable optimizers, a Volcano iterator execution engine, a benchmark suite with controllable distributions, reproducible experiment runners, and an interactive web visualizer.

---

## Visual Design Identity

QueryOpt Lab features a **calm, light, editorial visual identity** designed for high-density academic and systems research:
- **Canvas (`#E9ECE5`)**: Soft, warm, light green-tinted background.
- **Surfaces (`#FFFFFF`)**: Pure white cards, data tables, and inspection panels.
- **Deep Forest Green (`#0C3B2E`)**: Primary brand identity, navigation header, headings, and high-priority actions.
- **Sage Green (`#6D9773`)**: Secondary accents, Scan operator badges, success states, and latency indicators.
- **Warm Tan (`#BB8A52`)**: Join operator tags, primary key badges, and cardinality metadata.
- **Golden Yellow (`#FFBA00`)**: Speedup callouts, attention indicators, and Sort/Aggregate tags.

---

## Key Features

### 1. Pluggable Optimizer Architecture
QueryOpt Lab implements three distinct optimization paradigms evaluated against the exact same physical execution engine:
- **Naive Optimizer**: Unoptimized canonical plan baseline. Retains SQL AST join sequence with cross-joins and unpushed filters.
- **Rule-Based Optimizer (RBO)**: Heuristic rewriting via pattern-matching rules (Predicate Pushdown, Projection/Column Pruning, Constant Folding, Heuristic Join Ordering).
- **Cost-Based Optimizer (CBO)**: System R / Apache Calcite dynamic programming join enumeration, physical operator selection (`HashJoin` vs `NestedLoopJoin`, `SeqScan` vs `IndexScan`), and statistical cardinality estimation.

### 2. Physical Volcano Execution Engine
- Pure pull-based iterator execution model (`open()`, `next()`, `close()`).
- In-memory slotted page storage architecture (`PageStore`, `Catalog`, `TableHandle`).
- Equi-join `HashJoin` with dynamic hash table build/probe, plus fallback `NestedLoopJoin`.
- Projection pruning and predicate filtering evaluated row-by-row.

### 3. Statistics & Cardinality Estimation
- Column statistics engine computing min, max, null count, and distinct value counts.
- **Equi-depth histograms** for non-uniform data distributions.
- **HyperLogLog (HLL)** cardinality sketches for scalable distinct value tracking.
- Quantitative estimation quality measurement using **$Q\text{-error}$**:
  $$q = \max\left(\frac{\hat{c}}{c}, \frac{c}{\hat{c}}\right)$$

### 4. Benchmark Framework & Workloads (W1–W9)
Standard parameterized workloads designed to isolate specific optimizer behaviors:
- **W1**: Point lookups and single-predicate filtering (constant folding testbed).
- **W2**: Multi-predicate conjunction and predicate pushdown across joins.
- **W3**: 3-table asymmetric multi-join reordering (demonstrates **3.6x CBO speedup** over heuristic join orders).
- **W4**: Aggregation and GROUP BY cardinality reduction.
- **W5**: Zipfian skew distribution testing histogram vs uniform error.
- **W6**: Top-K ordering and LIMIT pushdown.
- **W7**: Column projection pruning across multi-table joins.
- **W8**: Correlated predicates (`humidity` follows `temperature`, `pressure` inverts it) — breaks attribute independence.
- **W9**: Repeated correlated joins for between-query feedback (adaptive optimizer).

### 5. Empirical Experiment Automation & Provenance
- Matrix experiments driven by declarative **ADR-006 YAML** files (`experiments/configs/`).
- Automated warmup passes, repeated trials, mean/stddev aggregations.
- Complete hardware and software provenance logging (`platform`, `python_version`, `git_commit`, `random_seed`, `timestamp`) written to `experiments/results/`.

### 6. Interactive Research Web Visualizer
- **SQL Workspace**: Query editor with workload presets, optimizer selector, execution metrics, and tabular result sets.
- **Interactive Plan DAG**: Powered by `@xyflow/react` and `dagre` automatic hierarchical layout. Click any operator node to inspect estimated cost, estimated vs actual rows, and applied rewrite rules.
- **Optimizer Comparison Arena**: Executes queries through all 3 optimizers side-by-side, verifies mathematical oracle result equivalence, calculates speedup ratios, and contrasts physical plan trees.
- **Workload Manager & Generator**: Seed standard tables or generate synthetic datasets with controllable scale and distribution skew (Uniform, Normal, Zipfian).
- **Catalog & Schema Inspector**: Real-time inspection of tables, column types, primary keys, and page counts.

### 7. AI Lab — learning and AI-guided optimization (v0.2)
Every AI component is *checked by the engine*, never trusted blindly.

| Optimizer / feature | What it does | Research question |
|---|---|---|
| `adaptive` | Records the true cardinality of every executed sub-expression under an **order-independent signature** and reuses it for *any* later plan — including join orders never executed (LEO-style feedback). | RQ4 |
| `learned` | Per-table **self-supervised neural estimator** (NumPy MLP) trained on thousands of auto-generated predicates; models conjunctions jointly instead of assuming independence. On the correlated W8 table: held-out p95 q-error **~3 vs ~55** for histograms. | RQ3 |
| `cost_based[qerr=10,mode=under]` | **Error injection**: distort estimates by a known q-error (under / over / deterministic random) — estimation error becomes an independent variable. | RQ2 |
| `llm_guided` | An LLM proposes the join order; the engine validates it, **costs it against the DP optimum** (`estimated_cost_ratio_vs_dp`), executes it and oracle-checks the result. Offline it runs a labelled Greedy Operator Ordering stand-in. | RQ5 |
| Plan narrator | Structured facts (est. vs actual, q-error per operator, join order, feedback/model provenance) → deterministic plain-English explanation, plus optional LLM prose that may only use those facts. | Gap 3 |
| NL → SQL | Question → SQL against the live catalog, with an **engine-in-the-loop repair cycle** (parser/validator errors are fed back). | — |
| Verified rewrites | LLM/heuristic rewrite → executed side by side → result multisets compared → median speedup → verdict (`faster`, `equivalent`, `not_equivalent`, …). | — |

LLM providers: Claude (`ANTHROPIC_API_KEY`), any OpenAI-compatible server incl. local **Ollama** (`QUERYOPT_LLM_BASE_URL`), or offline. All completions are cached on disk so LLM experiments replay bit-for-bit. See `.env.example`.

```bash
py -m queryopt.cli ai status
py -m queryopt.cli ai arena   --sql "SELECT ... "        # every optimizer family, one table
py -m queryopt.cli ai explain --sql "SELECT ... " -o adaptive
py -m queryopt.cli ai train                               # learned models vs histograms
py -m queryopt.cli ai feedback [--reset]
```

---

## Repository Structure

```
queryopt-lab/
├── apps/
│   ├── api/                     # FastAPI backend REST application
│   │   ├── main.py              # Application entrypoint & CORS
│   │   ├── models.py            # Pydantic request/response schemas
│   │   ├── serializers.py       # Logical & physical plan DAG serializers
│   │   └── routes/              # query, optimizer, schema, benchmark, experiment
│   └── web/                     # React 18 + TypeScript + Vite frontend
│       ├── src/
│       │   ├── api/client.ts    # Typed API client
│       │   ├── components/      # Navbar, PlanVisualizer, SqlWorkspace, Arena, etc.
│       │   ├── types/api.ts     # TypeScript interface definitions
│       │   ├── index.css        # Editorial design system & CSS variables
│       │   └── App.tsx          # Root tab navigation controller
├── packages/                    # Core relational database engine packages
│   ├── common/                  # Configuration, logging, types, errors
│   ├── storage/                 # Page store, catalog, slotted pages, statistics
│   ├── parser/                  # SQLGlot-based SQL parser & semantic validator
│   ├── planner/                 # AST-to-logical plan builder & printer
│   ├── optimizer/               # Naive, Rule-Based, Cost-Based DP, Cost Model
│   ├── executor/                # Volcano iterators (SeqScan, HashJoin, Filter, etc.)
│   ├── ai/                      # LLM client, plan narrator, LLM join ordering, NL→SQL, rewrite verifier
│   ├── benchmark/               # Dataset generator & W1–W9 workloads
│   ├── experiment/              # YAML config loader, matrix runner, result writer
│   └── queryopt/                # Unified QueryEngine API & CLI commands
├── experiments/
│   ├── configs/                 # Declarative YAML experiment configs (RQ1, RQ2, smoke)
│   └── results/                 # Output directories with JSON, CSV & provenance
├── docs/                        # Architecture Decision Records (ADRs) & documentation
├── tests/                       # 115 unit and integration tests
├── package.json                 # Root convenience scripts for web app
└── pyproject.toml               # Python project configuration & dependencies
```

---

## Getting Started

### Prerequisites
- **Python**: 3.12 or 3.13
- **Node.js**: 18+ (tested on Node v24, npm 10+)
- **Git**

---

### Step 1: Clone and Set Up Python Environment

```bash
git clone https://github.com/queryopt-lab/queryopt-lab.git
cd queryopt-lab

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install QueryOpt Lab core engine, dev tools, and API dependencies
pip install -e ".[dev,api]"
```

---

### Step 2: Install Web Frontend Dependencies

```bash
# From repository root:
npm install --prefix apps/web
```

---

### Step 3: Run the Application

You can launch both the backend API and frontend dev server simultaneously in two terminal windows:

#### Terminal 1 — FastAPI Backend (Port 8000)
```bash
# Starts backend server with auto-reload and automatic demo data seeding
py -m uvicorn apps.api.main:app --port 8000 --host 127.0.0.1 --reload
```
*API interactive documentation will be available at:* `http://127.0.0.1:8000/docs`

#### Terminal 2 — React Web Interface (Port 5173)
```bash
# From repository root:
npm run dev

# Or directly in apps/web:
cd apps/web
npm run dev
```
*Open your browser and navigate to:* **`http://localhost:5173`**

---

## CLI Usage

QueryOpt Lab includes a full-featured command-line interface:

### Run SQL Queries
```bash
# Execute query with Cost-Based Optimizer
py -m queryopt.cli query run --sql "SELECT orders.id, customers.name FROM orders JOIN customers ON orders.customer_id = customers.id WHERE orders.amount > 300" --optimizer cost_based

# Explain Volcano Physical Plan
py -m queryopt.cli query explain --sql "SELECT orders.id, customers.name FROM orders JOIN customers ON orders.customer_id = customers.id" --optimizer cost_based
```

### Run Research Experiments
```bash
# Execute RQ1: Cost-Based vs Rule-Based Join Ordering
py -m queryopt.cli experiment run experiments/configs/rq1_cost_vs_rule.yaml

# Execute RQ2: Cardinality Sensitivity under Skew
py -m queryopt.cli experiment run experiments/configs/rq2_cardinality_sensitivity.yaml
```

---

## Running Tests

Execute the complete 115-test verification suite across storage, parser, planner, optimizers, benchmarks, experiments, and REST API:

```bash
py -m pytest tests/ -v
```

All 115 tests run synchronously without external database dependencies.

---

## Research Experiments

Configs live in `experiments/configs/`; results (JSON/CSV + provenance) in `experiments/results/`.
Numbers below are from the included result files (scale noted in each config; Python-scale engine, single machine — treat timings as relative).

| RQ | Config | Current finding |
|---|---|---|
| RQ1 | `rq1_cost_vs_rule.yaml` | CBO and RBO both ~2× faster than naive on W3 joins. With conjunct-splitting pushdown (v0.2) RBO now matches CBO on these small joins — the earlier 3.6× figure was measured before that fix and on non-deterministic data. |
| RQ2 | `rq2_cardinality_sensitivity.yaml` | Zipfian skew: q-error 18.7 (dense head) and 9.6 (sparse tail) for the histogram estimator. |
| RQ2b | `rq2b_error_injection.yaml` | Injected error changes plans and cost estimates by orders of magnitude, but runtime is non-monotone: 10× *under*-estimation ran faster on W3 — the cost model's preferences, not only estimate quality, decide outcomes. |
| RQ3 | `rq3_correlated_estimation.yaml` | Learned estimator cuts root q-error on W8 from 5.7→1.3, 44.9→1.7 and 3.8→2.2. |
| RQ4 | `rq4_adaptive_feedback.yaml` | Clean learning curve: repetition 1 q-error 4.15, repetitions 2–5 exactly 1.0. Accurate estimates did not always yield faster plans (W9_Q2), echoing RQ2b. |
| RQ5 | `rq5_llm_join_ordering.yaml` | Shipped results use the **offline stand-in** (no API key): greedy ordering costs 1.2–2.2× the DP optimum. Set `ANTHROPIC_API_KEY` and re-run for real LLM data. |

---

## License

Distributed under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.
