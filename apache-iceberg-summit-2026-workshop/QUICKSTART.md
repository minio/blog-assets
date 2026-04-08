# Workshop 01 — AIStor Tables + OpenCode Quickstart

Query a live Iceberg data lakehouse using AI-generated code in 3 commands.

- Download the [course files](workshop-01.zip)
- Unzip and `cd` into the directory

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [OpenCode](https://opencode.ai) installed
- An AIStor license file — copy it to `license/license`

---

## 3 Commands to Get Started

```bash
# 1. Start MinIO AIStor + Trino and set up the data automatically
make up

# 2. Open OpenCode — AGENTS.md loads automatically with full AIStor context
make prompt

# 3. Paste any prompt from opencode-prompts.md, then run what OpenCode generates
.venv/bin/python3 prompt-output/<filename>.py
```

That's it. `make up` handles everything — warehouse, namespace, table, and 15 seed rows of sales data. When OpenCode opens, your data is already there waiting.

---

## What gets created automatically

| | |
|--|--|
| Warehouse | `saleswarehouse` |
| Namespace | `retail` |
| Table | `retail.sales` — 15 rows, 4 regions, 3 categories, 4 sales reps |
| Query engine | Trino connected and ready at http://localhost:8090 |

---

## Services

| | URL | Login |
|--|-----|-------|
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Trino Web UI | http://localhost:8090 | — |

---

## Prompts

Open `opencode-prompts.md` — it has 15 ready-to-paste prompts covering:

- PyIceberg scans and filters
- Trino SQL analytics (revenue, rankings, trends, cross-tabs)
- Charts with matplotlib
- Adding data, schema evolution, time travel
- 4 challenges

---

## Clean up

```bash
make clean    # stops containers and wipes all data
```
