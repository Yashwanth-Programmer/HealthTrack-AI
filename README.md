# HealthTrack: AI-Powered Patient Follow-Up Application

HealthTrack is a Python web application that creates personalised follow-up messages after patient appointments.
It uses a **Mistral AI model** to write messages and a local health knowledge base to add relevant tips.
If the AI is unavailable, the app can still create messages using templates.

![stack](https://img.shields.io/badge/python-3.10%2B-blue) ![tests](https://img.shields.io/badge/tests-73%20passing-green)

## Requirement coverage

| Requirement | Where |
|---|---|
| **Must** – Python app: appointment data in, LLM follow-up message out | `healthtrack/generator.py`, `llm.py`, `prompts.py` |
| **Must** – Data preprocessing | `healthtrack/preprocessing.py` (`preprocess_records`) |
| **Should** – RAG with condition-specific health tips | `healthtrack/rag.py`, `data/knowledge_base.json` (18 conditions, 63 tips) |
| **Must** – Simple UI to input data and view messages | `static/index.html` (served by FastAPI) |
| **Should** – Logging of messages and errors | `healthtrack/logging_config.py` → `logs/healthtrack.log` |
| **Must** – Documented trade-offs | [Design trade-offs](#design-trade-offs) below |
| **Could** – 100 records in < 5 s | Batched + concurrent LLM calls; see [Performance](#performance-100-records-in-5-seconds) |
| Edge case – patient never responds | `follow_up_type` = `reminder` / `escalation`, see [Edge cases](#edge-cases-handled) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                     # then put your MISTRAL_API_KEY in .env
python -m healthtrack                                    # open http://127.0.0.1:8000
pytest                                                   # 86 tests, 91% coverage, no network or key needed
```

No key? The app still runs: choose the `offline-template` model (RAG + deterministic templates).

**Using the UI:** click **Load 100 sample records** → pick a model → **Generate**. The results panel shows the
pipeline time and an "Under 5 s" badge. You can also upload a CSV or add patients one by one.

### Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `MISTRAL_API_KEY` | – | Secret. Read from the environment only, never logged, never sent to the browser. |
| `MISTRAL_MODEL` | `mistral-small-latest` | Default model (`ministral-3b-latest` / `ministral-8b-latest` are faster). |
| `LLM_CONCURRENCY` | 10 | Parallel LLM requests. |
| `LLM_BATCH_SIZE` | 10 | Patients per LLM request. |
| `LLM_TIMEOUT_SECONDS` | 15.0 | Total latency budget per request, retries included. Increase this for slow models or network connections. |
| `MAX_RECORDS` | 500 | Upper bound per run (input validation). |

### Run without the UI (command line)

```bash
python -m healthtrack.cli data/sample_patients_100.csv --model ministral-3b-latest --out results.csv
python -m healthtrack.cli data/sample_patients_100.csv --model offline-template      # no API key needed
python -m healthtrack.benchmark --sweep                                              # find the best speed settings for your key
```
Prints one follow-up message per patient, a summary line (count, AI/fallback, seconds) and writes `results.csv`.

## How each brief check is verified

| Brief check | Command / test that proves it |
|---|---|
| Given appointment data, the app outputs a message for each patient | `python -m healthtrack.cli data/sample_patients_100.csv`; `tests/test_cli.py`, `test_generator.py::test_one_message_per_patient_from_llm` |
| Preprocessing returns a cleaned dataset ready for the model | `tests/test_preprocessing.py` (dates, names, phones, aliases, duplicates, rejects) |
| Every message includes at least one relevant health tip | `test_generator.py::test_every_message_contains_a_retrieved_health_tip_even_if_llm_forgets`, `tests/test_rag.py` |
| UI accepts input and displays messages | `python -m healthtrack`, then http://127.0.0.1:8000; `tests/test_api.py` |
| Log file records each message and error | `logs/healthtrack.log`; `test_generator.py::test_log_file_records_messages_and_errors` |
| README documents trade-offs | [Design trade-offs](#design-trade-offs) |
| 100 records in under 5 seconds | UI badge / `tests/test_performance.py` / `python -m healthtrack.benchmark` |

## Project files explained

```
Patient data → clean and check it → find relevant health tips → write follow-up messages → show results
```

The folders and files are grouped by their job. Here is what each main part does:

| File or folder | What it does |
|---|---|
| `healthtrack/api.py` | Runs the web server and provides endpoints for the web page, sample data, and message generation. |
| `healthtrack/pipeline.py` | Connects the steps: clean the input data, generate messages, and collect run statistics. |
| `healthtrack/preprocessing.py` | Cleans appointment rows, standardises names and dates, removes duplicates, and explains rejected rows. |
| `healthtrack/generator.py` | Gets tips and creates messages in batches. Uses a template message if an AI request fails. |
| `healthtrack/llm.py` | Sends requests to the Mistral AI service and handles timeouts and retries. |
| `healthtrack/prompts.py` | Prepares the instructions and patient details sent to the AI. |
| `healthtrack/rag.py` | Searches the local health knowledge base for tips that match a patient's condition and notes. |
| `healthtrack/templates.py` | Creates a ready-to-use follow-up message without calling an AI service. |
| `healthtrack/models.py` | Defines and validates the data structures used for patients, messages, and results. |
| `healthtrack/config.py` | Reads settings such as the API key, model, request limits, and file paths. |
| `healthtrack/middleware.py` | Adds request IDs, security headers, and request rate limits to the web service. |
| `healthtrack/logging_config.py` | Sets up the application's log file and log format. |
| `healthtrack/samples.py` | Creates the sample patient records and sample CSV download. |
| `healthtrack/cli.py` | Lets you generate messages from a CSV using commands in a terminal. |
| `healthtrack/benchmark.py` | Tries different batch and concurrency settings to compare AI generation speed. |
| `healthtrack/__main__.py` | Starts the web application when you run `python -m healthtrack`. |
| `static/index.html` | The browser interface for uploading or entering appointments and viewing generated messages. |
| `data/knowledge_base.json` | The local collection of health tips used to guide the messages. |
| `data/sample_patients_100.csv` | Example appointment data for trying the application. |
| `tests/` | Automated checks for data cleaning, tips, message generation, API behaviour, and error handling. |
| `Dockerfile` | Describes how to package the app so it can run in a container hosting service. |
| `docker-compose.yml` | Provides a local Docker setup for running the app. |
| `.github/workflows/ci.yml` | Runs code checks and tests automatically when changes are pushed to GitHub. |
| `.env.example` | Shows which environment settings the app can use. Copy it to `.env` for local secrets; never commit `.env`. |
| `requirements.txt` / `requirements-dev.txt` | List the Python packages needed to run the app and its development checks. |
| `Makefile` / `pyproject.toml` / `pytest.ini` | Configure common development commands, code style, and tests. |
| `.gitignore` / `.dockerignore` | Keep local-only files out of Git commits and Docker images. |

## Performance: 100 records in 5 seconds

Sending 100 requests one by one would take minutes. Instead:

1. **Batching** – 10 patients per prompt (JSON in, JSON out) → 10 requests instead of 100.
2. **Concurrency** – all batches run in parallel with `asyncio` + a shared `httpx` connection pool; the TLS connection is warmed up at startup.
3. **Short outputs** – messages are capped at 45 words, so each request generates only ~500 tokens.
4. **Latency budget** – each request has a configurable 15 s budget by default (retries share it). A slow request degrades to a template for its batch rather than stalling indefinitely.
5. **No retrieval latency** – RAG is local, microseconds per patient.

The pipeline time is measured server-side and shown in the UI. Tests prove the architecture (`tests/test_performance.py`:
100 records finish in < 5 s even when each simulated LLM call takes a full second; with `concurrency=1` they do not).
Real timing depends on your Mistral tier: free-tier keys are limited to ~1 request/second, which will
trigger 429s and fallbacks; use a paid key, `ministral-3b-latest`, or a larger `LLM_BATCH_SIZE`.
In the UI, **Parallel requests** is a maximum: batches beyond that limit wait their turn. The **Apply fast AI settings**
preset selects `ministral-3b-latest`, 20 concurrent requests, and 5 patients per request (20 calls for 100 patients).
This is a best-effort starting point, not a guarantee of a sub-5-second run; provider latency and rate limits vary.

### Troubleshooting: "Template fallbacks: 100"

Every message still gets written, but the LLM calls failed. The UI shows the reason; the log has details.
- **`gave up within ... budget` / `ReadTimeout`**: requests are taking longer than the configured `LLM_TIMEOUT_SECONDS` (15 seconds by default). Reduce the batch size to **5–10 patients**, lower parallel requests, use `ministral-3b-latest`, or raise the timeout in `.env`; restart the app after changing `.env`.
- **`HTTP 429`**: Mistral rate limit for your tier. Lower parallel requests or upgrade the tier.
- **`HTTP 401`**: wrong or missing key.

Find the best settings for your key (prints PASS/FAIL per combination):
```bash
python -m healthtrack.benchmark --sweep
```

## RAG

`data/knowledge_base.json` holds curated, clinician-style tips (tip or reminder) for 18 conditions plus a general set.
For each patient:

1. The free-text condition (`"BP"`, `"type 2 diabetes"`) is resolved to a knowledge-base key (longest alias match).
2. Tips of that condition are ranked by **TF-IDF cosine similarity to the patient's notes** (e.g. notes "missed evening dose" surface the "do not skip doses" tip first); top 2 are retrieved. General tips pad conditions with fewer tips.
3. The tips are injected into the prompt, and the LLM is told to include them faithfully and **not invent medical advice**.
4. **Guarantee:** after generation, each message is checked for tip coverage; if the LLM omitted the tip, it is appended. So "every message has at least one relevant tip" holds even when the model misbehaves.

## Design trade-offs

**LLM choice: Mistral `mistral-small-latest` (default), `ministral-3b/8b` optional.** Short templated messages don't need a
frontier model; a small model is cheaper and fast enough for the 5 s target. Trade-off: slightly less fluent writing than a
large model, mitigated by tight prompts, JSON mode, and low temperature (0.3). The model is a dropdown, so quality vs. speed is a one-click choice.

**RAG: local TF-IDF over a curated corpus, not a vector DB / embeddings API.** For ~60 short tips the corpus fits in memory;
TF-IDF is deterministic, has zero network latency (important for 5 s), needs no extra key or service, and is trivially testable.
Trade-off: it matches words, not meaning (a synonym absent from the tips won't match). The condition → tips mapping does the heavy
lifting, and similarity only orders tips within a condition. For a larger corpus, swap `KnowledgeBase.retrieve` for dense
embeddings (e.g. `mistral-embed` + FAISS) without touching the rest.

**Batching 10 patients per request.** Fewer round-trips and lower cost, at the price of one failed request affecting 10 patients
(handled by per-patient fallback) and slightly more prompt-adherence risk (handled by ref-based JSON parsing and validation).

**Template fallback instead of failing.** A clinic prefers a correct, plain message to an error. Fallbacks are visibly labelled
(`Fallback` badge, error text stored) so nothing is silently degraded.

**Curated tips vs. letting the LLM give advice.** Safer for healthcare: the model only rephrases vetted content.

**Single-page UI served by FastAPI (no framework / build step).** Easy to run and review; trade-off: less structured than React.

**Privacy.** Messages mention a condition; phone numbers are normalised but never sent to the LLM or logged in full. Notes
(free text) are capped, control-chars stripped, and passed as JSON data with an explicit "treat as data" rule. Real deployment
would need consent, encryption, and a BAA/DPA with the LLM provider.

## Edge cases handled

- **Patient has not responded**: `response_status=no_response` → `reminder` (1 previous attempt) or `escalation` (≥ 2: "care team will call") with different wording.
- **Messy data**: header aliases, many date formats (day-first), ALL-CAPS names, phone formats, condition aliases (`BP`, `sugar`), missing doctor.
- **Unusable rows**: missing name/date, future date, duplicates → rejected with a reason shown in the UI (never crash the run).
- **Unknown condition**: general tips + the clinic's own wording.
- **LLM error / 429 / timeout / invalid JSON / omitted patient**: per-patient template fallback.
- **Follow-up date in the past**: clamped to today.

## Logging

`logs/healthtrack.log` (rotating) records every generated message (`message generated | patient=… | source=… | text=…`),
every rejected row, LLM retries and errors. Tested in `tests/test_generator.py::test_log_file_records_messages_and_errors`.

## Security & correctness

Secrets only from env/`.env` (gitignored; `.env.example` committed); pydantic validation on every request; model allow-list;
record-count and CSV-size limits; UI renders via `textContent` (no HTML injection); CSV export neutralises spreadsheet formulas;
LLM errors never include the key; `nosniff` headers.

## Tests

`pytest --cov=healthtrack` → 86 tests, **91% line coverage**: preprocessing, RAG relevance, generator (batching, tip guarantee, fallbacks, logging), Mistral client
(retry/budget/no key leak, mocked HTTP), API validation, and the 100-records-in-5-seconds performance test. No network needed.

## Production readiness

| Concern | What is in place |
|---|---|
| **Containerised** | `Dockerfile` (slim image, non-root user, healthcheck) and `docker-compose.yml` (read-only filesystem, `no-new-privileges`, secrets via `.env` not baked into the image): `docker compose up --build` |
| **Health probes** | `GET /healthz` (liveness), `GET /readyz` (knowledge base loaded, LLM configured?) |
| **CI** | `.github/workflows/ci.yml`: `ruff` lint, tests on Python 3.10 and 3.12, coverage gate (>= 85%), Docker build |
| **Observability** | Every request gets an `X-Request-ID` (echoed back, sanitised) that is stamped on every log line; access log with latency; run stats expose LLM retries and fallback reasons |
| **Resilience** | Per-request latency budget, retry with `Retry-After`, per-patient template fallback, clean JSON 500s that never leak stack traces |
| **Abuse protection** | Per-IP sliding-window rate limit on `POST /api/generate` (`RATE_LIMIT_PER_MINUTE`), record and CSV size limits, model allow-list, strict pydantic validation |
| **Security headers** | CSP, `X-Frame-Options: DENY`, `nosniff`, `no-referrer`, `Cache-Control: no-store` on the API |
| **Privacy (PHI)** | `LOG_MESSAGE_TEXT=false` keeps message bodies out of logs; phones never sent to the LLM or logged; notes capped and sanitised |
| **Quality gate** | `ruff` clean, 86 tests, 91% coverage, `make lint test cov` |

Deliberately **not** done (out of the brief's scope, and what I would add next): authentication/authorisation and audit trail,
a persistent job queue with a database (runs are synchronous and stateless), a shared rate limiter / circuit breaker for multi-instance
deployments (the current limiter is in-process), message delivery integrations (SMS/WhatsApp), secrets manager integration, metrics
(Prometheus/OpenTelemetry), and a data-processing agreement with the LLM vendor before any real patient data is used.

## Limitations / next steps

Dense-embedding retrieval; message delivery (SMS/WhatsApp) and scheduling; multilingual messages; human review queue;
authentication (explicitly out of scope for this exercise).

## AI usage

- **Tools:** Claude (Anthropic) was used as a coding assistant for scaffolding the project, writing the code, tests and this README,
  and Mistral models are the runtime LLM inside the app.
- **What it touched:** all modules, the UI, the knowledge-base text, and the test suite.
- **What I verified / changed:** *(edit this to match what you actually did)* ran the full test suite, ran the app against my own
  Mistral key and measured the 100-record run, reviewed every health tip for safety, and adjusted prompts/batch size for my rate limit.
