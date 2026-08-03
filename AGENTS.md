# AGENTS.md

## Cursor Cloud specific instructions

This repo is a single-product Python data-science project: **King Crimson**, a Canary
Islands tourism-forecasting demo. The only long-running service is a Streamlit app; the
rest is offline notebooks/scripts and a pytest suite. There is no database, backend API,
auth, or frontend build.

### Environment
- Python venv lives at `.venv/` (gitignored). It is created/refreshed by the startup
  update script (`python3 -m venv .venv` + `pip install -r requirements.txt`). Use
  `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/streamlit` (or activate the venv).
- The repo targets Python 3.11 (see `.github/workflows/ci.yml`), but it runs fine on the
  VM's Python 3.12 because `requirements.txt` uses `>=` constraints. Creating the venv
  requires the `python3.12-venv` system package (already installed in the snapshot).

### Run / test / build (standard commands, see `Makefile` and `README.md`)
- Tests: `make test` (i.e. `.venv/bin/python -m pytest tests/ -v`). 39 tests, all
  file-based/synthetic — no services or network needed.
- App (dev): `make app` (i.e. `.venv/bin/streamlit run app/streamlit_app.py`) on port
  8501. In cloud, run headless and bind all interfaces:
  `.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true`.
- No lint tooling is configured (CI runs only pytest); there is no ruff/flake8/black
  config or pre-commit hook.

### Non-obvious gotchas
- The Streamlit app uses **relative paths** (`models/`, `data/processed/...`), so it MUST
  be started from the repo root or it will fail to find the models/data.
- Pre-trained models (`models/*.pkl` + `models/registry.json`) and processed data
  (`data/processed/*.csv`) are **committed to the repo**, so the app and the E2E RevPAR
  flow work out of the box with no data rebuild. `models/` and `data/processed/` are
  documented as gitignored in the README but the artifacts are present in this checkout.
- Rebuilding data/models is optional and needs internet (ISTAC/INE APIs): `make notebooks`,
  `make metrics` (~15 min), `make reports`. Not required to run or test the product.
- Docker (`make docker-build && make docker-run`) is an optional wrapper around the same
  Streamlit app; not needed for local dev.
