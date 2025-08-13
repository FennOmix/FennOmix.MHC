Contributing to FeNNet.MHC

Thanks for your interest in contributing! This guide helps you set up a dev environment, run checks, and submit high‑quality pull requests.

Getting started
- Python: 3.10–3.11 (matches CI matrix)
- OS: Linux, macOS, or Windows
- Package manager: pip or conda

Local setup (pip/venv)
1) Clone and create a virtual environment
   - python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
2) Install in editable mode with dev extras
   - pip install -U pip
   - pip install -e ".[development]"

Local setup (conda)
1) Create env and install
   - conda create -n fennet-mhc python=3.11 -y
   - conda activate fennet-mhc
   - pip install -e ".[development]"

Pre-commit hooks
- Install once: pre-commit install
- Run on all files: pre-commit run -a

Running tests
- Quick run with pytest (skips slow tests):
  - pytest -k "not slow"
- With coverage:
  - coverage run --source=fennet_mhc -m pytest -k "not slow" && coverage report
- The CI also provides helper scripts under tests/ and misc/ for conda-based runs.

Docs
- Build docs locally (optional):
  - pip install -e ".[docs]"
  - sphinx-build -b html docs ./site

Commit style and branching
- Create a feature branch from main, e.g. chore/add-community-docs
- Keep commits focused; run hooks and tests before pushing
- Reference issues in commit messages and PR descriptions when relevant

Pull requests
- Ensure: hooks pass, tests pass, and new/changed code is covered
- Add or update documentation when behavior changes
- Fill out the PR template with context, testing notes, and screenshots/output

Reporting issues
- Use the GitHub issue templates (Bug report / Feature request)
- Include repro steps, expected vs. actual behavior, environment info, and logs

Code of Conduct
- By participating you agree to abide by our Code of Conduct (see CODE_OF_CONDUCT.md). Report concerns as described therein.

