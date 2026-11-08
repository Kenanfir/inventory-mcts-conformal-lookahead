# Tests

Run the project-source regression tests from the project root:

```bash
.venv/bin/python -m pytest -q tests
```

The suite checks data-cutoff and no-lookahead behavior in the M5 feature
builder, price-as-of handling, rolling conformal refits, and whole-unit
MCTS action reproducibility. Demo tests are separate; their commands are in
`scripts/gamified_demo/README.md`.
