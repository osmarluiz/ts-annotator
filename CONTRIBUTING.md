# Contributing to TSA

Thanks for your interest! Issues and pull requests are welcome.

## Reporting issues

Open a GitHub issue with: what you did, what you expected, what happened, your
OS/Python versions, and — if the app was open — the last lines of the terminal
log and of the status bar. For data-dependent bugs, the synthetic demo project
(`python examples/make_demo_project.py`) is the preferred reproduction base.

## Development setup

```bash
git clone https://github.com/osmarluiz/ts-annotator
cd ts-annotator
pip install -e .[dev]        # + [train] if you'll touch the training path
pytest                       # core suite is headless (no display/GPU needed)
```

To try changes in the real app without any external data:

```bash
python examples/make_demo_project.py
tsa examples/demo_project
```

## Layout and conventions

- `core/` — pure domain logic, **no Qt**, fully testable headless.
- `ui/` — PyQt6/pyqtgraph views ("dumb views" that emit signals).
- `app/` — application layer: controller (all cross-cutting logic), workers,
  workspace loader, CLI.
- New behavior in `core/` needs a pytest; new UI flows should extend the
  offscreen smoke pattern (see `tests/` for examples with
  `QT_QPA_PLATFORM=offscreen`).
- Use `logging`, not `print`. User-facing feedback also goes to the window
  status bar via the controller.

## Pull requests

Keep PRs focused (one concern), include tests for core changes, and make sure
`pytest` passes. Describe the user-visible effect in the PR body.
