# Contributing to Facetion

Thanks for your interest in improving Facetion! Bug reports, ideas and pull requests are all welcome.

## Development setup

```bash
git clone https://github.com/dzungluu25/Facetion.git
cd Facetion
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .     # lint
pytest           # tests
```

Both run automatically in CI on Windows, macOS and Linux.

## Guidelines

- Keep pull requests focused on one change.
- Add or update tests for behavior you change.
- Update the README if you add a command or option.
- Never commit face images or trained models (`data/images/`, `data/trainer.yml`).

## Good first issues

Look for issues labeled [`good first issue`](https://github.com/dzungluu25/Facetion/labels/good%20first%20issue), or pick something from the [Roadmap](README.md#roadmap).
