# Project setup

Give the classifier a real Python project to live in, before writing a line of it.

## Key concept

**`uv` turns a dependency list into a reproducible environment.** `pyproject.toml` states what your
code needs — `pandas`, `xgboost`, `scikit-learn`, `pytest` — and `uv.lock` pins the exact versions
`uv` resolved for that list, down to the hash. Anyone who runs `uv sync` against the same lock file
gets the same environment you did; without the lock, "it works on my machine" is the best anyone
could promise.

The **`src/` layout** is what makes `import aml_triage` mean something specific: with your package
under `src/aml_triage/`, it is only importable once it is actually installed into the project's
environment (which `uv` does automatically), never accidentally importable just because a script
happens to run from the repository root. A flat layout — `aml_triage/` next to `pyproject.toml` —
would let a broken package import work by accident from one directory and fail everywhere else.

This lesson creates no model code. It creates the ground every later lesson's `import aml_triage`
stands on.

## Implementation order

1. **Initialize the project.** From the root of `aml-triage`, run `uv init --lib --name aml_triage`
   (or `uv init` followed by renaming the generated package if `uv`'s default layout differs — check
   what your `uv` version produced). Confirm it created `pyproject.toml` and a `src/aml_triage/`
   directory with an `__init__.py`.
2. **Add the dependencies.** Run `uv add pandas xgboost scikit-learn` for the library the classifier
   is built on, then `uv add --dev pytest` since the baked-in tests this tutorial runs against your
   code use it. Watch `uv` write `uv.lock` — this is the file that makes the environment
   reproducible; you do not edit it by hand.
3. **Create the test directory.** Add an empty `tests/` directory at the repository root. Nothing
   goes in it yet — the tutorial's own baked-in tests live in `aml-tutor/tests/`, not here; this one
   is yours if you ever want a scratch test of your own.
4. **Confirm the import.** Run `uv run python -c "import aml_triage"`. If it fails, the package
   layout from step 1 is wrong before anything else has a chance to be.

### If you ask the tutor to do this step for you

The doer has no shell access, so it cannot run `uv init`, `uv add`, or `uv lock` itself. Instead it
writes `pyproject.toml` by hand — with `pandas`, `xgboost`, `scikit-learn`, and `pytest`
listed as dependencies — plus `src/aml_triage/__init__.py` and an empty `tests/` marker file. It
does **not** write `uv.lock`: that file only has one correct source, which is `uv` actually
resolving your dependency list, and the first real `uv run` (whether you trigger it or the
validation check below does) creates or updates it automatically before anything else runs. If the
check fails immediately after the doer's turn, that first `uv run` is normal.

## Checks

Ask the learner to answer these in their own words:

- What does `uv.lock` pin down that `pyproject.toml` alone does not, and why would a teammate care?
- Why does putting the package under `src/aml_triage/` rather than a bare `aml_triage/` change
  whether `import aml_triage` can accidentally succeed for the wrong reason?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/001_test_project_setup.py
```

```json validation
[
  {
    "id": "001-project-setup",
    "label": "Project setup",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/001_test_project_setup.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Every later lesson assumes `aml_triage` already imports cleanly and `pytest` already runs. The next
lesson is the first one that actually reads data — if the environment here is wrong, that failure
will look like a data problem instead of a setup problem, which is exactly why this check exists on
its own before any data is involved.
