# Contributing

What you need to know to get from a fresh clone to a merged PR.

## Quick start

```sh
git clone https://github.com/ksd3/rinexpy
cd rinexpy
uv sync --all-extras        # package + every optional extra + dev tools
uv run pytest tests/ -q     # full suite, a few seconds
```

The base package and every reader has no system dependencies. The
`numba` JIT and the C++ extension are pulled in by `--all-extras`.

If you have a system C++17 compiler and want to test the native build
path:

```sh
uv pip install -e ./native      # builds rinexpy-native locally
uv run pytest tests/test_native.py -q
```

## Repository layout

```
rinexpy/
├── src/rinexpy/        # the pure-Python package
├── native/             # the optional C++17 extension (rinexpy-native)
├── tests/              # test modules, fixtures under tests/data/
├── examples/           # runnable scripts
├── benchmarks/         # bench_obs3.py + bench_numba.py
├── docs/               # this docs site (mkdocs-material)
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

The [Architecture](../internals/architecture.md) page has a per-module
table if you are hunting for where a feature lives.

## Development workflow

### Tests

```sh
uv run pytest tests/ -q --tb=short          # all tests
uv run pytest tests/test_obs3.py -v         # one module
uv run pytest tests/ -k native -v           # by keyword
```

If you change the OBS3 reader, run the parity tests against
`georinex` too:

```sh
uv pip install georinex
uv run pytest tests/test_parity.py -q
```

### Linting and formatting

Both have to pass before a PR lands:

```sh
uv run ruff check src/ tests/ examples/ benchmarks/
uv run ruff format src/ tests/ examples/ benchmarks/
```

`ruff format` is the formatter. Line length is 100, target version is
`py311`.

### Type checking

Optional locally, not gated yet:

```sh
uv run mypy src/rinexpy
```

### Benchmarks

If a PR claims a perf change, include a fresh `bench_obs3.py` run:

```sh
uv pip install georinex
uv run python benchmarks/bench_obs3.py | tee benchmarks/last_run.txt
git add benchmarks/last_run.txt
```

## Commit conventions

[Conventional Commits](https://www.conventionalcommits.org/) with the
Angular flavour:

```
<type>(<scope>): <imperative subject ≤72 chars>

<body explaining the why, wrapped at 72 chars, optional>
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`,
`build`, `ci`.

Scopes: the module name (`io`, `version`, `time`, `headers`, `nav2`,
`nav3`, `obs2`, `obs3`, `sp3`, `keplerian`, `netcdf`, `api`, `batch`,
`cli`, `tests`, `bench`, `docs`, `rtcm3`, `nmea`, `ubx`, `sbf`,
`novatel`, `binex`, `rtcm2`, `beidou`, `lambda_ar`, `multifreq`, `rtk`,
`geodesy`, `gpstime`, `gpt2w`, `native`, ...).

Rules:

- Each commit lands one cohesive change. Many small commits are better
  than one big one. `git log --oneline` should read like a plan.
- No skipping pre-commit hooks (`--no-verify`) or signing
  (`--no-gpg-sign`).
- When a perf change needs a rewrite, the perf commit is the rewrite.
  No separate "refactor + perf" pair.

Examples from the repo:

```
feat(rtk): add float-ambiguity double-difference RTK solver
perf(obs3): rewrite hot path to drop O(N^2) xarray.merge per epoch
docs(api): rewrite API.md with all public entries + submodules
test(parity): add cross-checks against installed georinex
```

## Pull request checklist

Before opening a PR:

- [ ] Tests pass: `uv run pytest tests/ -q`
- [ ] Lint clean: `uv run ruff check src/ tests/ examples/ benchmarks/`
- [ ] Format clean: `uv run ruff format --check src/ tests/ examples/ benchmarks/`
- [ ] New public functions have NumPy-style docstrings (Parameters / Returns / Raises blocks)
- [ ] New public functions are re-exported from `src/rinexpy/__init__.py` (in `__all__`)
- [ ] New features have at least one positive and one error-path test
- [ ] If you added a new external file format, add a fixture under `tests/data/` (or synthesize one in the test module)
- [ ] If you added a new module, list it in `docs/internals/architecture.md`
- [ ] If you changed the public API, add an entry to `CHANGELOG.md` (under `## [Unreleased]`)
- [ ] Conventional-commit subject lines (`<type>(<scope>): <subject>`)

## Adding a new reader format

The shape is consistent across existing readers; cribbing from the
most recent one is the fastest path. Roughly:

1. New module at `src/rinexpy/myformat.py`.
2. Public entry: `iter_messages(stream)` for streaming feeds, or
   `load_X(fn)` returning an `xarray.Dataset` for archival formats.
3. Tests at `tests/test_myformat.py`. Synthesise fixtures inline if you
   cannot find a small public sample.
4. Document on its own page under `docs/formats/`.
5. Add a row to the README compatibility table.
6. Add a recipe to `docs/cookbook.md` (3-5 lines).
7. Re-export the public symbols from `src/rinexpy/__init__.py` and
   add them to the module index page.

## Releasing (maintainers only)

rinexpy is local-only. There is no PyPI publish step. A "release" is a
tagged commit on `main` plus a built wheel under `dist/`.

1. Update `CHANGELOG.md`: move `## [Unreleased]` content into a new
   dated `## [X.Y.Z]` section.
2. Bump `__version__` in `src/rinexpy/__init__.py` and the `version`
   field in `pyproject.toml`. If the matching `rinexpy-native`
   extension also changed, bump `native/pyproject.toml` and
   `native/python/rinexpy_native/__init__.py` too.
3. Commit with `chore: release X.Y.Z`.
4. Tag: `git tag -s vX.Y.Z -m "vX.Y.Z"`.
5. Push: `git push origin main --tags`.
6. Build the wheel and sdist so users cloning at the tag can install
   without rebuilding:
   ```sh
   uv build --wheel --sdist
   ```
   Artefacts land under `dist/`. Attach them to the GitHub Release
   manually if you want a downloadable wheel.
7. Edit the auto-created GitHub Release to paste the changelog
   excerpt as the release notes.

## Reporting bugs

- RINEX or binary parsing bugs: attach a minimal file that reproduces
  the issue (or a 1 KB excerpt). Most bugs in this space come from
  real-world files that violate the spec in surprising ways.
- Positioning or RTK bugs: include the input pseudoranges, satellite
  ECEFs, and expected output. A `tests/test_*` snippet that fails is
  ideal.
- Perf regressions: include `bench_obs3.py` output before and after.

## Code of conduct

By participating you agree to the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
See `CODE_OF_CONDUCT.md` for the full text.

## Questions

Open a [GitHub Discussion](https://github.com/ksd3/rinexpy/discussions)
for design questions. File an [Issue](https://github.com/ksd3/rinexpy/issues)
for bugs and feature requests.
