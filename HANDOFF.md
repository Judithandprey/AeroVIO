# Handoff

## Read first

1. [README](README.md)
2. [Requirements](docs/requirements.md)
3. [Setup](docs/setup.md)
4. [Original sources](docs/course/README.md)

## Current state

- Repository scaffold and English-first documentation are prepared for EECS 116/216 Project 3.
- Two original course PDFs and extracted text are included with provenance and checksums.
- viobench documentation was reviewed at commit `2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca`.
- No algorithm has been selected or implemented.
- No environment installation, model training, bundle, local evaluation, or course submission has been performed.
- No team token is stored in this repository.

## Next action

Audit the pinned task/ABI, choose a minimal baseline, initialize its workspace, and verify a reproducible build and public-set evaluation. Record results in `experiments/` and update this file with the exact verified state.

The local Windows copy is for repository/document work; upstream recommends WSL2 with files inside the Linux filesystem for Windows builds.

## Boundaries to preserve

- Latest first-stage quota is 2/team/day; generic upstream documentation still states a default of 1.
- The PDF's deadline lacks an explicit timezone; America/Los_Angeles is the course-context interpretation.
- The course figure says median ATE; the reviewed upstream guide describes mean ATE. Verify live scoring before interpreting results.
- Course PDFs retain their authors' rights and are stored in this private repository.
- Keep credentials, datasets, model checkpoints, generated binaries, and large logs outside tracked source.
- Report local build, synthetic checks, public evaluation, and held-out scoring separately.
