# Repository maintenance

Commit source code, tests, configurations, frozen opening datasets, compact
result summaries, and reviewed PNG/SVG figures with their plotted data and
provenance. The completed 600-round study is published in
[results/alpha_zero_600](results/alpha_zero_600/README.md).

Keep training/evaluation runs, network exports, replay buffers, optimizer
snapshots, large per-game logs, cloud launch records, local credentials,
virtual environments, Python caches, and build output outside Git. The root
`.gitignore` covers those files. Untracking an existing artifact does not
delete its local copy or remove it from older commits.

Published figure/result exports and frozen opening datasets retain their
original bytes through `.gitattributes`, so Git line-ending conversion does
not invalidate their recorded hashes. Source code and Markdown use LF endings.

## Checks before a commit

```sh
python -m pip install -r requirements-dev.txt
python -B -m pytest tests -q
git diff --check
python -B scripts/check_repository.py --worktree
```

After reviewing and staging the intended changes:

```sh
python -B scripts/check_repository.py --staged
git diff --cached --check
git diff --cached --stat
```

The repository checker reports paths and rule names without printing credential
values. It checks file sizes, generated/credential file names, common credential
patterns, README links, public summary hashes, and figure, published-source,
and plotted-data hashes. Its pattern
scan is a useful check, not a guarantee that every possible secret is detected.

For a separate audit of reachable Git history:

```sh
python -B scripts/check_repository.py --history
```

Deleting or ignoring a file in the current tree leaves older commits intact.
History rewriting requires a separate decision because it changes shared
commit IDs. Credentials discovered in history must also be revoked or rotated.

## Local evidence and optional upstream code

[LOCAL_DATA.md](../experiments/alpha_zero/LOCAL_DATA.md) describes the retained
models and training data. The figure scripts that verify complete raw logs
require those local artifacts; the published charts and compact summaries
are available in a fresh clone.

The pinned OpenSpiel fork is a Git submodule. Initialize it with
`git submodule update --init --recursive`. The additional EoH checkout is an
optional local dependency, excluded from Git; see the
[EoH setup instructions](eoh/README.md). Its upstream contract tests skip
when the package is absent.

## Archived experiment code

The 6 October 2026 cleanup keeps the core implementation and completed studies
described in the root README. The following groups were archived with user
approval:

| Group | Files | Reason |
| --- | ---: | --- |
| Generic plotting loader, batch plotters, and their test | 4 | Superseded by the four dedicated figure scripts |
| Queue-freshness pilot, phase-specific loss diagnostic, and balanced-holdout comparison | 15 | Completed side studies outside the selected publication scope; includes their dedicated tests and holdout configuration/dataset |

The archive is local under
`runs/maintenance/github-cleanup-20261006/archived/`. Its
`archive-manifest.json` records each original path, archive path, byte count,
and SHA-256 hash. All 19 files were verified after moving. Restoring them means
copying each archived file to its recorded original path after checking that
the destination is free. Their original module paths are preserved in the
archive layout; archived scripts are not intended to run from that location.

The side studies' result files, models, training histories, and verification
records remain in their original local `runs/` directories. The training loss,
C600 endpoint, greedy H*, and 13-checkpoint MCTS progression figures and their
supporting data remain versioned. Core game, training, resume, inference, and
evaluation regression tests remain in the test suite.
