# Contributing

Contributions are welcome: new optimizer implementations, Canon entries, corrections, and test improvements.

## Setup

```bash
conda env create -f environment.yml
conda activate zij-optim
pytest pytorch/tests -q
ruff check pytorch
```

## Adding an optimizer implementation

1. **Source and license.** Prefer the original authors' repository. The source license must be redistribution-compatible (Apache-2.0, MIT, BSD). GPL, LGPL, CC non-commercial, and unlicensed sources are not accepted; list such methods in the Canon only.
2. **Provenance header.** Every vendored or adapted file starts with the upstream URL, pinned commit, copyright holder, and license, in the format used throughout `pytorch/zij/contrib/`.
3. **Module placement.** One module per method under the matching category subpackage (`first_order/`, `second_order/`, `memory_efficient/`, `lr_free/`, `sharpness_aware/`, `distributed/`). Class names exactly as the authors named them.
4. **Docstring contract.** A one-line description, the update rule in standard notation (`.. math::` block matching the paper; when the official implementation deviates from the paper, document what the code does), the paper reference with venue and link, and a short note for any usage caveat.
5. **Registration.** Export from the subpackage `__init__.py`, import and register in `pytorch/zij/__init__.py`.
6. **Tests.** The parametrized harness in `pytorch/tests/` picks up every registered optimizer. Add a recipe in `pytorch/tests/recipes.py` if the defaults do not fit (learning rate, closure protocol, parameter shape). All tests must pass.
7. **Canon row.** Add or update the method's row in the matching `canon/*.md` page. Keep the provenance header accurate so `THIRD_PARTY_NOTICES.md` stays consistent with the vendored sources.

## Adding a Canon entry

Canon rows are facts, not link dumps. Before submitting: the paper link must resolve and the title must match the published title character for character; the venue is the peer-reviewed venue only when the acceptance is confirmable, otherwise `arXiv <year>`; the code link is labeled `official` only when released by the paper's authors; the category must match the method. Methods with no public implementation belong in the Canon too, with an em dash in the code column.

## Style

American English. Comments only where the code cannot speak for itself. `ruff check pytorch` must pass (the vendored `zij/core/` mirror is excluded from lint by design — do not edit it except through a torch version sync).
