# Checklist before making cellmap public

`CLAUDE.md` says the repository stays private until this list is done. Items marked **owner** are decisions only you can make; the rest are already done or have a command.

## Done

- [x] No credentials in the repository or its history. Scanned on 2026-11-18 for GitHub tokens, AWS keys, private keys and API keys across every commit: none found.
- [x] No downloaded data in git. `data/` is in `.gitignore`; every number is reproducible from the scripts.
- [x] CI (lint and tests) runs on every push and passes.
- [x] Every number in the README comes from a file under `results/`, written by a script under `scripts/`.
- [x] Dataset citations are in the README (section "Data and citations").
- [x] A GitHub Pages workflow exists for the site (`.github/workflows/pages.yml`), run by hand.

## Owner decisions

- [ ] **License.** Pick one before going public; without one, others cannot legally reuse the code. Two common choices:
  - **MIT:** short and permissive, the most common for portfolio projects.
  - **GPL-3.0:** worth considering because `harmonypy`, a dependency, is GPL-3.0. Whether a GPL dependency that is installed separately (not copied into this repo) constrains your own code's license is a legal question; this is not legal advice.
  - Add the file with GitHub's "Add file > Create new file > LICENSE" template picker, or tell Claude which one to add.
- [ ] **Merge to main.** All work is on `claude/clever-hamilton-wj375w`. Open a pull request into `main` and merge it, so the public default branch shows the finished project.
- [ ] **Delete the extra branches** after merging: `colab/m5-results` and `colab/m4-transformer` (their commits are already merged).
- [ ] **Make the repository public:** Settings > General > Danger zone > Change visibility.
- [ ] **Publish the site:** Settings > Pages > Source: GitHub Actions, then Actions > pages > Run workflow. The front page is the explainer; the atlas is `atlas.html`.
- [ ] **Revoke the Colab GitHub tokens** if you have not already (Settings > Developer settings > Personal access tokens).

## Worth knowing

- The pretrained-model embeddings (Census scVI, Geneformer, TranscriptFormer) are not stored in the repository, only scores computed from them. The fine-tuned Geneformer and TranscriptFormer have their own licenses; you only redistribute numbers, not the models.
- `results/m5/pbmc/model_seed0.pt` (4 MB) is a trained model of your own; fine to publish.
- The atlas and explainer contain per-cell coordinates and marker-gene levels derived from public CELLxGENE data (CC BY 4.0), which is why the citations matter.
