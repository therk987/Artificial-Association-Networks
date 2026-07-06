# All-In-One: Artificial Association Networks (AAN)

Code for **"All-In-One: Artificial Association Neural Networks"** ([arXiv:2111.00424](https://arxiv.org/abs/2111.00424)).

AAN expresses the information-delivery structure of existing neural networks
(MLP / RNN / RecNN / CNN / GNN) as a **neurotree** data structure and learns
any of them with a single association cell, so that datasets from multiple
domains (image, sound, text, graph, tree, tabular, source code) can be
trained simultaneously in one model.

## Repository layout

```
aan/                        # the package (single source of truth)
  config/                   #   device auto-detection (cuda -> mps -> cpu)
  constant/                 #   shared attribute keys
  data_structures/          #   NeuroNode, BatchNeuroTree, NeuroDataset/Dataloader
  models/
    encoder_cell/           #   RNN / GRU / GCN / GAT / EGCN cells
    encoders/               #   RAN series (shared cell) & LAN series (per-level cell)
    decoders/               #   DFD (deconvolution) counterparts
    feature_encoders/       #   multi-domain feature extraction connector (Psi)
      domains/              #   per-domain extractors: image/sound/text/tabular/graph/code
    feature_decoders/       #   multi-domain restoration connector (Psi^-1)
    subtasks/               #   node-level task connector (Phi_s)
    maintasks/              #   root-level task connector (Phi_m)
  baselines/                #   single-domain baselines (LeNet-5, M5, SCNN, RNN/GRU, MLP, GCN/GAT)
  datas/
    image/                  #   MNIST loader
    code/sort/              #   sorting-algorithm dataset + neurotree pkl (paper, Exp. 4)
    text/                   #   SST vocabulary pickle
tests/                      # CPU smoke tests
benchmarks/                 # DFC throughput benchmark (bench_dfc.py)
docs/
  img/                      # figures used by the notebook
  experiments/              # experiment-report PDFs from the paper runs
AAN-SimpleImageTraining.ipynb   # minimal MNIST walkthrough (run from repo root)
```

## Quickstart

```bash
pip install -r requirements.txt     # or: pip install -e .
python -m pytest tests/            # or: python tests/test_smoke.py
python benchmarks/bench_dfc.py     # DFC engine throughput
jupyter notebook AAN-SimpleImageTraining.ipynb
```

All commands run from the repository root; imports use the `aan.*` namespace
(e.g. `from aan.models.artificial_association_networks import ArtificialAssociationNeuralNetworks`).

Model versions: `ran` (RNN+GCN), `raan` (RNN+GAT), `gau` (GRU+GCN),
`gaau` (GRU+GAT), `egau` (GRU+edge-GCN), plus per-level `lan` / `laan`.

## Notes

- The legacy paper-experiment tree (`models/torch/`, `dataset/`, `newversion/`)
  has been consolidated into the `aan` package. The sort neurotree pickle was
  created with a legacy class path — load it via `aan/datas/code/load.py`,
  which remaps it transparently.
- Text extractors (`domains/text2vec.py`, `baselines/scnn.py`) expect a
  torchtext(-legacy) `TEXT` field object injected by the caller.
- Performance work (flattened, level-bucketed DFC) is tracked in
