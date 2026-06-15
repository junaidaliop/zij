# zij

**A comprehensive library of deep learning optimizers.**

`zij` packages a large, growing collection of PyTorch optimizers — from the
`torch.optim` classics to current research methods (Muon, SOAP, Sophia, Prodigy,
Schedule-Free, GaLore, Lion, AdEMAMix, and many more) — behind one consistent
interface, each vendored from its original source with attribution and covered
by convergence tests.

```bash
pip install zij
```

```python
import zij

opt = zij.AdamW(model.parameters(), lr=3e-4)
opt = zij.Prodigy(model.parameters())              # learning-rate-free
opt = zij.GaLoreAdamW(model.parameters(), lr=1e-2)  # memory-efficient
opt = zij.SAM(model.parameters(), base_optimizer=zij.SGD, lr=0.1)  # sharpness-aware

zij.list_optimizers("adam*")              # discover by name
opt = zij.load_optimizer("soap")(model.parameters(), lr=2e-2)
```

Both `zij.AdamW` and `zij.optim.AdamW` work; the latter mirrors `torch.optim`.

When you use an optimizer, please cite **both** its original paper and `zij` —
the papers are linked in every class docstring.

The full **Canon of 573 optimization methods** (including paper-only methods
and the memory-efficient, fractional-order, and quantum families) lives in the
project repository:

**https://github.com/junaidaliop/zij**

Apache-2.0. Vendored components retain their original licenses; see
`THIRD_PARTY_NOTICES.md` in the repository.
