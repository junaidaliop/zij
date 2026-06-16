# zij &nbsp;زِيج

A canon of deep learning optimization algorithms.

A *zij* (Arabic: زِيج, pronounced *"zeej"*) is an astronomical handbook from the
Islamic golden age: a set of tables and computational methods that astronomers
consulted instead of re-deriving the field from scratch. The best known is the
*Zīj al-Sindhind* of Muḥammad ibn Mūsā al-Khwārizmī, whose Latinized name became
the word *algorithm* and whose book *al-Jabr* gave us *algebra*. This project
takes the name in that spirit: one reference for the optimization algorithms of
machine learning — the equation, the paper, and runnable code in one place.

The Canon spans **740 methods across 11 categories**, with **100+** implemented as
a PyTorch library. Every optimizer's name links to its update-rule page.

## Installation

```bash
pip install zij
```

## Quick start

```python
import zij

# torch.optim, vendored at tag v2.12.0
opt = zij.AdamW(model.parameters(), lr=3e-4)

# research optimizers, same interface
opt = zij.Muon([p for p in model.parameters() if p.ndim == 2], lr=2e-2)
opt = zij.Prodigy(model.parameters())          # no learning rate to set

# look up by name
opt_cls = zij.load_optimizer("soap")
```

`zij.optim` mirrors `torch.optim`, so `zij.optim.AdamW` is the same class as
`zij.AdamW`.

!!! note
    A few families use a documented non-standard call protocol. Schedule-Free
    needs `opt.train()` and `opt.eval()`; the SAM family takes a closure or an
    explicit `first_step` / `second_step` pair; Adam-mini and LOMO are built from
    a model rather than a parameter list. Each optimizer's page notes which.

## Browse the Canon

Each category lists the canonical name, venue, paper, the best available
implementation, and the `zij` class where one exists. Every name links to its
update-rule page.

| Category | |
|---|---|
| [First-order](first-order.md) | SGD, the Adam family, sign-based and variance-reduced methods |
| [Memory-efficient](memory-efficient.md) | Adafactor, 8-bit and low-rank state methods |
| [Fractional-order](fractional.md) | optimizers built on fractional calculus |
| [Distributed](distributed.md) | communication-efficient and large-batch methods |
| [Second-order](second-order.md) | curvature, quasi-Newton, and preconditioned methods |
| [Zeroth-order](zeroth-order.md) | gradient-free and finite-difference methods |
| [Privacy-preserving](privacy-preserving.md) | differentially private optimization |
| [Sharpness-aware](sharpness-aware.md) | SAM and flat-minima methods |
| [Quantum](quantum.md) | optimizers for variational quantum circuits |
| [Learning-rate-free](lr-free.md) | parameter-free and adaptive step-size methods |
| [Schedulers](schedulers.md) | learning-rate schedules |
