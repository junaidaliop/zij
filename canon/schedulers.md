# Learning Rate Schedulers

`zij.core.lr_scheduler` vendors the PyTorch core learning rate schedulers under their original class names. The first table lists every vendored class, including the `LRScheduler` base class, with the published work it derives from where one exists. The second table covers notable schedules from the literature that zij does not yet implement.

## In zij

| Scheduler | Origin |
|---|---|
| `ChainedScheduler` | — |
| `ConstantLR` | — |
| `CosineAnnealingLR` | Loshchilov & Hutter ICLR 2017 (SGDR) |
| `CosineAnnealingWarmRestarts` | Loshchilov & Hutter ICLR 2017 (SGDR) |
| `CyclicLR` | Smith WACV 2017 (cyclical learning rates) |
| `ExponentialLR` | — |
| `LambdaLR` | — |
| `LinearLR` | — |
| `LRScheduler` | — |
| `MultiplicativeLR` | — |
| `MultiStepLR` | — |
| `OneCycleLR` | Smith & Topin 2019 (super-convergence) |
| `PolynomialLR` | — |
| `ReduceLROnPlateau` | — |
| `SequentialLR` | — |
| `StepLR` | — |

## Notable schedules elsewhere

| Scheduler | Venue | Paper | Code | `zij` |
|---|---|---|---|---|
| [Inverse square root](math/inversesquareroot.md) | NeurIPS 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [official](https://github.com/tensorflow/tensor2tensor) | — |
| [AdaS](math/adas.md) | arXiv 2020 | [AdaS: Adaptive Scheduling of Stochastic Gradients](https://arxiv.org/abs/2006.06587) | [official](https://github.com/mahdihosseini/AdaS) | — |
| [Untuned Warmup](math/untunedwarmup.md) | AAAI 2021 | [On the adequacy of untuned warmup for adaptive optimization](https://arxiv.org/abs/1910.04209) | — | — |
| [AutoDrop](math/autodrop.md) | UAI 2024 | [AutoDrop: Training Deep Learning Models with Automatic Learning Rate Drop](https://arxiv.org/abs/2111.15317) | — | — |
| [Schedule-Free](math/sgdschedulefree.md) | NeurIPS 2024 | [The Road Less Scheduled](https://arxiv.org/abs/2405.15682) | [official](https://github.com/facebookresearch/schedule_free) | `SGDScheduleFree`, `AdamWScheduleFree`, `RAdamScheduleFree`, `ScheduleFreeWrapper` |
| [WSD (Warmup-Stable-Decay)](math/wsdwarmupstabledecay.md) | COLM 2024 | [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/abs/2404.06395) | [official](https://github.com/OpenBMB/MiniCPM) | — |
| [GreedyLR](math/greedylr.md) | arXiv 2025 | [Dynamic Learning Rate Scheduling based on Loss Changes Leads to Faster Convergence](https://arxiv.org/abs/2512.14527) | — | — |
| [Refined SF-AdamW](math/refinedsfadamw.md) | NeurIPS 2025 | [Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](https://arxiv.org/abs/2507.09846) | — | — |
| [SF-NorMuon](math/sfnormuon.md) | arXiv 2026 | [Anytime Training with Schedule-Free Spectral Optimization](https://arxiv.org/abs/2605.23061) | — | — |
| [WSM](math/wsm.md) | ICLR 2026 | [WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](https://arxiv.org/abs/2507.17634) | — | — |
| [Power Decay / Warmup-Stable-Decay (WSD)](math/powerdecaywarmupstabledecaywsd.md) | arXiv 2026 | [Optimal Learning-Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay](https://arxiv.org/abs/2602.06797) | — | — |
| [Anytime (Horizon-Free WA schedule)](math/anytimehorizonfreewaschedule.md) | arXiv 2026 | [Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging](https://arxiv.org/abs/2602.03702) | — | — |

Schedule-Free is not a schedule on top of an optimizer but a replacement for scheduling, achieved through online iterate averaging inside the optimizer; see the [learning-rate-free optimizers](lr-free.md).

Weight averaging is available separately in `zij.core.swa_utils`, which provides stochastic weight averaging and exponential moving average utilities (`AveragedModel`, `SWALR`, `update_bn`, and the SWA/EMA averaging functions), following [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) (Izmailov et al., UAI 2018).
