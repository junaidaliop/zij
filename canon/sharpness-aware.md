# Sharpness-Aware Optimizers

Sharpness-aware methods seek parameters that lie in neighborhoods with uniformly low loss rather than at isolated minima, which tends to improve generalization. Introduced by SAM (Foret et al., ICLR 2021), these methods wrap a base optimizer such as SGD or AdamW and add a gradient ascent perturbation step before the descent update. Later work makes the perturbation scale-invariant, closes the surrogate gap, reweights the sharpness term, amortizes the extra forward-backward cost, or extends the idea to second-order optimization.

| Optimizer | Venue | Paper | Code | `zij` |
|---|---|---|---|---|
| [SAM](math/sam.md) | ICLR 2021 | [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) | [community](https://github.com/davda54/sam) | `SAM` |
| [ASAM](math/asam.md) | ICML 2021 | [ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks](https://arxiv.org/abs/2102.11600) | [community](https://github.com/davda54/sam) | `ASAM` |
| [ESAM](math/esam.md) | ICLR 2022 | [Efficient Sharpness-aware Minimization for Improved Training of Neural Networks](https://arxiv.org/abs/2110.03141) | — | — |
| [GSAM](math/gsam.md) | ICLR 2022 | [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/abs/2203.08065) | [official](https://github.com/google-research/big_vision/tree/main/big_vision/trainers/proj/gsam) | `GSAM` |
| [LookSAM](math/looksam.md) | CVPR 2022 | [Towards Efficient and Scalable Sharpness-Aware Minimization](https://arxiv.org/abs/2203.02714) | [community](https://github.com/kozistr/pytorch_optimizer) | `LookSAM` |
| [AE-SAM](math/aesam.md) | ICLR 2023 | [An Adaptive Policy to Employ Sharpness-Aware Minimization](https://arxiv.org/abs/2304.14647) | — | — |
| [bSAM](math/bsam.md) | ICLR 2023 | [SAM as an Optimal Relaxation of Bayes](https://arxiv.org/abs/2210.01620) | [official](https://github.com/team-approx-bayes/bayesian-sam) | — |
| [GAM](math/gam.md) | CVPR 2023 | [Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization](https://arxiv.org/abs/2303.03108) | — | — |
| [WSAM](math/wsam.md) | KDD 2023 | [Sharpness-Aware Minimization Revisited: Weighted Sharpness as a Regularization Term](https://arxiv.org/abs/2305.15817) | [official](https://github.com/intelligent-machine-learning/atorch/tree/main/atorch/optimizers) | `WSAM` |
| [AdaSAM](math/adasam.md) | Neural Networks 2024 | [AdaSAM: Boosting Sharpness-Aware Minimization with Adaptive Learning Rate and Momentum for Training Deep Neural Networks](https://arxiv.org/abs/2303.00565) | — | — |
| [F-SAM](math/fsam.md) | CVPR 2024 | [Friendly Sharpness-Aware Minimization](https://arxiv.org/abs/2403.12350) | [official](https://github.com/nblt/F-SAM) | — |
| [FGSAM](math/fgsam.md) | NeurIPS 2024 | [Fast Graph Sharpness-Aware Minimization for Enhancing and Accelerating Few-Shot Node Classification](https://arxiv.org/abs/2410.16845) | — | — |
| [Lookbehind-SAM](math/lookbehindsam.md) | ICML 2024 | [Lookbehind-SAM: k steps back, 1 step forward](https://arxiv.org/abs/2307.16704) | — | — |
| [MSAM](math/msam.md) | arXiv 2024 | [Momentum-SAM: Sharpness Aware Minimization without Computational Overhead](https://arxiv.org/abs/2401.12033) | [official](https://github.com/MarlonBecker/MSAM) | — |
| [SAMPa](math/sampa.md) | NeurIPS 2024 | [SAMPa: Sharpness-aware Minimization Parallelized](https://arxiv.org/abs/2410.10683) | — | — |
| [AsyncSAM](math/asyncsam.md) | arXiv 2025 | [Asynchronous Sharpness-Aware Minimization For Fast and Accurate Deep Learning](https://arxiv.org/abs/2503.11147) | — | — |
| [GCSAM](math/gcsam.md) | arXiv 2025 | [GCSAM: Gradient Centralized Sharpness Aware Minimization](https://arxiv.org/abs/2501.11584) | [official](https://github.com/mhassann22/GCSAM) | — |
| [LightSAM](math/lightsam.md) | arXiv 2025 | [LightSAM: Parameter-Agnostic Sharpness-Aware Minimization](https://arxiv.org/abs/2505.24399) | — | — |
| [SASSHA](math/sassha2.md) | ICML 2025 | [SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](https://arxiv.org/abs/2502.18153) | [official](https://github.com/LOG-postech/Sassha) | — |
| [SSAM](math/ssam.md) | JMLR 2025 | [Stabilizing Sharpness-aware Minimization Through A Simple Renormalization Strategy](https://arxiv.org/abs/2401.07250) | — | — |
| [SAM-Polyak (Adaptive SAM with Polyak step size)](math/sampolyakadaptivesamwithpolyakstepsize.md) | ICML 2026 | [Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](https://arxiv.org/abs/2606.01827) | [official](https://github.com/dimitris-oik/sam_sps) | — |
| [X-SAM](math/xsam.md) | arXiv 2026 | [X-SAM: Boosting Sharpness-Aware Minimization with Dominant-Eigenvector Gradient Correction](https://arxiv.org/abs/2601.10251) | — | — |
| [M-SAM (Modality-Aware SAM)](math/msammodalityawaresam.md) | NeurIPS 2025 | [Modality-Aware SAM: Sharpness-Aware-Minimization Driven Gradient Modulation for Harmonized Multimodal Learning](https://arxiv.org/abs/2510.24919) | — | — |
| [ZSharp (SAM with Z-Score Gradient Filtering)](math/zsharpsamwithzscoregradientfiltering.md) | NeurIPS 2025 OPT Workshop (also accepted to ICASSP 2026) | [Sharpness-Aware Minimization with Z-Score Gradient Filtering](https://arxiv.org/abs/2505.02369) | [official](https://github.com/YUNBLAK/Sharpness-Aware-Minimization-with-Z-Score-Gradient-Filtering) | — |
| [Focal-SAM](math/focalsam.md) | ICML 2025 | [Focal-SAM: Focal Sharpness-Aware Minimization for Long-Tailed Classification](https://arxiv.org/abs/2505.01660) | [official](https://github.com/scongl/Focal-SAM) | — |
| [Functional SAM](math/functionalsam.md) | ICML 2025 | [Avoiding spurious sharpness minimization broadens applicability of SAM](https://arxiv.org/abs/2502.02407) | — | — |
| [FedGMT](math/fedgmt.md) | ICML 2025 | [One Arrow, Two Hawks: Sharpness-aware Minimization for Federated Learning via Global Model Trajectory](https://openreview.net/forum?id=80mK2Mqaph) | [official](https://github.com/harrylee999/FL-SAM) | — |
| [LE-SAM](math/lesam.md) | ICML 2026 | [Fix the Loss, Not the Radius: Rethinking the Adversarial Perturbation of Sharpness-Aware Minimization](https://arxiv.org/abs/2605.10183) | — | — |
