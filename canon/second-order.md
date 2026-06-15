# Second-Order and Orthogonalized Optimizers

Second-order and orthogonalized optimizers exploit curvature information or the matrix structure of gradients rather than purely elementwise first-order statistics. This group spans quasi-Newton and Hessian-diagonal methods (L-BFGS, AdaHessian, Sophia), full-matrix and Kronecker-factored preconditioning (PSGD, Shampoo, SOAP), and orthogonalized-update methods in the Muon family. Venues reflect peer-reviewed acceptance where applicable; otherwise the arXiv year is listed.

| Optimizer | Venue | Paper | Code | `zij` |
| --- | --- | --- | --- | --- |
| [Gauss-Newton Method](math/gaussnewtonmethod.md) | Biometrika 1974 | [Quasi-likelihood functions, generalized linear models, and the Gauss-Newton method](https://doi.org/10.1093/biomet/61.3.439) | — | — |
| [Newton's Method](math/newtonsmethod.md) | ANL Technical Report 1982 | [Newton's method (ANL-82-8)](https://www.osti.gov/biblio/5326201) | — | — |
| [L-BFGS](math/lbfgs.md) | Mathematical Programming 1989 | [On the limited memory BFGS method for large scale optimization](https://doi.org/10.1007/BF01589116) | [official](https://users.iems.northwestern.edu/~nocedal/lbfgs.html) | `LBFGS` |
| [Natural Gradient](math/naturalgradient.md) | Neural Computation 1998 | [Natural Gradient Works Efficiently in Learning](https://doi.org/10.1162/089976698300017746) | — | — |
| [K-FAC](math/kfac.md) | ICML 2015 | [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671) | — | — |
| [PSGD](math/psgd.md) | IEEE TNNLS 2018 | [Preconditioned Stochastic Gradient Descent](https://arxiv.org/abs/1512.04202) | [official](https://github.com/lixilinx/psgd_torch) | — |
| [Shampoo](math/shampoo.md) | ICML 2018 | [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568) | [official](https://github.com/google-research/google-research/tree/master/scalable_shampoo) | `Shampoo` |
| [AdaHessian](math/adahessian.md) | AAAI 2021 | [ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning](https://arxiv.org/abs/2006.00719) | [official](https://github.com/amirgholami/adahessian) | `Adahessian` |
| [Apollo](math/apollo-v.md) | arXiv 2020 | [Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization](https://arxiv.org/abs/2009.13586) | [official](https://github.com/XuezheMax/apollo) | — |
| [K-BFGS / K-BFGS(L)](math/kbfgskbfgsl.md) | NeurIPS 2020 | [Practical Quasi-Newton Methods for Training Deep Neural Networks](https://arxiv.org/abs/2006.08877) | — | — |
| [SGN](math/sgn.md) | arXiv 2020 | [On the Promise of the Stochastic Generalized Gauss-Newton Method for Training DNNs](https://arxiv.org/abs/2006.02409) | — | — |
| [SpiderSQN](math/spidersqn.md) | IEEE TNNLS 2022 | [Faster Stochastic Quasi-Newton Methods](https://arxiv.org/abs/2004.06479) | — | — |
| [TKFAC](math/tkfac.md) | AAAI 2021 | [A Trace-restricted Kronecker-Factored Approximation to Natural Gradient](https://arxiv.org/abs/2011.10741) | — | — |
| [SGDHess](math/sgdhess.md) | NeurIPS 2022 | [Better SGD using Second-order Momentum](https://arxiv.org/abs/2103.03265) | — | — |
| [SketchySGD](math/sketchysgd.md) | SIMODS 2024 | [SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates](https://arxiv.org/abs/2211.08597) | [official](https://github.com/udellgroup/SketchySGD) | — |
| [Distributed Shampoo](math/distributedshampoo2.md) | arXiv 2023 | [A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale](https://arxiv.org/abs/2309.06497) | [official](https://github.com/facebookresearch/optimizers) | — |
| [mL-BFGS](math/mlbfgs.md) | TMLR 2023 | [mL-BFGS: A Momentum-based L-BFGS for Distributed Large-Scale Neural Network Optimization](https://arxiv.org/abs/2307.13744) | — | — |
| [Sophia](math/sophiag.md) | ICLR 2024 | [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342) | [official](https://github.com/Liuhong99/Sophia) | `SophiaG` |
| [AdaFisher](math/adafisher.md) | ICLR 2025 | [AdaFisher: Adaptive Second Order Optimization via Fisher Information](https://arxiv.org/abs/2405.16397) | [official](https://github.com/AtlasAnalyticsLab/AdaFisher) | — |
| [CRNAS](math/crnas.md) | arXiv 2024 | [Novel Optimization Techniques for Parameter Estimation](https://arxiv.org/abs/2407.04235) | — | — |
| [HesScale](math/hesscale.md) | ICML 2024 | [Revisiting Scalable Hessian Diagonal Approximations for Applications in Reinforcement Learning](https://arxiv.org/abs/2406.03276) | [official](https://github.com/mohmdelsayed/HesScale) | — |
| [Muon](math/muon.md) | Blog post 2024 | [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) | [official](https://github.com/KellerJordan/Muon) | `Muon` |
| [NysAct](math/nysact.md) | IEEE BigData 2024 | [NysAct: A Scalable Preconditioned Gradient Descent using Nystrom Approximation](https://arxiv.org/abs/2506.08360) | — | — |
| [OptiQ](math/optiq.md) | arXiv 2024 | [Second-Order Optimization via Quiescence](https://arxiv.org/abs/2410.08033) | — | — |
| [Q-Newton](math/qnewton.md) | arXiv 2024 | [Q-Newton: Hybrid Quantum-Classical Scheduling for Accelerating Neural Network Training with Newton's Gradient Descent](https://arxiv.org/abs/2405.00252) | [official](https://github.com/UNITES-Lab/q-newton) | — |
| [SOAA](math/soaa.md) | arXiv 2024 | [Efficient Second-Order Neural Network Optimization via Adaptive Trust Region Methods](https://arxiv.org/abs/2410.02293) | — | — |
| [SOAP](math/soap.md) | ICLR 2025 | [SOAP: Improving and Stabilizing Shampoo using Adam for Language Modeling](https://arxiv.org/abs/2409.11321) | [official](https://github.com/nikhilvyas/SOAP) | `SOAP` |
| [AdaDiag](math/adadiag.md) | arXiv 2025 | [Improving Adaptive Moment Optimization via Preconditioner Diagonalization](https://arxiv.org/abs/2502.07488) | — | — |
| [ADAGB2](math/adagb2.md) | arXiv 2025 | [Fast Stochastic Second-Order Adagrad for Nonconvex Bound-Constrained Optimization](https://arxiv.org/abs/2505.06374) | — | — |
| [AdaGO](math/adago.md) | arXiv 2025 | [AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal Updates](https://arxiv.org/abs/2509.02981) | — | — |
| [AdaMuon](math/adamuon.md) | arXiv 2025 | [AdaMuon: Adaptive Muon Optimizer](https://arxiv.org/abs/2507.11005) | [official](https://github.com/Chongjie-Si/AdaMuon) | `AdaMuon` |
| [ASGO](math/asgo.md) | NeurIPS 2025 | [ASGO: Adaptive Structured Gradient Optimization](https://arxiv.org/abs/2503.20762) | [official](https://github.com/infinity-stars/ASGO) | — |
| [AuON](math/auon.md) | arXiv 2025 | [AuON: A Linear-time Alternative to Orthogonal Momentum Updates](https://arxiv.org/abs/2509.24320) | [official](https://github.com/ryyzn9/AuON) | — |
| [COSMOS](math/cosmos.md) | arXiv 2025 | [COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs](https://arxiv.org/abs/2502.17410) | [official](https://github.com/lliu606/COSMOS) | — |
| [FUSE](math/fuse.md) | IEEE CAI 2025 | [FUSE: First-Order and Second-Order Unified SynthEsis in Stochastic Optimization](https://arxiv.org/abs/2503.04204) | — | — |
| [Hessian-aware Scaling](math/hessianawarescaling.md) | arXiv 2025 | [First-ish Order Methods: Hessian-aware Scalings of Gradient Descent](https://arxiv.org/abs/2502.03701) | — | — |
| [MAC](math/mac.md) | IEEE ICDM 2025 | [MAC: An Efficient Gradient Preconditioning using Mean Activation Approximated Curvature](https://arxiv.org/abs/2506.08464) | — | — |
| [MuonClip](math/muonclip.md) | arXiv 2025 | [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) | [community](https://github.com/AkulDatta/muonclip) | — |
| [NorMuon](math/normuon.md) | ICML 2026 | [NorMuon: Making Muon more efficient and scalable](https://arxiv.org/abs/2510.05491) | [official](https://github.com/zichongli5/NorMuon) | `NorMuon` |
| [OCAR](math/ocar.md) | ICML 2025 | [Online Curvature-Aware Replay: Leveraging 2nd Order Information for Online Continual Learning](https://arxiv.org/abs/2502.01866) | — | — |
| [PolarGrad](math/polargrad.md) | arXiv 2025 | [PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective](https://arxiv.org/abs/2505.21799) | [official](https://github.com/timlautk/polargrad) | `PolarGrad` |
| [ROOT](math/root.md) | arXiv 2025 | [ROOT: Robust Orthogonalized Optimizer for Neural Network Training](https://arxiv.org/abs/2511.20626) | [official](https://github.com/huawei-noah/noah-research/tree/master/ROOT) | — |
| [S-BFGS](math/sbfgs.md) | arXiv 2025 | [Efficient Stochastic BFGS methods Inspired by Bayesian Principles](https://arxiv.org/abs/2507.07729) | — | — |
| [SASSHA](math/sassha.md) | ICML 2025 | [SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation](https://arxiv.org/abs/2502.18153) | [official](https://github.com/LOG-postech/Sassha) | — |
| [Scion](math/scion.md) | ICML 2025 | [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529) | [official](https://github.com/LIONS-EPFL/scion) | `Scion` |
| [SPlus](math/splus.md) | arXiv 2025 | [A Stable Whitening Optimizer for Efficient Neural Network Training](https://arxiv.org/abs/2506.07254) | [official](https://github.com/kvfrans/splus) | `SPlus` |
| [Muon^2](math/muon2.md) | arXiv 2026 | [Muon^2: Boosting Muon via Adaptive Second-Moment Preconditioning](https://arxiv.org/abs/2604.09967) | — | — |
| [Nora](math/nora.md) | arXiv 2026 | [Nora: Normalized Orthogonal Row Alignment for Scalable Matrix Optimizer](https://arxiv.org/abs/2605.03769) | — | — |
| [Pion](math/pion2.md) | arXiv 2026 | [Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR](https://arxiv.org/abs/2605.19282) | — | — |
| [Spectral Sphere Optimizer (SSO)](math/spectralsphereoptimizersso.md) | arXiv 2026 | [Controlled LLM Training on Spectral Sphere](https://arxiv.org/abs/2601.08393) | [official](https://github.com/Unakar/Spectral-Sphere-Optimizer) | — |
| [LoRA-Muon](math/loramuon.md) | arXiv 2026 | [LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold](https://arxiv.org/abs/2606.12921) | — | — |
| [FOAM](math/foam2.md) | arXiv 2026 | [FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo](https://arxiv.org/abs/2606.02365) | — | — |
| [Mousse](math/mousse.md) | arXiv 2026 | [Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning](https://arxiv.org/abs/2603.09697) | [official](https://github.com/Anti-Entrophic/Mousse) | — |
| [FISMO](math/fismo.md) | arXiv 2026 | [FISMO: Fisher-Structured Momentum-Orthogonalized Optimizer](https://arxiv.org/abs/2601.21750) | — | — |
| [DyKAF](math/dykaf.md) | arXiv 2025 | [DyKAF: Dynamical Kronecker Approximation of the Fisher Information Matrix for Gradient Preconditioning](https://arxiv.org/abs/2511.06477) | — | — |
| [Double Preconditioning (DoPr)](math/doublepreconditioningdopr.md) | arXiv 2026 | [Double Preconditioning (DoPr): Optimization for Test-Time Performance, not Validation Loss](https://arxiv.org/abs/2606.06418) | — | — |
| [AdaCubic](math/adacubic.md) | TMLR 2026 | [AdaCubic: An Adaptive Cubic Regularization Optimizer for Deep Learning](https://arxiv.org/abs/2604.09437) | [official](https://github.com/iTsingalis/AdaCubic) | — |
| [IFNSO](math/ifnso.md) | arXiv 2026 | [IFNSO: Iteration-Free Newton-Schulz Orthogonalization](https://arxiv.org/abs/2602.02500) | [official](https://github.com/greekinRoma/Unified_Newton_Schulz_Orthogonalization) | — |
| [CAO](math/cao.md) | arXiv preprint 2025 | [CAO: Curvature-Adaptive Optimization via Periodic Low-Rank Hessian Sketching](https://arxiv.org/abs/2511.12548) | — | — |
| [Turbo-Muon](math/turbomuon.md) | arXiv 2025 | [Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning](https://arxiv.org/abs/2512.04632) | [official](https://github.com/thib-s/flash-newton-schulz) | — |
| [SR1 Cubic Quasi-Newton](math/sr1cubicquasinewton.md) | arXiv 2025 | [Symmetric Rank-One Quasi-Newton Methods for Deep Learning Using Cubic Regularization](https://arxiv.org/abs/2502.12298) | — | — |
| [KL-Shampoo](math/klshampoo.md) | ICLR 2026 | [Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](https://arxiv.org/abs/2509.03378) | [official](https://github.com/yorkerlin/KL-Methods) | — |
| [LLQR](math/llqr.md) | arXiv 2026 | [Layerwise LQR for Geometry-Aware Optimization of Deep Networks](https://arxiv.org/abs/2605.04230) | [official](https://github.com/SimonDufLab/LLQR) | — |
| [Freon / Kaon](math/freonkaon.md) | arXiv 2026 | [Muon is Not That Special: Random or Inverted Spectra Work Just as Well](https://arxiv.org/abs/2605.11181) | — | — |
| [Mano](math/mano.md) | arXiv 2026 | [Mano: Restriking Manifold Optimization for LLM Training](https://arxiv.org/abs/2601.23000) | [official](https://github.com/xie-lab-ml/Mano-Restriking-Manifold-Optimization-for-LLM-Training) | — |
| [Atlas](math/atlas.md) | OPT 2025: 17th Annual Workshop on Optimization for Machine Learning (co-located with NeurIPS 2025) | [Atlas – Rethinking Optimizer Design for Stability and Speed](https://opt-ml.org/papers/2025/paper6.pdf) | — | — |
