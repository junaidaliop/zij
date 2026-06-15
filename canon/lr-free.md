# Learning-Rate-Free Optimizers

Learning-rate-free (also called parameter-free or tuning-free) optimizers select their step size automatically during training instead of requiring a manually tuned learning rate. Most methods in this family estimate a quantity such as the distance from the initial point to the solution and set the effective step size from observed gradients, while others wrap an existing base optimizer and tune its global scale factor online. The goal is to match the performance of a well-tuned baseline without a learning-rate search.

| Optimizer | Venue | Paper | Code | `zij` |
|---|---|---|---|---|
| [AdGD](math/adgd.md) | ICML 2020 | [Adaptive Gradient Descent without Descent](https://arxiv.org/abs/1910.09529) | [official](https://github.com/ymalitsky/adaptive_GD) | — |
| [ALI-G](math/alig.md) | ICML 2020 | [Training Neural Networks for and by Interpolation](https://arxiv.org/abs/1906.05661) | [official](https://github.com/oval-group/ali-g) | — |
| [AdaBFE](math/adabfe.md) | arXiv 2022 | [BFE and AdaBFE: A New Approach in Learning Rate Automation for Stochastic Optimization](https://arxiv.org/abs/2207.02763) | — | — |
| [D-Adaptation](math/dadaptsgd.md) | ICML 2023 | [Learning-Rate-Free Learning by D-Adaptation](https://arxiv.org/abs/2301.07733) | [official](https://github.com/facebookresearch/dadaptation) | `DAdaptSGD`, `DAdaptAdam` |
| [DoG](math/dog.md) | ICML 2023 | [DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule](https://arxiv.org/abs/2302.12022) | [official](https://github.com/formll/dog) | `DoG`, `LDoG` |
| [Mechanic](math/mechanic.md) | NeurIPS 2023 | [Mechanic: A Learning Rate Tuner](https://arxiv.org/abs/2306.00144) | [official](https://github.com/optimizedlearning/mechanic) | `mechanize` |
| [Adam++](math/adam2.md) | arXiv 2024 | [Towards Simple and Provable Parameter-Free Adaptive Gradient Methods](https://arxiv.org/abs/2412.19444) | — | — |
| [MoMo](math/momo.md) | ICML 2024 | [MoMo: Momentum Models for Adaptive Learning Rates](https://arxiv.org/abs/2305.07583) | [official](https://github.com/fabian-sp/MoMo) | `Momo`, `MomoAdam` |
| [Prodigy](math/prodigy.md) | ICML 2024 | [Prodigy: An Expeditiously Adaptive Parameter-Free Learner](https://arxiv.org/abs/2306.06101) | [official](https://github.com/konstmish/prodigy) | `Prodigy` |
| [AdamG](math/adamg.md) | arXiv 2024 | [Towards Stability of Parameter-free Optimization](https://arxiv.org/abs/2405.04376) | [community](https://github.com/kozistr/pytorch_optimizer) | `AdamG` |
| [TRAC](math/trac.md) | NeurIPS 2024 | [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://arxiv.org/abs/2405.16642) | [official](https://github.com/ComputationalRobotics/TRAC) | `TRAC` |
| [Accelerated GRAAL](math/acceleratedgraal.md) | arXiv 2025 | [Nesterov Finds GRAAL: Optimal and Adaptive Gradient Method for Convex Optimization](https://arxiv.org/abs/2507.09823) | — | — |
| [AutoSGD](math/autosgd.md) | arXiv 2025 | [AutoSGD: Automatic Learning Rate Selection for Stochastic Gradient Descent](https://arxiv.org/abs/2505.21651) | — | — |
| [EAGLE](math/eagle.md) | arXiv 2025 | [eagle: early approximated gradient based learning rate estimator](https://arxiv.org/abs/2502.01036) | — | — |
| [ScheduleFree+](math/schedulefree.md) | arXiv 2026 | [ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models](https://arxiv.org/abs/2605.19095) | [official](https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamc_schedulefree_plus_paper.py) | — |
| [AMUSE](math/amuse.md) | arXiv 2026 | [AMUSE: Anytime Muon with Stable Gradient Evaluation](https://arxiv.org/abs/2605.22432) | — | — |
| [Adaptive Polyak Steps (SF-SGD / SF-Adam)](math/adaptivepolyakstepssfsgdsfadam.md) | arXiv 2025 | [Taking the Road Less Scheduled with Adaptive Polyak Steps](https://arxiv.org/abs/2511.07767) | — | — |
| [GGD (Geodesic Gradient Descent)](math/ggdgeodesicgradientdescent.md) | arXiv 2026 | [Geodesic Gradient Descent: A Generic and Learning-rate-free Optimizer on Objective Function-induced Manifolds](https://arxiv.org/abs/2603.06651) | — | — |
| [Accelerated Distance-adaptive Method (DoG-lineage)](math/accelerateddistanceadaptivemethoddoglineage.md) | NeurIPS 2025 | [Accelerated Distance-adaptive Methods for Hölder Smooth and Convex Optimization](https://arxiv.org/abs/2510.22135) | — | — |
| [GeN](math/gen.md) | ICLR 2025 | [Gradient descent with generalized Newton's method](https://arxiv.org/abs/2407.02772) | [official](https://github.com/ShiyunXu/gen-optim) | — |
| [DoWG](math/dowg.md) | NeurIPS 2023 | [DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method](https://arxiv.org/abs/2305.16284) | [official](https://github.com/rka97/dowg) | — |
| [U-DoG](math/udog.md) | COLT 2024 | [Accelerated Parameter-Free Stochastic Optimization](https://arxiv.org/abs/2404.00666) | — | — |
| [Sign-SGD via Parameter-Free Optimization](math/signsgdviaparameterfreeoptimization.md) | ICLR 2026 | [Sign-SGD via Parameter-Free Optimization](https://arxiv.org/abs/2506.03725) | — | — |
| [OptEMA](math/optema.md) | arXiv 2026 | [OptEMA: Adaptive Exponential Moving Average for Stochastic Optimization with Zero-Noise Optimality](https://arxiv.org/abs/2603.09923) | — | — |
