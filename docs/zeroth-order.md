# Zeroth-Order Optimizers

Zeroth-order (gradient-free) methods train models using only function evaluations, estimating gradients from randomized perturbations of the parameters instead of backpropagation. Because they need no backward pass or activation storage, they run at roughly inference-level memory, which has made them a practical option for fine-tuning large language models on constrained hardware. The lineage runs from SPSA in classical stochastic approximation to recent variance-reduced and low-rank variants built on MeZO.

| Optimizer | Venue | Paper | Code | `zij` |
| --- | --- | --- | --- | --- |
| [SPSA](math/spsa2.md) | IEEE Transactions on Automatic Control 1992 | [Multivariate stochastic approximation using a simultaneous perturbation gradient approximation](https://doi.org/10.1109/9.119632) | [official](https://www.jhuapl.edu/spsa/) | — |
| [Evolution Strategies](math/evolutionstrategies.md) | arXiv 2017 | [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) | [official](https://github.com/openai/evolution-strategies-starter) | — |
| [ZO-AdaMM](math/zoadamm.md) | NeurIPS 2019 | [ZO-AdaMM: Zeroth-Order Adaptive Momentum Method for Black-Box Optimization](https://arxiv.org/abs/1910.06513) | [official](https://github.com/KaidiXu/ZO-AdaMM) | — |
| [MeZO](math/mezo2.md) | NeurIPS 2023 | [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333) | [official](https://github.com/princeton-nlp/MeZO) | — |
| [DeepZero](math/deepzero.md) | ICLR 2024 | [DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training](https://arxiv.org/abs/2310.02025) | [official](https://github.com/OPTML-Group/DeepZero) | — |
| [LeZO](math/lezo.md) | arXiv 2024 | [Simultaneous Computation and Memory Efficient Zeroth-Order Optimizer for Fine-Tuning Large Language Models](https://arxiv.org/abs/2410.09823) | [official](https://github.com/WangFei-2019/LeZO) | — |
| [MeZO-SVRG](math/mezosvrg.md) | ICML 2024 | [Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models](https://arxiv.org/abs/2404.08080) | [official](https://github.com/amazon-science/mezo_svrg) | — |
| [ZO-AdaMU](math/zoadamu.md) | AAAI 2024 | [ZO-AdaMU Optimizer: Adapting Perturbation by the Momentum and Uncertainty in Zeroth-order Optimization](https://arxiv.org/abs/2312.15184) | [official](https://github.com/MathIsAll/ZO-AdaMU) | — |
| [ZoPro](math/zopro.md) | CDC 2024 | [A Zeroth-Order Proximal Algorithm for Consensus Optimization](https://arxiv.org/abs/2406.09816) | — | — |
| [Addax](math/addax2.md) | ICLR 2025 | [Addax: Utilizing Zeroth-Order Gradients to Improve Memory Efficiency and Performance of SGD for Fine-Tuning Language Models](https://arxiv.org/abs/2410.06441) | [official](https://github.com/optimization-for-data-driven-science/Addax) | — |
| [DiZO](math/dizo.md) | NeurIPS 2025 | [Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning](https://arxiv.org/abs/2502.03304) | [official](https://github.com/Skilteee/DiZO) | — |
| [ElasticZO](math/elasticzo.md) | arXiv 2025 | [ElasticZO: A Memory-Efficient On-Device Learning with Combined Zeroth- and First-Order Optimization](https://arxiv.org/abs/2501.04287) | — | — |
| [HELENE](math/helene.md) | EMNLP 2025 | [HELENE: Hessian Layer-wise Clipping and Gradient Annealing for Accelerating Fine-tuning LLM with Zeroth-order Optimization](https://arxiv.org/abs/2411.10696) | — | — |
| [KerZOO](math/kerzoo.md) | arXiv 2025 | [KerZOO: Kernel Function Informed Zeroth-Order Optimization for Accurate and Accelerated LLM Fine-Tuning](https://arxiv.org/abs/2505.18886) | — | — |
| [LORENZA](math/lorenza2.md) | arXiv 2025 | [LORENZA: Enhancing Generalization in Low-Rank Gradient LLM Training via Efficient Zeroth-Order Adaptive SAM](https://arxiv.org/abs/2502.19571) | — | — |
| [LOZO](math/lozo.md) | ICLR 2025 | [Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures](https://arxiv.org/abs/2410.07698) | [official](https://github.com/optsuite/LOZO) | — |
| [MaZO](math/mazo.md) | arXiv 2025 | [MaZO: Masked Zeroth-Order Optimization for Multi-Task Fine-Tuning of Large Language Models](https://arxiv.org/abs/2502.11513) | — | — |
| [QuZO](math/quzo.md) | EMNLP 2025 | [QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models](https://arxiv.org/abs/2502.12346) | [official](https://github.com/lloo099/QuZO) | — |
| [R-AdaZO](math/radazo.md) | ICML 2025 | [Refining Adaptive Zeroth-Order Optimization at Ease](https://arxiv.org/abs/2502.01014) | [official](https://github.com/shuyao95/R-AdaZO) | — |
| [Sparse MeZO](math/sparsemezo.md) | NeurIPS 2025 | [Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](https://arxiv.org/abs/2402.15751) | [official](https://github.com/NUS-HPC-AI-Lab/SparseMeZO) | — |
| [SubZero](math/subzero.md) | ICCV 2025 | [Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](https://arxiv.org/abs/2410.08989) | [official](https://github.com/zimingyy/SubZero) | — |
| [TeZO](math/tezo.md) | arXiv 2025 | [TeZO: Empowering the Low-Rankness on the Temporal Dimension in the Zeroth-Order Optimization for Fine-tuning LLMs](https://arxiv.org/abs/2501.19057) | — | — |
| [VAMO](math/vamo.md) | arXiv 2025 | [VAMO: Efficient Zeroth-Order Variance Reduction for SGD with Faster Convergence](https://arxiv.org/abs/2505.13954) | — | — |
| [VR-SZD](math/vrszd.md) | arXiv 2025 | [A Structured Proximal Stochastic Variance Reduced Zeroth-order Algorithm](https://arxiv.org/abs/2506.23758) | [official](https://github.com/MarcoRando/vr_szd) | — |
| [ZO-SAH](math/zosah.md) | arXiv 2025 | [Subspace-based Approximate Hessian Method for Zeroth-Order Optimization](https://arxiv.org/abs/2507.06125) | — | — |
| [ZO2](math/zo2.md) | COLM 2025 | [ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory](https://arxiv.org/abs/2503.12668) | [official](https://github.com/liangyuwang/zo2) | — |
| [ZOQO](math/zoqo.md) | ICASSP 2025 | [ZOQO: Zero-Order Quantized Optimization](https://arxiv.org/abs/2501.06736) | — | — |
| [AdaMeZO](math/adamezo.md) | arXiv 2026 | [AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments](https://arxiv.org/abs/2605.00650) | [official](https://github.com/shawnnn3di/AdaMeZO) | — |
| [FZOO](math/fzoo.md) | ICLR 2026 | [FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed](https://arxiv.org/abs/2506.09034) | [official](https://github.com/DKmiyan/FZOO) | — |
| [MEAZO](math/meazo.md) | arXiv 2026 | [On Adaptivity in Zeroth-Order Optimization](https://arxiv.org/abs/2605.03869) | — | — |
| [QZO](math/qzo.md) | ICLR 2026 | [Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](https://arxiv.org/abs/2505.13430) | [official](https://github.com/maifoundations/QZO) | — |
| [GRZO](math/grzo.md) | arXiv 2026 | [GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning](https://arxiv.org/abs/2606.02857) | — | — |
| [AGZO](math/agzo.md) | ICML 2026 | [AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](https://arxiv.org/abs/2601.17261) | — | — |
| [ZO-MOPI](math/zomopi.md) | arXiv 2026 | [Accelerating Zeroth-Order Spectral Optimization with Partial Orthogonalization from Power Iteration](https://arxiv.org/abs/2605.09034) | [official](https://github.com/MOFA-LAB/ZO-MOPI) | — |
| [ZO-Muon](math/zomuon.md) | arXiv 2026 | [Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization](https://arxiv.org/abs/2602.17155) | [official](https://github.com/OPTML-Group/ZO-Muon) | — |
| [RLR (Recursive Likelihood Ratio)](math/rlrrecursivelikelihoodratio.md) | ICLR 2026 | [Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer](https://arxiv.org/abs/2502.00639) | [official](https://github.com/RTkenny/RLR-Optimizer) | — |
| [ZO Fine-tuner](math/zofinetuner.md) | arXiv (accepted to ICML 2026) 2025 | [Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](https://arxiv.org/abs/2510.00419) | [official](https://github.com/ASTRAL-Group/ZO_Fine_tuner) | — |
