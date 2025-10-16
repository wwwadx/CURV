# [NeurIPS 2025] CURV: Coherent Uncertainty-Aware Reasoning in Vision-Language Models for X-Ray Report Generation

[]([https://nips.cc/](https://neurips.cc/virtual/2025/poster/120063))
[](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation for the paper **"CURV: Coherent Uncertainty-Aware Reasoning in Vision-Language Models for X-Ray Report Generation"**, accepted at NeurIPS 2025.

Model file at: https://modelscope.cn/models/wzaAAAAA/CURV

## 📝 Abstract

Vision-language models (VLMs) have shown promise in generating radiology reports, but they often lack the ability to explicitly model diagnostic uncertainty and the reasoning process used to reach clinical impressions. This limits their clinical accuracy and trustworthiness. We introduce **CURV**, a novel framework that integrates **uncertainty awareness** and **explicit reasoning** capabilities. 
Our results show that CURV generates clinically relevant reports with appropriate uncertainty and transparent reasoning, significantly outperforming previous methods.

-----

## 🏛️ Framework Overview

CURV employs a three-stage training pipeline that combines uncertainty-aware fine-tuning, reasoning initialization, and reinforcement learning to produce high-quality, trustworthy reports. The overall architecture is shown below.

<img width="4244" height="3772" alt="model_architecture (1)" src="https://github.com/user-attachments/assets/61e81f00-d378-4cff-9d87-6d7efd115397" />
*Figure: The CURV framework consists of three stages: (1) Uncertainty-aware SFT, (2) Reasoning Initialization SFT, and (3) Reinforcement Learning with Group Relative Policy Optimization (GRPO) to refine the final VLM.* 

-----

## ✨ Key Contributions

1.  **Uncertainty-Aware Generation**: We propose a framework that integrates a specialized fine-tuning strategy and an uncertainty-calibrated reward mechanism to model both **Structural Uncertainty** (in findings) and **Semantic Uncertainty** (in impressions).
2.  **Structured Reasoning with TRACE-CXR**: We introduce a structured reasoning framework that generates explicit "thinking" pathways. To enable this, we developed **TRACE-CXR**, a new dataset of 2,000 reports augmented with LLM-generated reasoning steps. This dataset will be made publicly available.
3.  **State-of-the-Art Performance**: Through extensive experiments, we show that CURV outperforms existing methods on both standard and clinical accuracy metrics. It also demonstrates strong generalization to out-of-distribution data.

-----




## 📜 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{wang2025curv,
  title={{CURV}: Coherent Uncertainty-Aware Reasoning in Vision-Language Models for {X-Ray} Report Generation},
  author={Ziao Wang and Sixing Yan and Kejing Yin and Xiaofeng Zhang and William K. Cheung},
  booktitle={Thirty-ninth Conference on Neural Information Processing Systems},
  year={2025},
  url={}
}
```
