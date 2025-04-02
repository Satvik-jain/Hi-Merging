# LLM-Merging


## Overview

Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse natural language processing (NLP) tasks. The release of open-source LLMs like LLaMA and Qwen has triggered the development of numerous fine-tuned models tailored for various tasks and languages. 

This project explores an important question: **Is it possible to combine specialized models to create a unified model with multi-task capabilities?**

We introduce **Hierarchical Iterative Merging (Hi-Merging)**, a training-free method for unifying different specialized LLMs into a single model. Specifically, Hi-Merging employs model-wise and layer-wise pruning and scaling, guided by contribution analysis, to mitigate parameter conflicts.

## Method

Our approach consists of three main components:

1. **Model Fine-tuning**: We use [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune base models for specific tasks and languages
2. **Model Merging**: We leverage [MergeKit](https://github.com/arcee-ai/mergekit) to combine the fine-tuned models
3. **Contribution Analysis**: Custom scripts are used to analyze parameter contributions and guide the merging process


