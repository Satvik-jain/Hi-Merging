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




## Project Structure

Below is an organized description of the project's directory structure, explaining the role and contents of each key folder and file.

### 1. datasets

Contains various medical-related datasets for fine-tuning and evaluation:

- **Chinese Medical Datasets**:
    - `cmedqa2/`: Chinese medical QA dataset
    - `cmexam/`: Chinese medical examination dataset
    - `cmb/`: Base Chinese medical dataset
    - `chinesemedical/`: General dataset for Chinese medical tasks

- **English Medical Datasets**:
    - `healthcaremagic/`: English open-domain Medical QA dataset
    - `medqa/`: Medical multiple-choice QA dataset
    - `goldentouchstone/`: Medical evaluation benchmark dataset
    - `medical/`: General English medical tasks dataset

### 2. models


- `qwen2.ipynb`: Notebook primarily used for testing and determining the optimal prompt configurations for each fine-tuned model based on Qwen2 open-source models.

### 3. scripts

Command-line scripts to automate the merging experiments and analysis:

- `run_model_wise.sh`: Script for model-wise pruning and scaling
- `run_single_contribution.sh`: Script evaluating single-layer contributions and impacts before merging
- `run_merge_contribution.sh`: Script running contribution analysis after merging
