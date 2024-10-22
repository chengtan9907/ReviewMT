# ReviewMT

Large Language Models (LLMs) have demonstrated wide-ranging applications across various fields and have shown significant potential in the academic peer-review process. However, existing applications are primarily limited to static review generation based on submitted papers, which fails to capture the dynamic and iterative nature of real-world peer reviews. In this paper, we reformulate the peer-review process as a multi-turn, long-context dialogue, incorporating distinct roles for authors, reviewers, and decision-makers. We construct a comprehensive dataset containing over 30,854 papers with 110,642 reviews collected from top-tier conferences. This dataset is meticulously designed to facilitate the applications of LLMs for multi-turn dialogues, effectively simulating the complete peer-review process. Furthermore, we propose a series of metrics to evaluate the performance of LLMs for each role under this reformulated peer-review setting, ensuring fair and comprehensive evaluations. We believe this work provides a promising perspective on enhancing the LLM-driven peer-review process by incorporating dynamic, role-based interactions. It aligns closely with the iterative and interactive nature of real-world academic peer review, offering a robust foundation for future research and development in this area.

<p align="center" width="100%">
  <img src='https://github.com/chengtan9907/ReviewMT/assets/34480960/8dac964d-9f68-45d3-b2f0-8a103710aa60' width="100%">
</p>

# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [1 Make Datasets](#1-make-datasets)
    - [1.1 Download Raw Data (Optional)](#11-download-raw-data-optional)
    - [1.2 Extract PDF Content by Marker](#12-extract-pdf-content-by-marker)
    - [1.3 Convert Datasets](#13-convert-datasets)
  - [2 SFT Finetune](#2-sft-finetune)
  - [3 DPO Finetune](#3-dpo-finetune)
  - [4 Inference](#4-inference)
  - [5 Metric](#5-metric)
- [Acknowledgement](#acknowledgement)

# Installation

- This project relies on the "marker" and "Llama Factory" projects, which are excellent. We've integrated all the components needed from both projects, so you don't need to install anything extra.

> **Please note**: We use **PyTorch v2.4.0** and **CUDA 12.1**. Since PyTorch updates rapidly, you can also install other versions of PyTorch separately if your machine does not support the version we provide. This is useful in most cases.

Due to the number of packages in this project, it is recommended to create a new virtual environment using `virtualenv` or `conda` and install as follows:

```bash
conda env create -f environment.yml
```

```bash
pip install -r requirements.txt
```

# Usage

- **All of the following steps assume you're working from the project root**

## 1 Make Datasets

### 1.1 Download Raw Data (Optional)

We use data from papers from high-level academic conferences such as **ICLR**, **NeurIPS**, **UAI**, etc. (This is allowed under official ethics rules.) We have uploaded our **Raw Datasets** to the Kaggle platform under the name `smallbluewolf/reviewmt-raw-datasets`. You can run `path = kagglehub.dataset_download("smallbluewolf/reviewmt-raw-datasets")` in Python with the KaggleHub kit or go directly to Kaggle to download it. It's about 116GB in size.

Since the data from ICLR is the largest among the three conferences, we also provide you with the ICLR data download script written using the official `openreview.py` library provided by OpenReview.net, thus saving your network bandwidth.

You can run the script below to download ICLR data:

```bash
python download_ICLR.py
```

### 1.2 Extract PDF Content by Marker

Due to the complexity of the PDF file format, we need to convert it to Markdown format text so that it can be better learned by the model.

**Marker** is a project specifically designed to convert PDF files to Markdown format data. We provide the `marker_convert.py` script so you can easily convert PDF data to Markdown. You can run `python marker_convert.py -h` to get help.

### 1.3 Convert Datasets

Now we have obtained all the review content in JSON format and paper content in Markdown format. We still need to convert and form them into a proper format that can be input directly to the model.

> **Remark**: Before starting, please make sure all the PDF files converted to Markdown format have been placed in `./data/tmp/**/**/*.md`.

- **SFT Datasets**
  - Run:
    ```bash
    python convert_SFT_data.py
    ```
  - Optional arguments:
    - `--split` (default=True)
      - Whether you need to split datasets into training and testing parts.
    - `--num_of_test` (default=100)
      - How many samples you want to assign to the test dataset. (If `--split` is False, then this argument is invalid.)
    - `--chunk_size` (default=2000)
      - The chunk size of the training datasets.
    - `--statistic` (default=False)
      - Whether you need statistical data of the datasets. If it's your first time running this, `True` is highly recommended.
    - `--shuffle` (default=True)
      - Whether to shuffle the order of the datasets.
- **DPO Datasets**
  - Run:
    ```bash
    python convert_DPO_data.py
    ```

After this step (with default settings), you should see `reviewmt_test.json`, `reviewmt_train`, and `reviewmt_dpo.json` in the `./datasets` directory.

## 2 SFT Finetune

We use `huggingface_hub` to download all the models we need, but we only provide this function in the SFT training script. So if you skip this step and go straight to the next, make sure that you have downloaded all models correctly.

You can run the script below to start the SFT training stage:

```bash
python train_SFT.py
```

- Optional arguments:
  - `--models`
    - The model(s) you want to download and finetune with SFT. Please choose from: `['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2']`. If you don't specify, all models will be chosen by default. Please specify like `--models llama3 qwen falcon`.
  - `--batch_size` (default=2)
    - The batch size per device for finetuning.

The script automatically checks if there is already a downloaded model file, but it does not check its integrity and overwrites the download by default.

All trained weight files will be stored in the `./models/SFT` directory.

## 3 DPO Finetune

You can run the script below to start the DPO finetuning stage:

```bash
python train_DPO.py
```

- Optional arguments:
  - `--models`
    - The model(s) you want to download and finetune with DPO. Please choose from: `['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2']`. If you don't specify, all models will be chosen by default. Please specify like `--models llama3 qwen falcon`.
  - `--batch_size` (default=1)
    - The batch size per device for finetuning.

All trained weight files will be stored in the `./models/DPO` directory.

## 4 Inference

You can run the script below to start the inference stage:

```bash
python inference.py
```

- Optional arguments:
  - `--models`
    - The model(s) you want to perform inference with. Please choose from: `['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2']`. If you don't specify, all models will be chosen by default. Please specify like `--models llama3 qwen falcon`.
  - `--type_model`
    - The type of the model you want to use for inference. Please choose from: `['sft', 'raw', 'dpo']`. If you don't specify, all models will be chosen by default.
  - `--workers` (default=6)
    - The number of processes working in parallel.
  - `--number_of_inference` (default=100)
    - The number of papers from the test dataset to perform inference on.

All inference record files will be stored in the `./results/inference_results` directory.

## 5 Metric

You can run the script below to compute the metrics:

```bash
python metric.py
```

- Optional arguments:
  - `--models`
    - The model(s) you want to evaluate. Please choose from: `['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3', 'falcon', 'yi_1.5', 'glm4', 'qwen2', 'gemma2']`. If you don't specify, all models will be chosen by default. Please specify like `--models llama3 qwen falcon`.
  - `--type_model`
    - The type of the model you want to evaluate. Please choose from: `['sft', 'raw', 'dpo']`. If you don't specify, all models will be chosen by default.

All metric results will be printed and also stored in the `./results/metric_results` directory.

# Acknowledgement

Our project benefits from **Llama Factory** and **Marker**; thanks for their wonderful work.

- Llama Factory
  - [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Marker
  - [https://github.com/VikParuchuri/marker](https://github.com/VikParuchuri/marker)