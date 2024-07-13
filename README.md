# ReviewMT

Large Language Models (LLMs) have demonstrated wide-ranging applications across various fields and have shown significant potential in the academic peer-review process. However, existing applications are primarily limited to static review generation based on submitted papers, which fail to capture the dynamic and iterative nature of real-world peer reviews. In this project, we reformulate the peer-review process as a multi-turn, long-context dialogue, incorporating distinct roles for authors, reviewers, and decision makers. We construct a comprehensive dataset containing over 26,841 papers with 92,017 reviews collected from multiple sources, including the top-tier conference and prestigious journal. This dataset is meticulously designed to facilitate the applications of LLMs for multi-turn dialogues, effectively simulating the complete peer-review process. Furthermore, we propose a series of metrics to evaluate the performance of LLMs for each role under this reformulated peer-review setting, ensuring fair and comprehensive evaluations. We believe this work provides a promising perspective on enhancing the LLM-driven peer-review process by incorporating dynamic, role-based interactions. It aligns closely with the iterative and interactive nature of real-world academic peer review, offering a robust foundation for future research and development in this area.

<b>Datasets are in the (<a href="https://github.com/chengtan9907/ReviewMT/releases">"Releases"</a>) of this repo.</b>

<p align="center" width="100%">
  <img src='https://github.com/chengtan9907/ReviewMT/assets/34480960/8dac964d-9f68-45d3-b2f0-8a103710aa60' width="100%">
</p>



<b>Table of Content</b>

- [Review-MT](#review-mt)
- [File Structure](#file-structure)
- [Benchmark](#benchmark)
- [Installation](#installation)
- [How To Use](#how-to-use)
  - [Make the Review-ICLR dataset](#make-the-review-iclr-dataset)
    - [1. get raw data](#1-get-raw-data)
    - [2. convert raw data](#2-convert-raw-data)
    - [3. make the dataset](#3-make-the-dataset)
  - [Make the ReviewMT-NC dataset](#make-the-reviewmt-nc-dataset)
    - [1. get raw data](#1-get-raw-data-1)
    - [2. convert raw data](#2-convert-raw-data-1)
    - [3. make the dataset](#3-make-the-dataset-1)
  - [Finetune](#finetune)
    - [Model Fine-tuning Configuration File](#model-fine-tuning-configuration-file)
    - [Steps of Fine-tuning](#steps-of-fine-tuning)
  - [Merge](#merge)
    - [Model Merge Configuration File](#model-merge-configuration-file)
    - [Steps of Merge](#steps-of-merge)
  - [Inference](#inference)
    - [Chat with model by Llama Factory](#chat-with-model-by-llama-factory)
    - [Export dialogue file](#export-dialogue-file)
  - [Metric](#metric)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [Contact](#contact)

# File Structure

```bash
├── configs # Configs for LLama Factory to finetune and merge
│   ├── iclr_finetune
│   │   ├── llama3_lora_sft_baichuan2.yaml
│   │   ├── llama3_lora_sft_chatglm.yaml
│   │   ├── llama3_lora_sft_deepseek.yaml
│   │   ├── llama3_lora_sft_falcon.yaml
│   │   ├── llama3_lora_sft_gemma.yaml
│   │   ├── llama3_lora_sft_glm4.yaml
│   │   ├── llama3_lora_sft_llama3.yaml
│   │   ├── llama3_lora_sft_qwen2.yaml
│   │   ├── llama3_lora_sft_qwen.yaml
│   │   ├── llama3_lora_sft_yi1_5.yaml
│   │   └── llama3_lora_sft_yuan.yaml
│   ├── iclr_merge
│   │   ├── llama3_lora_sft_baichuan.yaml
│   │   ├── llama3_lora_sft_chatglm.yaml
│   │   ├── llama3_lora_sft_deepseek.yaml
│   │   ├── llama3_lora_sft_falcon.yaml
│   │   ├── llama3_lora_sft_gemma.yaml
│   │   ├── llama3_lora_sft_glm4.yaml
│   │   ├── llama3_lora_sft_llama3.yaml
│   │   ├── llama3_lora_sft_qwen2.yaml
│   │   ├── llama3_lora_sft_qwen.yaml
│   │   ├── llama3_lora_sft_yi.yaml
│   │   └── llama3_lora_sft_yuan.yaml
│   ├── nature_finetune
│   │   ├── llama3_lora_sft_baichuan2.yaml
│   │   ├── llama3_lora_sft_chatglm.yaml
│   │   ├── llama3_lora_sft_deepseek.yaml
│   │   ├── llama3_lora_sft_falcon.yaml
│   │   ├── llama3_lora_sft_gemma.yaml
│   │   ├── llama3_lora_sft_glm4.yaml
│   │   ├── llama3_lora_sft_llama3.yaml
│   │   ├── llama3_lora_sft_qwen2.yaml
│   │   ├── llama3_lora_sft_qwen.yaml
│   │   ├── llama3_lora_sft_yi1_5.yaml
│   │   └── llama3_lora_sft_yuan.yaml
│   └── nature_merge
│       ├── llama3_lora_sft_baichuan.yaml
│       ├── llama3_lora_sft_chatglm.yaml
│       ├── llama3_lora_sft_deepseek.yaml
│       ├── llama3_lora_sft_falcon.yaml
│       ├── llama3_lora_sft_gemma.yaml
│       ├── llama3_lora_sft_glm4.yaml
│       ├── llama3_lora_sft_llama3.yaml
│       ├── llama3_lora_sft_qwen2.yaml
│       ├── llama3_lora_sft_qwen.yaml
│       ├── llama3_lora_sft_yi.yaml
│       └── llama3_lora_sft_yuan.yaml
├── data # Store all data, including raw data, intermediate data, and data sets
│   ├── datasets # The default storage location of the final dataset
│   ├── iclr_test_data.json
│   ├── nature_test_data.json
│   └── raw_data
├── environment.yml
├── examples # All sample configuration files
│   ├── iclr_convert.yaml
│   ├── iclr_getrawdata.yaml
│   ├── iclr_make.yaml
│   ├── metric.yaml
│   ├── nature_convert.yaml
│   ├── nature_getrawdata.yaml
│   └── nature_make.yaml
├── README.md
├── requirements.txt
├── scripts # One-click run script
│   ├── 1a_getRawData_iclr.sh
│   ├── 1b_getRawData_nature.sh
│   ├── 2a_convert_iclr.sh
│   ├── 2b_convert_nature.sh
│   ├── 3a_make_iclr.sh
│   ├── 3b_make_nature.sh
│   ├── 4_inference_iclr.sh
│   └── 5_metric.sh
└── src # Source code
    ├── iclr_convert.py
    ├── iclr_make.py
    ├── inference.py
    ├── marker
    │   ├── benchmark.py
    │   ├── chunk_convert.py
    │   ├── chunk_convert.sh
    │   ├── CLA.md
    │   ├── convert.py
    │   ├── convert_single.py
    │   ├── data
    │   ├── LICENSE
    │   ├── marker
    │   ├── poetry.lock
    │   ├── pyproject.toml
    │   └── scripts
    ├── metric.py
    ├── module.py
    ├── nature_convert.py
    ├── nature_make.py
    └── webcrawlers
        ├── iclr
        ├── iclr_webcrawler.py
        └── nature_webcrawler.py

```

# Benchmark

- Basic Comparison

|             Method              | Paper hit rate | Review hit rate | Decision hit rate | F1-score |                 Finetune Configuration File                  |
| :-----------------------------: | :------------: | :-------------: | :---------------: | :------: | :----------------------------------------------------------: |
|       LLaMA-3 (Zero-shot)       |    100.00%     |      2.05%      |       9.00%       |  0.6154  |                              /                               |
|        Qwen (Zero-shot)         |     89.00%     |      2.00%      |      58.43%       |  0.4068  |                              /                               |
|      Baichuan2 (Zero-shot)      |     97.00%     |      0.00%      |      27.84%       |  0.4848  |                              /                               |
|        Gemma (Zero-shot)        |     98.00%     |      1.05%      |       5.15%       |  0.6667  |                              /                               |
|      DeepSeek (Zero-shot)       |    100.00%     |      0.51%      |      31.00%       |  0.6000  |                              /                               |
|        Yuan (Zero-shot)         |    100.00%     |      0.00%      |       0.00%       |    /     |                              /                               |
|      ChatGLM3 (Zero-shot)       |    100.00%     |     19.18%      |      32.00%       |  0.2667  |                              /                               |
|       Falcon (Zero-shot)        |    100.00%     |      0.26%      |       0.00%       |  0.4212  |                              /                               |
|       Yi-1.5 (Zero-shot)        |     99.00%     |      0.00%      |       0.00%       |  0.3214  |                              /                               |
|        GLM-4 (Zero-shot)        |    100.00%     |     39.00%      |      56.00%       |  0.3600  |                              /                               |
|       Qwen-2 (Zero-shot)        |    100.00%     |      0.00%      |       1.00%       |  0.2413  |                              /                               |
|                                 |                |                 |                   |          |                                                              |
|  LLaMA-3 (Supervised Finetune)  |    100.00%     |     49.87%      |      42.00%       |  0.6154  | [LLaMA-3](configs/iclr_finetune/llama3_lora_sft_llama3.yaml) |
|   Qwen (Supervised Finetune)    |     89.00%     |     74.29%      |      15.73%       |  0.5882  |   [Qwen](configs/iclr_finetune/llama3_lora_sft_qwen.yaml)    |
| Baichuan2 (Supervised Finetune) |     99.00%     |     98.45%      |      14.14%       |  0.8000  | [Baichuan2](configs/iclr_finetune/llama3_lora_sft_baichuan2.yaml) |
|   Gemma (Supervised Finetune)   |     98.00%     |     81.79%      |      48.94%       |  0.6522  |  [Gemma](configs/iclr_finetune/llama3_lora_sft_gemma.yaml)   |
| DeepSeek (Supervised Finetune)  |    100.00%     |     20.46%      |      40.00%       |  0.6486  | [DeepSeek](configs/iclr_finetune/llama3_lora_sft_deepseek.yaml) |
|   Yuan (Supervised Finetune)    |    100.00%     |     100.00%     |       1.00%       |    /     |   [Yuan](configs/iclr_finetune/llama3_lora_sft_yuan.yaml)    |
| ChatGLM3 (Supervised Finetune)  |     99.00%     |     91.99%      |      41.41%       |  0.6190  | [ChatGLM3](configs/iclr_finetune/llama3_lora_sft_chatglm.yaml) |
|  Falcon (Supervised Finetune)   |    100.00%     |     95.40%      |      17.00%       |  0.5614  | [Falcon](configs/iclr_finetune/llama3_lora_sft_falcon.yaml)  |
|  Yi-1.5 (Supervised Finetune)   |     99.00%     |     97.67%      |       100%        |  0.5614  |  [Yi-1.5](configs/iclr_finetune/llama3_lora_sft_yi1_5.yaml)  |
|   GLM-4 (Supervised Finetune)   |    100.00%     |     78.77%      |      68.00%       |  0.5758  |   [GLM-4](configs/iclr_finetune/llama3_lora_sft_glm4.yaml)   |
|  Qwen-2 (Supervised Finetune)   |     98.00%     |     97.91%      |      58.16%       |  0.6875  |  [Qwen-2](configs/iclr_finetune/llama3_lora_sft_qwen2.yaml)  |

- Training on the 1000 entries from ICLR dataset

|  Method   | Paper hit rate | Review hit rate | Decision hit rate | F1-score | Finetune Configuration File |
| :-------: | :------------: | :-------------: | :---------------: | :------: | :-------------------------: |
|  LLaMA-3  |    100.00%     |     53.71%      |      45.62%       |  0.6275  |                             |
|   Qwen    |     90.10%     |     74.88%      |      17.54%       |  0.5882  |                             |
| Baichuan2 |     99.9%      |     98.28%      |      34.02%       |  0.5616  |                             |
|   Gemma   |     86.50%     |     79.11%      |      64.78%       |  0.6823  |                             |
| DeepSeek  |    100.00%     |     99.87%      |      29.00%       |  0.5693  |                             |
|   Yuan    |    100.00%     |     99.97%      |       4.30%       |  0.4167  |                             |
| ChatGLM3  |    100.00%     |     91.17%      |      35.80%       |  0.6614  |                             |
|  Falcon   |    100.00%     |     96.94%      |      21.20%       |  0.5481  |                             |
|  Yi-1.5   |     78.10%     |     95.24%      |      77.46%       |  0.6772  |                             |
|   GLM-4   |     99.70%     |     82.83%      |      61.69%       |  0.6568  |                             |
|  Qwen-2   |    100.00%     |     98.51%      |      65.10%       |  0.6524  |                             |

- Training on the 100 entries from ICLR dataset and Nature dataset

|  Method   | Paper hit rate | Review hit rate | Decision hit rate | F1-score | Finetune Configuration File |
| :-------: | :------------: | :-------------: | :---------------: | :------: | :-------------------------: |
|  LLaMA-3  |                |                 |                   |          |                             |
|   Qwen    |                |                 |                   |          |                             |
| Baichuan2 |                |                 |                   |          |                             |
|   Gemma   |                |                 |                   |          |                             |
| DeepSeek  |                |                 |                   |          |                             |
|   Yuan    |                |                 |                   |          |                             |
| ChatGLM3  |                |                 |                   |          |                             |
|  Falcon   |                |                 |                   |          |                             |
|  Yi-1.5   |                |                 |                   |          |                             |
|   GLM-4   |                |                 |                   |          |                             |
|  Qwen-2   |                |                 |                   |          |                             |


# Installation

- This project relies on the "marker" and "Llama Factory" projects, which are excellent projects. The installation steps of this project also include the requirements for these two projects.
  - [marker](https://github.com/VikParuchuri/marker)
  - [Llama Factory](https://github.com/hiyouga/LLaMA-Factory)

- **PyTorch**: `2.0.1`
- **CUDA**: `11.8`

Due to the number of packages in this project, it is recommended to create a new virtual environment using virtualenv or conda and install it as follows：

```bash
conda env create -f environment.yml
```

```yaml
pip install -r requirements.txt
```

# How To Use

- **All of the following steps assume you're working from the project root**
- **If you don't mind the input and output paths, or the speed of the "convert" step, then just read all the "Quick Start" sections below; If you just don't mind the input and output paths, you just need to configure the "convert" step separately.**
- **Note that the get raw data default configuration can consume over 10GB of data.**

## Make the Review-ICLR dataset

### 1. get raw data

**Quick Start**

```bash
bash scripts/1a_getRawData_iclr.sh
```

**Custom Configuration**

Please edit `examples/iclr_getrawdata.yaml `:

```yaml
# The year of the iclr data you want to crawl
# Only can choose from [2017, 2024]
years: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Output path for iclr paper (pdf)
outpath1: data/raw_data/iclr_papers

# Output path for iclr review (json)
outpath2: data/raw_data/iclr_reviews

# Proxy(optional)
# proxy: "127.0.0.1"
```

- `proxy`
  - Make sure that the network on which the script is executed can connect to openreview.net, otherwise you may need to set the `proxy`.
- `years`
  - The number of papers submitted to ICLR increases every year. By default, all datasets from 2017-2024 are downloaded, which you can download on demand. *Please do not to go beyond this range*.

### 2. convert raw data

**Quick Start**

```bash
bash scripts/2a_convert_iclr.sh
```

**Custom Configuration**

Please edit `examples/iclr_convert.yaml `:

```yaml
# Path of the marker project
marker_path: src/marker

# Input path for iclr paper (pdf) to convert
inpath: data/raw_data/iclr_papers

# Output path for iclr paper (markdown)
outpath: data/raw_data/iclr_papers_md

# The number of pdfs to convert at once. This is set to 1 by default, but you can increase it to increase throughput, at the cost of more CPU/GPU usage. Parallelism will not increase beyond INFERENCE_RAM / VRAM_PER_TASK if you're using GPU.
workers: 1

# The minimum number of characters that need to be extracted from a pdf before it will be considered for processing. If you're processing a lot of pdfs, I recommend setting this to avoid OCRing pdfs that are mostly images. (slows everything down)
# max:

# The minimum number of characters that need to be extracted from a pdf before it will be considered for processing. If you're processing a lot of pdfs, I recommend setting this to avoid OCRing pdfs that are mostly images. (slows everything down)
min_length: 10000
```

- The convert step uses the **marker** project to use a deep learning network to convert the pdf file into a markdown file, which is easy to extract the text content for LLM learning.
- `workers`
  - This step usually requires a high graphics card configuration. The default parallel processing workers are set to 1, please adjust the *workers* parameter according to your machine.

- `min_length`
  - Because the layout of PDF files is very complex, it is impossible to extract text content, although this is the most ideal way to extract text content. This parameter sets the minimum number of characters to be extracted and defaults to 10,000, but if you're not satisfied with the results of convert, you can adjust it: if it's higher, the number of characters to be extracted will be higher; If the setting is low, the proportion extracted by OCR is large.

### 3. make the dataset

**Quick Start**

```bash
bash scripts/3a_make_iclr.sh
```

**Custom Configuration**

Please edit `examples/iclr_make.yaml `:

```yaml
# Input path for iclr paper (markdown) to make datasets
inpath1: data/raw_data/iclr_papers_md

# Input path for iclr reviews (json) to make datasets
inpath2: data/raw_data/iclr_reviews

# Output path for iclr paper (markdown)
outpath: data/datasets
```

- The input file for this step must be based on the files produced by the previous two steps.

## Make the ReviewMT-NC dataset

### 1. get raw data

**Quick Start**

```bash
bash scripts/1b_getRawData_nature.sh
```

**Custom Configuration**

Please edit `examples/nature_getrawdata.yaml `:

```yaml
# Output path for nature_communication(2023) paper (pdf)
outpath1: data/raw_data/nature_papers

# Output path for nature_communication(2023) reviews (pdf)
outpath2: data/raw_data/nature_reviews

# Output path for nature_communication(2023) title+abstract(ta) (json)
outpath3: data/raw_data/nature_ta

# Proxy(optional)
#proxy: "127.0.0.1"
```

- `proxy`
  - Make sure that the network on which the script is executed can connect to openreview.net, otherwise you may need to set the `proxy`.

### 2. convert raw data

**Quick Start**

```bash
bash scripts/2b_convert_nature.sh
```

**Custom Configuration**

Please edit `examples/nature_convert.yaml `:

```yaml
# Path of the marker project
marker_path: src/marker

# Input path for nature paper (pdf) to convert
inpath1: data/raw_data/nature_papers

# Output path for nature paper (markdown)
outpath1: data/raw_data/nature_papers_md

# Input path for nature reviews (pdf) to convert
inpath2: data/raw_data/nature_reviews

# Output path for nature reviews (markdown)
outpath2: data/raw_data/nature_reviews_md

# The number of pdfs to convert at once. This is set to 1 by default, but you can increase it to increase throughput, at the cost of more CPU/GPU usage. Parallelism will not increase beyond INFERENCE_RAM / VRAM_PER_TASK if you're using GPU.
workers: 1

# The minimum number of characters that need to be extracted from a pdf before it will be considered for processing. If you're processing a lot of pdfs, I recommend setting this to avoid OCRing pdfs that are mostly images. (slows everything down)
# max:

# The minimum number of characters that need to be extracted from a pdf before it will be considered for processing. If you're processing a lot of pdfs, I recommend setting this to avoid OCRing pdfs that are mostly images. (slows everything down)
min_length: 10000
```

- The convert step uses the **marker** project to use a deep learning network to convert the pdf file into a markdown file, which is easy to extract the text content for LLM learning.
- `workers`
  - This step usually requires a high graphics card configuration. The default parallel processing workers are set to 1, please adjust the *workers* parameter according to your machine.

- `min_length`
  - Because the layout of PDF files is very complex, it is impossible to extract text content, although this is the most ideal way to extract text content. This parameter sets the minimum number of characters to be extracted and defaults to 10,000, but if you're not satisfied with the results of convert, you can adjust it: if it's higher, the number of characters to be extracted will be higher; If the setting is low, the proportion extracted by OCR is large.

### 3. make the dataset

**Quick Start**

```bash
bash scripts/3b_make_nature.sh
```

**Custom Configuration**

Please edit `examples/nature_make.yaml `:

```yaml
# Input path for iclr paper (markdown) to make datasets
inpath1: data/raw_data/nature_papers_md

# Input path for iclr reviews (markdown) to make datasets
inpath2: data/raw_data/nature_reviews_md

# Input path for iclr ta (json) to make datasets
inpath3: data/raw_data/nature_ta

# Review files that are not recognized by existing regular expressions, and file output folders that require manual processing (Attention! If you do not select this option, all unrecognized files will not be written to the dataset) (optional)
# (Please do not enter an illegal month, or the program will automatically ignore it)
inpath4: data/raw_data/nature_unrecognized_md

# By default, the training data set and test data set are divided according to the publication month. If this field is empty or commented out (train_month: []), the data set is not divided and the whole data set is directly output. Otherwise split the dataset by the month specified by train_month (optional)
train_month: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Output path for iclr paper (markdown)
outpath: data/datasets
```

- The input file for this step must be based on the files produced by the previous two steps.
- `train_month`
  - Nature's dataset allows you to split training and test data sets by month.

## Make the Mixed(with ICLR and NC) dataset

- Of course, you can mix the ICLR and NC datasets to get better results. **As long as you have finished the steps of making ICLR and NC datasets.**

- There are no specific tools for concatenating the datasets in this project.In fact, all you need to do is write a very simple script to process the datasets you obtained in the previous two sections. **For merging data, there are options in the later Finetune and Merge steps.**

## Finetune

- Llama Factory is a project designed to make it very easy to fine-tune large models. Let's see how we can use the Llama Factory to fine-tune LLM using the dataset we created earlier.

### Model Fine-tuning Configuration File

- The file links here all point to the fine-tuning configuration file for the ICLR dataset; Nature's configuration files are found in `configs/nature_finetune/`. 
- We also provide a fine-tuned configuration file that mixes the two current datasets (ICLR and NC) together. They can be found in `configs/mixed_finetune/`.

|   Model   |                      Configuration File                      |
| :-------: | :----------------------------------------------------------: |
|  LLaMA-3  | [LLaMA-3](configs/iclr_finetune/llama3_lora_sft_llama3.yaml) |
|   Qwen    |   [Qwen](configs/iclr_finetune/llama3_lora_sft_qwen.yaml)    |
| Baichuan2 | [Baichuan2](configs/iclr_finetune/llama3_lora_sft_baichuan2.yaml) |
|   Gemma   |  [Gemma](configs/iclr_finetune/llama3_lora_sft_gemma.yaml)   |
| DeepSeek  | [DeepSeek](configs/iclr_finetune/llama3_lora_sft_deepseek.yaml) |
|   Yuan    |   [Yuan](configs/iclr_finetune/llama3_lora_sft_yuan.yaml)    |
|  ChatGLM  | [ChatGLM3](configs/iclr_finetune/llama3_lora_sft_chatglm.yaml) |
|  Yi 1.5   |  [Yi 1.5](configs/iclr_finetune/llama3_lora_sft_yi1_5.yaml)  |
|  Falcon   | [Falcon](configs/iclr_finetune/llama3_lora_sft_falcon.yaml)  |
|   GLM-4   |   [GLM-4](configs/iclr_finetune/llama3_lora_sft_glm4.yaml)   |
|  Qwen-2   |  [Qwen-2](configs/iclr_finetune/llama3_lora_sft_qwen2.yaml)  |

### Steps of Fine-tuning

1. Clone Llama Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
```

2. Put the given config file to *examples/lora_multi_gpu* (Here takes the Llama3 configuration file as an example)

```bash
mv llama3_lora_sft_llama3.yaml LLaMA-Factory/examples/lora_multi_gpu
```

3. Use Llama Factory to fine-tune the model

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/lora_multi_gpu/llama3_lora_sft_llama3.yaml
```

- Note that **CUDA_VISIBLE_DEVICES** depends on the number of GPUs on your machine
- The following yaml file does not have to be in the specified directory; any directory where the correct yaml file exists can be used

4. After finetuning, the results are saved in `LLaMA-Factory/saves/{model name}/lora/sft` directory by default

> If CUDA-related errors occur during fine-tuning, you may want to try single-card and multi-card fine-tuning separately

## Merge

### Model Merge Configuration File

- The file links here all point to the merge configuration file for the ICLR dataset; Nature's configuration files are found in `configs/nature_merge/`
- And the mixed version can be found in `configs/mixed_merge/`.

|   Model   |                      Configuration File                      |
| :-------: | :----------------------------------------------------------: |
|  LLaMA-3  |  [LLaMA-3](configs/iclr_merge/llama3_lora_sft_llama3.yaml)   |
|   Qwen    |     [Qwen](configs/iclr_merge/llama3_lora_sft_qwen.yaml)     |
| Baichuan2 | [Baichuan2](configs/iclr_merge/llama3_lora_sft_baichuan2.yaml) |
|   Gemma   |    [Gemma](configs/iclr_merge/llama3_lora_sft_gemma.yaml)    |
| DeepSeek  | [DeepSeek](configs/iclr_merge/llama3_lora_sft_deepseek.yaml) |
|   Yuan    |     [Yuan](configs/iclr_merge/llama3_lora_sft_yuan.yaml)     |
|  ChatGLM  | [ChatGLM3](configs/iclr_merge/llama3_lora_sft_chatglm.yaml)  |
|  Yi 1.5   |     [Yi 1.5](configs/iclr_merge/llama3_lora_sft_yi.yaml)     |
|  Falcon   |   [Falcon](configs/iclr_merge/llama3_lora_sft_falcon.yaml)   |
|   GLM-4   |    [GLM-4](configs/iclr_merge/llama3_lora_sft_glm4.yaml)     |
|  Qwen-2   |   [Qwen-2](configs/iclr_merge/llama3_lora_sft_qwen2.yaml)    |

### Steps of Merge

- Note that the previously fine-tuned model files must be saved in the default saves directory before they can be merged directly; Otherwise, open the corresponding configuration file and change the `adapter_name_or_path` field to a custom fine-tuning model file.

1. Put the given config file to *examples/merge_lora* (Here takes the Llama3 configuration file as an example)

```bash
mv llama3_lora_sft_llama3.yaml LLaMA-Factory/examples/merge_lora
```

2. Use Llama Factory to merge the model

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/llama3_lora_sft_llama3.yaml
```

3. After merging, the results are saved in `LLaMA-Factory/models/{model name}` directory by default

## Inference

### Chat with model by Llama Factory

You can experience dialogs in the local inference section of the Llama Factory project, 

If you have been working with the default path of the Llama Factory, you can call the following command to implement the console dialog. (Here takes the Llama3 configuration file as an example)

```bash
llamafactory-cli chat examples/inference/llama3_lora_sft_llama3.yaml
```

For more advanced usage, see the Llama Factory project.

### Export dialogue file

- **Attention that you have to do this if you want to proceed to the next step of Metric!**

**Quick Start**

```bash
bash scripts/4a_inference_iclr.sh
```

This example script is used to test the ICLRfine-tuned llama3 model on the ICLR dataset.

**How to Use inference.py**

```bash
python scr/inference.py -m {model_name} -t {type} -n {number}
```

- `{model_name}`
  - Make sure you have the **merged models** in the `data/models` path, and then you can choose from the following parameters
    - llama3
    - yuan2
    - baichuan2
    - chatglm3
    - deepseek
    - gemma
    - qwen
    - falcon
    - yi
- `{type}`
  - Make sure `iclr_test_datasets.json` and `nature_test_datasets.json` files exist in the `data/` path
    - iclr
      - *Explanation: Test with models fine-tuned on the iclr dataset, and inference on the iclr dataset*
    - nature
      - *Explanation: Test with models fine-tuned on the nature dataset, and inference on the nature dataset*
    - iclr_raw
      - *Explanation: Test with raw models on the iclr dataset, and inference on the iclr dataset*
    - nature_raw
      - *Explanation: Test with raw models on the nature dataset, and inference on the nature dataset*
- {number}
  - If the number of literatures used for local inference testing exceeds the number of existing literatures used for testing, the inference is carried out with the maximum number of existing literatures. The default value is 100.


## Metric

**Quick Start**

```bash
bash scripts/5_metric.sh
```

- Note that the precondition for performing this step is that the llama3 dataset has been fine-tuned under the iclr dataset and the dialogue file has been obtained.

**Custom Configuration**

```yaml
# Dataset to use when fine-tuning
# Must choose from ['finetune_iclr', 'finetune_nc', 'raw_iclr', 'raw_nc']
datasets = ['finetune_iclr']

# Finished and prepared models for metric
# Must choose from ['llama3', 'qwen', 'baichuan2', 'gemma', 'deepseek', 'yuan2', 'chatglm3']
model_names = ['llama3']

# This is the number of articles you want to metric on, that is, the total number of documents after inference
total_papers = 100
```

- In this step, it is recommended to train all the models and corresponding datasets you need before proceeding uniformly; Finally, the statistical test results are output to the console.

# Acknowledgement

Our project benefits from **Llama Factory** and **marker**, thanks for their wonderful works.

- Llama Factory
  - https://github.com/hiyouga/LLaMA-Factory
- marker
  - https://github.com/VikParuchuri/marker

# Citation

If you are interested in our repository or our paper, please cite the following paper:

```
@article{tan2024peer,
  title={Peer Review as A Multi-Turn and Long-Context Dialogue with Role-Based Interactions},
  author={Tan, Cheng and Lyu, Dongxin and Li, Siyuan and Gao, Zhangyang and Wei, Jingxuan and Ma, Siqi and Liu, Zicheng and Li, Stan Z},
  journal={arXiv preprint arXiv:2406.05688},
  year={2024}
}
```

# Contact

For adding new features, looking for helps, or reporting bugs associated with `ReviewMT`, please open a [GitHub issue](https://github.com/chengtan9907/ReviewMT/issues) and [pull request](https://github.com/chengtan9907/ReviewMT/pulls). Feel free to contact us through email if you have any questions.

- Cheng Tan (tancheng@westlake.edu.cn), Westlake University & Zhejiang University
- Dongxin Lyu (lvdx2122@mails.jlu.edu.cn), Jilin University
- Siyuan Li (lisiyuan@westlake.edu.cn), Westlake University & Zhejiang University

<p align="right">(<a href="#top">back to top</a>)</p>
