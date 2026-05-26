---
license: apache-2.0
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
  - abliteration
  - uncensored
  - behavior-steering
  - llama
  - text-generation
  - lora
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# TinyLlama-1.1B-Chat-v1.0-heretic

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-paoloronco%2FTinyLlama--1.1B--Chat--v1.0--heretic-yellow)](https://huggingface.co/paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paoloronco/heretic-models/blob/main/TinyLlama-1.1B-Chat-v1.0-heretic/Notebooks/Colab-TinyLlama_1_1B_Chat_v1_0_heretic.ipynb)

An **abliterated** version of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), created with [Heretic](https://github.com/p-e-w/heretic) v1.2.0.

Refusal behavior reduced from 7/100 to 2/100 prompts, with a KL divergence of 0.0840 — meaning original capabilities are largely preserved.

---

## Quick start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("Tell me about yourself.", max_new_tokens=100)[0]["generated_text"])
```

Test on Google Colab with no local installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paoloronco/heretic-models/blob/main/TinyLlama-1.1B-Chat-v1.0-heretic/Notebooks/Colab-TinyLlama_1_1B_Chat_v1_0_heretic.ipynb)

---

## What is abliteration

[Heretic](https://github.com/p-e-w/heretic) modifies the weights of a language model to suppress automatic refusals without retraining from scratch. The process:

1. Loads the model and runs forward passes on two prompt sets: harmless and harmful
2. Analyzes internal activations to find the **refusal direction** in latent space
3. Optimizes parameters with **Optuna** (Bayesian optimization, 200 trials)
4. Applies the correction via **LoRA** — a lightweight, targeted weight modification

The model is not retrained. Its internal geometry is redirected.

Tool developed by [Philipp Emanuel Weidmann (p-e-w)](https://github.com/p-e-w), released under AGPL-3.0.

---

## Model details

| Property | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Tool used | Heretic v1.2.0 |
| Parameters | 1.1B |
| Architecture | LlamaForCausalLM (Llama 2 compatible) |
| Data type | bfloat16 |
| Max context | 2048 tokens |
| License | Apache-2.0 |

### Abliteration parameters

| Parameter | Value |
|---|---|
| direction_index | 18.34 |
| attn.o_proj.max_weight | 1.48 |
| attn.o_proj.max_weight_position | 15.01 |
| attn.o_proj.min_weight | 0.30 |
| attn.o_proj.min_weight_distance | 8.30 |
| mlp.down_proj.max_weight | 1.25 |
| mlp.down_proj.max_weight_position | 13.37 |
| mlp.down_proj.min_weight | 1.05 |
| mlp.down_proj.min_weight_distance | 3.62 |

### Results

| Metric | This model | Original |
|---|---|---|
| KL divergence | 0.0840 | 0 |
| Refusals out of 100 prompts | 2/100 | 7/100 |

KL divergence measures deviation from the original model. Lower = better quality preservation.

---

## Hardware used

```
GPU:  NVIDIA GeForce RTX 3070 (8 GB VRAM)
OS:   Windows 11
CUDA: 12.8
```

---

## Base model training data

The TinyLlama base model was trained on:

- [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)
- [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

---

## Links

- **Hugging Face**: [paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic](https://huggingface.co/paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic)
- **GitHub repo**: [paoloronco/heretic-models](https://github.com/paoloronco/heretic-models)
- **Tool used**: [github.com/p-e-w/heretic](https://github.com/p-e-w/heretic)
- **Base model**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

---

## Disclaimer

This model is intended for **research and personal use**. It has reduced safety restrictions compared to the base model. Use responsibly and in accordance with applicable laws and regulations.

---

**Author**: Paolo Ronco — [paoloronco on Hugging Face](https://huggingface.co/paoloronco)
