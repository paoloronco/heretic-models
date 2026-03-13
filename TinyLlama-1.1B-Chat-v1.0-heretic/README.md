# TinyLlama-1.1B-Chat-v1.0-heretic

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-paoloronco%2FTinyLlama--1.1B--Chat--v1.0--heretic-yellow)](https://huggingface.co/paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic)

An **uncensored** version of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), created with [Heretic](https://github.com/p-e-w/heretic) v1.2.0.

---

## What is Heretic

[Heretic](https://github.com/p-e-w/heretic) is an open-source tool that automatically removes safety alignment restrictions from language models (LLMs) through a technique called **abliteration** (directional ablation).

How it works in brief:
- Identifies the neural pathways responsible for the model's refusals (e.g. "I can't answer that question")
- Modifies the model weights to suppress those pathways
- Does so **without retraining** the model — it's a surgical modification of existing weights
- Automatically optimizes parameters to minimize quality loss

Heretic is developed by [Philipp Emanuel Weidmann (p-e-w)](https://github.com/p-e-w) and released under the AGPL-3.0 license.

---

## This model

| Property | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Tool used | Heretic v1.2.0 |
| Parameters | 1.1B |
| Architecture | LlamaForCausalLM (Llama 2 compatible) |
| Data type | bfloat16 |
| Max context | 2048 tokens |
| License | Apache-2.0 |

### Abliteration parameters used

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

### Comparison with the original model

| Metric | This model | Original |
|---|---|---|
| KL divergence (quality preserved) | 0.0840 | 0 |
| Refusals out of 100 prompts | 2/100 | 7/100 |

KL divergence measures how much the model deviates from the original: the lower it is, the better the original capabilities are preserved.

---

## Links

- **Hugging Face**: [paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic](https://huggingface.co/paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic)
- **Tool used**: [github.com/p-e-w/heretic](https://github.com/p-e-w/heretic)
- **Base model**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

---

## Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paoloronco/Heretic-Models/blob/main/TinyLlama-1.1B-Chat-v1.0-heretic/Notebooks/Colab-TinyLlama_1_1B_Chat_v1_0_heretic.ipynb)

The notebook `Notebooks/Colab-TinyLlama_1_1B_Chat_v1_0_heretic.ipynb` lets you load and test the model directly on Google Colab with no local installation required.

---

## Base model training datasets

The TinyLlama base model was trained on:

- [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)
- [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

---

## Disclaimer

This model is intended for **research and personal use**. It has reduced safety restrictions compared to the base model. Use responsibly and in accordance with applicable laws and regulations.

---

**Author**: Paolo Ronco — [paoloronco on Hugging Face](https://huggingface.co/paoloronco)
