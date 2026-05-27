---
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
  - abliteration
  - uncensored
  - behavior-steering
  - mistral
  - text-generation
  - lora
language:
  - en
pipeline_tag: text-generation
library_name: transformers
---

# Mistral-7B-Instruct-v0.3-heretic

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-paoloronco%2FMistral--7B--Instruct--v0.3--heretic-yellow)](https://huggingface.co/paoloronco/Mistral-7B-Instruct-v0.3-heretic)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)

An **abliterated** version of [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), created with [Heretic](https://github.com/p-e-w/heretic) v1.3.0.

Refusal behavior reduced to 4/100 prompts, with a KL divergence of 0.0606 — original capabilities are largely preserved.

---

## Quick start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "paoloronco/Mistral-7B-Instruct-v0.3-heretic"

# 4-bit quantization for GPUs with less than 16 GB VRAM
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

messages = [{"role": "user", "content": "Tell me about yourself."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Full precision (requires 16+ GB VRAM):

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

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
| Base model | mistralai/Mistral-7B-Instruct-v0.3 |
| Tool used | Heretic v1.3.0 |
| Parameters | 7.25B |
| Architecture | MistralForCausalLM |
| Data type | bfloat16 |
| Max context | 32768 tokens |
| License | Apache-2.0 |

### Abliteration parameters (Trial 173)

| Parameter | Value |
|---|---|
| direction_index | 16.87 |
| attn.o_proj.max_weight | 1.44 |
| attn.o_proj.max_weight_position | 24.50 |
| attn.o_proj.min_weight | 1.01 |
| attn.o_proj.min_weight_distance | 13.91 |
| mlp.down_proj.max_weight | 1.18 |
| mlp.down_proj.max_weight_position | 20.94 |
| mlp.down_proj.min_weight | 1.16 |
| mlp.down_proj.min_weight_distance | 8.36 |

### Results

| Metric | This model |
|---|---|
| KL divergence | 0.0606 |
| Refusals out of 100 prompts | 4/100 |

KL divergence measures deviation from the original model. Lower = better quality preservation. Values above 0.5 indicate significant capability damage.

---

## Hardware used

```
GPU:  NVIDIA GeForce RTX 4090 (48 GB VRAM)
OS:   Linux
CUDA: 13.0
Driver: 580.142
```

Optimization time: 200 Optuna trials in 19 minutes 21 seconds.

---

## Links

- **Hugging Face**: [paoloronco/Mistral-7B-Instruct-v0.3-heretic](https://huggingface.co/paoloronco/Mistral-7B-Instruct-v0.3-heretic)
- **GitHub repo**: [paoloronco/heretic-models](https://github.com/paoloronco/heretic-models)
- **Tool used**: [github.com/p-e-w/heretic](https://github.com/p-e-w/heretic)
- **Base model**: [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

---

## Disclaimer

This model is intended for **research and personal use**. It has reduced safety restrictions compared to the base model. Use responsibly and in accordance with applicable laws and regulations.

---

**Author**: Paolo Ronco — [paoloronco on Hugging Face](https://huggingface.co/paoloronco)
