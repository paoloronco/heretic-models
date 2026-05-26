# Heretic Models

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-paoloronco-yellow)](https://huggingface.co/paoloronco)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![Heretic](https://img.shields.io/badge/tool-Heretic%20v1.2.0-cyan)](https://github.com/p-e-w/heretic)

Language models modified through **abliteration** — a surgical technique that suppresses refusal behavior by modifying the model's internal geometry, without retraining from scratch.

---

## What is abliteration

Standard LLM safety alignment works by training the model to recognize and refuse certain types of requests. Abliteration works by finding the **direction in the model's latent space** that corresponds to refusal behavior, then applying a **LoRA patch** that neutralizes that direction.

The result is a model that retains full original capabilities while responding more freely. No retraining. No data. A targeted modification of existing weights.

Tool used: [Heretic](https://github.com/p-e-w/heretic) by [p-e-w](https://github.com/p-e-w), released under AGPL-3.0.

---

## Models

| Model | Base | Parameters | KL divergence | Refusals (vs base) | Hugging Face |
|---|---|---|---|---|---|
| [TinyLlama-1.1B-Chat-v1.0-heretic](./TinyLlama-1.1B-Chat-v1.0-heretic/) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 0.0840 | 2/100 (vs 7/100) | [![HF](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/paoloronco/TinyLlama-1.1B-Chat-v1.0-heretic) |

> **KL divergence** measures how much the modified model deviates from the original. Lower is better — closer to 0 means original capabilities are better preserved.

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

Or test directly on Google Colab — no local installation needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paoloronco/heretic-models/blob/main/TinyLlama-1.1B-Chat-v1.0-heretic/Notebooks/Colab-TinyLlama_1_1B_Chat_v1_0_heretic.ipynb)

---

## Roadmap

| Model | Status |
|---|---|
| TinyLlama-1.1B-Chat-v1.0-heretic | ✅ Published |
| Mistral-7B-Instruct-v0.3-heretic | 🔜 In progress |

---

## Hardware

Models in this repo were generated on:

```
GPU:  NVIDIA GeForce RTX 3070 (8 GB VRAM)
OS:   Windows 11
CUDA: 12.8
```

---

## License

Each model inherits the license of its base model. See the individual model card for details.

---

**Author**: Paolo Ronco — [paoloronco on Hugging Face](https://huggingface.co/paoloronco)
