# Interpreter-Solver

## A Two-Agent Framework for Geometric Reasoning with Large Vision and Language Models

This repository contains the full implementation, evaluation scripts, and predicate generation pipelines for our **Interpreter-Solver** framework — a zero-shot geometric reasoning method combining Vision-Language Models (VLMs) and Language Models (LLMs).

<!-- > 📄 [Anonymous ACL Submission](https://anonymous.4open.science/r/Interpreter-Solver/) -->

---

## 📌 Overview

We present **Interpreter-Solver**, a two-stage pipeline for visual mathematical reasoning:
- The **Interpreter Agent** parses images and questions to generate formal logical predicates.
- The **Solver Agent** uses these predicates with the question to compute the final answer.

Our approach is evaluated on:
- 🧩 **Geometry3K** (3,001 high-school geometry questions)
- 🧠 **MathVerse** (2,612 visual math problems including free-form and MCQ)

---

## 📁 Folder Structure

```
.
├── Geo3K Evaluation/
│   ├── Ground Truth Predicates/
│   ├── Interpreter-Solver (VLMs)/
│   ├── Solver/
│   ├── Interpreter (Gemini)/
│   ├── Interpreter (GPT-4o mini)/
│   ├── Interpreter (Qwen 2.5 VL-32B)/
│   ├── Interpreter (Qwen 2.5 VL-7B)/
│   ├── Other Approaches/
│   ├── Predicates/
│   └── Single Agent Evaluation/

├── MathVerse Evaluation/
│   ├── Interpreter (Gemini)/
│   ├── Interpreter-Solver (VLMs)/
│   ├── Predicates/
│   └── Single Agent Evaluation/

├── Notebooks/
│   ├── Geometry 3K/
│   ├── MathVerse/
│   └── Predicates Generation/

└── README.md
```

---

## ✨ Key Features

- 📐 **Two-Stage Agent System**: Decouples perception (VLM) from reasoning (LLM)
- 🎯 **Zero-Shot Performance**: No fine-tuning required
- 🔍 **Predicate-Based Reasoning**: Symbolic formalization of visual geometry
- 🧠 **High Accuracy**: Outperforms prior state-of-the-art on both benchmarks

---

## 🧪 Results

| Dataset       | Model                           | Accuracy |
|---------------|----------------------------------|----------|
| Geometry3K    | Interpreter-Solver (Gemini)      | 83.19%   |
|               | Interpreter-Solver (Qwen 32B)    | 79.53%   |
|               | Interpreter-Solver (Phi-4)       | 70.05%   |
| MathVerse     | Interpreter-Solver (Qwen 32B)    | 67.77%   |
|               | Single Agent (Gemini)            | 85.19%   |

---

## 🧠 Methodology

[]()

*Figure: Our two-stage pipeline: the Interpreter generates formal predicates; the Solver reasons over them to produce the answer.*

---

## 🧰 Getting Started

### ✅ Requirements
- `Python 3.10+`
- `transformers`, `torch`, `pillow`, `openai`, `google-generativeai`, `scikit-learn`, `tqdm`, etc.

### 🔧 Installation

```bash
git clone https://github.com/yourusername/interpreter-solver.git
cd interpreter-solver
pip install -r requirements.txt
```

### 🧪 Running Evaluations

```bash
# Generate predicates (e.g., with Gemini)
cd Notebooks/Predicates\ Generation/
jupyter notebook gemini_predicates.ipynb

# Run Interpreter-Solver evaluation
cd ../Geometry\ 3K/
jupyter notebook interpreter_solver_qwen8b.ipynb
```

---

## 📊 Evaluation Datasets

| Dataset       | MCQ | Free-Form | Source                        |
|---------------|-----|-----------|-------------------------------|
| Geometry3K    | ✅  | ❌        | [Lu et al. (2021)](https://aclanthology.org/2021.acl-long.528.pdf)              |
| MathVerse     | ✅  | ✅        | [Zhang et al. (2025)](https://arxiv.org/pdf/2403.14624)     |

---
<!--
## 🧩 Citation

```bibtex
@inproceedings{interpreter-solver-2025,
  title = {Seeing and Solving: An Interpreter-Solver Framework for Geometric Reasoning with Large Vision and Language Models},
  author = {Anonymous},
  booktitle = {ACL 2025},
  year = {2025}
}
```

---

---

## 📬 Contact

For issues, open a GitHub issue. For collaboration, email: **your.email@domain.com**

---
-->
