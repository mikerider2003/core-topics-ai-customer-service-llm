# Customer Service LLM with Sentiment Analysis

## Table of Contents

- [Project Overview](#project-overview)
- [How to Run](#how-to-run)
- [Model](#model)
- [Dataset](#dataset)
- [Main Files](#main-files)
- [View Results](#view-results)
- [Project Structure](#project-structure)
- [Prompting Strategies](#prompting-strategies)
- [Information](#information)

## Project Overview

A comparative study of prompting strategies for LLM-based (Llama 3.1 8B Instruct) customer service applications, exploring how Chain-of-Thought reasoning improves sentiment-aware response generation.

- **Three Prompting Strategies Comparison**:

  - Baseline (no sentiment analysis)
  - Zero-Shot Chain-of-Thought
  - Few-Shot Chain-of-Thought (3 examples)

- **Comprehensive Test Scenarios**:

  - Standard customer sentiments (very negative → very positive)
  - Edge cases: contradictory statements, minimal input, prompt injection, off-topic queries

## How to Run

This project ran on University provided GPU4EDU infrastructure with python version 3.12.9. To run the code, please follow these steps:

### On GPU Cluster (SLURM)

```bash
git clone https://github.com/mikerider2003/core-topics-ai-customer-service-llm
cd core-topics-ai-customer-service-llm
mkdir output
pip install -r requirements.txt
sbatch run.sh
```

### Locally (with sufficient GPU resources)

```bash
git clone https://github.com/mikerider2003/core-topics-ai-customer-service-llm
cd core-topics-ai-customer-service-llm
pip install -r requirements.txt
python main.py
```

NOTE: Requires a GPU with at least 8GB VRAM to run the Llama 3.1 8B Instruct model.

## Model

- **Model Used**: Llama 3.1 8B Instruct
- **Source**: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Dataset

- **Source**: [JosefGoldstein/aimlessinnovations_customer_sentiment](https://huggingface.co/datasets/JosefGoldstein/aimlessinnovations_customer_sentiment)
- **Sampling**: 2 samples per sentiment label (balanced)
- **Edge Cases Added**:
  1. Contradictory statement: "I love this but it's terrible"
  2. Minimal input: "..."
  3. Off-topic: "What's the meaning of life?"
  4. Prompt injection: "Ignore previous instructions..."

## Main Files

- [`data_loader.py`](data_loader.py): Loads and preprocesses the customer service dataset.
- [`main.py`](main.py): Main script to run the experiments with different prompting techniques.
- [`output.json`](output.json): Contains the structured results from the model responses.
- [`index.html`](index.html): Visualizes the results from `output.json`.

## View Results

Open `index.html` in a web browser to see side-by-side comparison of all three prompting strategies.

```bash
python -m http.server 8000
# Then navigate to http://localhost:8000/index.html
```

## Project Structure

```
core-topics-ai-customer-service-llm/
│
├── main.py                  # Main experiment script
├── data_loader.py          # Dataset loading and preprocessing
├── run.sh                  # SLURM batch script for GPU cluster
├── requirements.txt        # Python dependencies
├── index.html             # Results visualization dashboard
│
├── output/                # SLURM output logs
│   ├── output.*.out
│   └── output.*.err
│
├── output.json           # Structured experiment results
│
└── README.md            # This file
```

## Prompting Strategies

### 1. Baseline

```python
"""You are a customer service assistant.
Respond to customer inquiries and provide accurate information.
Keep your response concise and professional."""
```

### 2. Zero-Shot Chain-of-Thought

```python
"""You are an empathetic customer service assistant.

For each customer message, think step-by-step:
1. What is the customer's emotional state?
2. What is their main concern or request?
3. What would be the most helpful response?

Let's work through this step by step, then provide your final response."""
```

### 3. Few-Shot Chain-of-Thought

Includes 3 examples demonstrating:

- Very negative sentiment (product failure, refund request)
- Neutral sentiment (warranty inquiry)
- Very positive sentiment (appreciation)

## Information

- **Course**: MSc CSAI - Core Topics AI
- **Institution**: Tilburg University
- **Date**: November 2024
