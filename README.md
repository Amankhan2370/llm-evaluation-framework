<div align="center">

# 🔍 LLM Evaluation & Hallucination Detection Framework

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-green?style=flat-square)]()

**Production-grade framework for evaluating LLM outputs, detecting hallucinations, and measuring grounding quality**

*Built for reliability engineers, ML practitioners, and production validation*

---

</div>

## 🎯 Purpose

This framework provides **systematic, explainable evaluation** of Large Language Model (LLM) and Retrieval-Augmented Generation (RAG) system outputs. It implements explicit checks for hallucination detection, grounding quality assessment, and adversarial robustness testing.

### What This Framework Evaluates

- **Hallucinations**: Detects when LLM outputs contain information not supported by context
- **Grounding Quality**: Measures how well answers are grounded in provided context
- **Faithfulness**: Assesses factual accuracy and citation validity
- **Robustness**: Tests system behavior under adversarial and stress conditions

---

## 🔬 Evaluation Methodology

### Hallucination Detection

The framework implements **four explicit checks** for hallucination detection:

1. **Answer-Context Overlap**: Semantic similarity between answer and retrieved context
   - Threshold: Configurable (default: 0.7)
   - Method: Sentence transformer embeddings with cosine similarity
   - Explainable: Returns overlap score and threshold comparison

2. **Citation Validation**: Verifies presence and format of citations
   - Checks: Citation markers, expected vs. found citations
   - Configurable: Can require citations or make optional
   - Output: Validation status with specific failure reasons

3. **Unsupported Claim Detection**: Identifies claims not supported by context
   - Method: Sentence-level semantic analysis
   - Threshold: Configurable similarity threshold
   - Output: List of unsupported claim sentences

4. **Retrieval Confidence**: Validates confidence scores from retrieval system
   - Checks: Minimum and average confidence thresholds
   - Configurable: Per-threshold validation
   - Output: Confidence analysis with low-confidence warnings

### Grounding Scoring

Grounding quality is measured through:

- **Answer-Context Alignment**: Semantic alignment score (0-1)
- **Context Coverage**: Percentage of context covered by answer
- **Retrieval Quality**: Weighted score based on retrieval confidence
- **Citation Quality**: Validation of citation presence and relevance

### Adversarial Testing

Generates and tests against:

- **Leading Questions**: Prompts designed to elicit confirmation bias
- **Negative Framing**: Questions phrased to test contradiction handling
- **Ambiguous Phrasing**: Vague prompts to test interpretation
- **Overly Specific Requests**: Detailed prompts testing precision
- **Contradictory Instructions**: Conflicting requirements testing robustness

---

## 📋 Requirements

- Python 3.10+
- 8GB+ RAM (for sentence transformers)
- API keys for LLM providers (if evaluating against live models)
- Optional: GPU for faster embedding computation

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Amankhan2370/llm-evaluation-framework.git
cd llm-evaluation-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

Create a `.env` file with required variables:

```env
# Required: LLM Configuration
OPENAI_API_KEY=ADD_YOUR_OWN_OPENAI_API_KEY
EVAL_MODEL_NAME=ADD_YOUR_OWN_MODEL_NAME

# Required: Dataset Path
EVAL_DATASET_PATH=data/samples/eval_dataset.json

# Optional: Thresholds
HALLUCINATION_THRESHOLD=0.7
GROUNDING_THRESHOLD=0.6
```

### Running Evaluation

```bash
# Using script
./scripts/run_eval.sh

# Or directly
python main.py
```

---

## 📊 Dataset Format

### JSON Format

```json
[
  {
    "prompt": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "context": "Paris is the capital and largest city of France.",
    "citations": ["[1]"],
    "retrieval_scores": [0.95]
  },
  {
    "prompt": "Explain quantum computing",
    "answer": "Quantum computing uses quantum mechanics...",
    "context": "Quantum computing is a computing paradigm..."
  }
]
```

### CSV Format

| prompt | answer | context | citations | retrieval_scores |
|--------|--------|---------|-----------|------------------|
| What is X? | Answer text | Context text | [1] | [0.95] |

---

## 📈 Evaluation Output

### Example Results

```json
{
  "summary": {
    "total_evaluations": 100,
    "hallucination": {
      "mean": 0.23,
      "median": 0.18,
      "p95": 0.67,
      "p99": 0.89
    },
    "grounding": {
      "mean": 0.82,
      "median": 0.85,
      "p95": 0.95
    },
    "hallucination_rate": 0.12,
    "pass_rate": 0.88
  },
  "results": [
    {
      "prompt": "...",
      "answer": "...",
      "hallucination": {
        "hallucination_score": 0.15,
        "is_hallucination": false,
        "checks": {
          "answer_context_overlap": {
            "score": 0.89,
            "passed": true
          }
        }
      },
      "grounding": {
        "grounding_score": 0.87,
        "faithfulness_score": 0.85
      },
      "overall_score": 0.86
    }
  ]
}
```

### Metrics Explained

| Metric | Description | Range |
|--------|-------------|-------|
| **hallucination_score** | Likelihood of hallucination | 0.0 (no hallucination) to 1.0 (severe hallucination) |
| **grounding_score** | Quality of grounding in context | 0.0 (ungrounded) to 1.0 (fully grounded) |
| **faithfulness_score** | Factual accuracy score | 0.0 to 1.0 |
| **overall_score** | Combined evaluation score | 0.0 to 1.0 |
| **hallucination_rate** | Percentage of outputs with hallucinations | 0.0 to 1.0 |
| **pass_rate** | Percentage passing quality threshold | 0.0 to 1.0 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│     Evaluation Runner                │
│  (Orchestrates evaluation flow)      │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼──────────┐   ┌──────▼──────────┐
│ Hallucination│   │  Grounding      │
│  Detector    │   │   Scorer        │
└───┬──────────┘   └──────┬───────────┘
    │                     │
    ├─ Answer-Context     ├─ Alignment
    │  Overlap            │
    ├─ Citation           ├─ Coverage
    │  Validation         │
    ├─ Unsupported        ├─ Retrieval
    │  Claims             │  Quality
    └─ Retrieval         └─ Citations
       Confidence
```

### Component Responsibilities

- **Evaluation Runner**: Coordinates evaluation flow, manages async processing
- **Hallucination Detector**: Implements explicit hallucination checks
- **Grounding Scorer**: Calculates grounding and faithfulness metrics
- **Adversarial Tester**: Generates and analyzes adversarial variants
- **Metrics Aggregator**: Computes summary statistics and rates

---

## 🔧 Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HALLUCINATION_THRESHOLD` | 0.7 | Score above which output is considered hallucinated |
| `GROUNDING_THRESHOLD` | 0.6 | Minimum grounding score for acceptance |
| `CONTEXT_OVERLAP_THRESHOLD` | 0.5 | Minimum answer-context overlap |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | 0.7 | Minimum retrieval confidence |
| `CITATION_REQUIRED` | true | Whether citations are mandatory |
| `ADVERSARIAL_ENABLED` | true | Enable adversarial testing |
| `BATCH_SIZE` | 10 | Examples per batch |
| `MAX_CONCURRENT_EVALS` | 5 | Concurrent evaluation limit |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=evaluation --cov=metrics --cov-report=html

# Specific test
pytest tests/test_evaluation.py::test_hallucination_detection
```

---

## ⚠️ Known Limitations

1. **Semantic Similarity**: Uses sentence transformers; may not capture all nuances
2. **Citation Parsing**: Relies on regex patterns; may miss non-standard formats
3. **Unsupported Claims**: Detection based on semantic similarity; false positives possible
4. **Adversarial Testing**: Variants are template-based; may not cover all edge cases
5. **Batch Processing**: Memory usage scales with batch size and model size

### Failure Modes

- **Low Context Quality**: Poor context leads to false positives in hallucination detection
- **Ambiguous Prompts**: Framework may flag legitimate interpretations as hallucinations
- **Model-Specific**: Some checks may not generalize across all LLM architectures
- **Language**: Optimized for English; performance may vary for other languages

---

## 📁 Project Structure

```
llm-evaluation-framework/
├── evaluation/
│   ├── runner.py           # Main evaluation orchestrator
│   ├── hallucination.py    # Hallucination detection logic
│   ├── grounding.py       # Grounding quality scoring
│   └── adversarial.py     # Adversarial testing
├── metrics/
│   └── scoring.py         # Metrics aggregation
├── pipelines/
│   └── batch_eval.py      # Batch evaluation pipeline
├── config/
│   └── settings.py        # Configuration management
├── data/
│   └── samples/           # Sample datasets
├── reports/               # Evaluation results
├── tests/                 # Test suite
├── scripts/
│   └── run_eval.sh        # Evaluation script
└── main.py               # Entry point
```

---

## 🎓 Use Cases

- **Production Validation**: Validate LLM outputs before deployment
- **Benchmarking**: Compare different models or configurations
- **Reliability Analysis**: Identify failure modes and edge cases
- **Quality Assurance**: Ensure outputs meet quality thresholds
- **Research**: Study hallucination patterns and grounding behavior

---

## 📝 License

**Proprietary** - All rights reserved.

This software and associated documentation are proprietary and confidential. Unauthorized use is prohibited.

---

<div align="center">

**For questions or issues, please open an issue on GitHub.**

[Repository](https://github.com/Amankhan2370/llm-evaluation-framework) • [Issues](https://github.com/Amankhan2370/llm-evaluation-framework/issues)

</div>
