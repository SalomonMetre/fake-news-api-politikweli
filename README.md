Here is a professional, production-ready `README.md` for your project. It highlights the architecture, the `uv` workflow, and the specific model/dataset attributions you requested.

You can save this as `README.md` in your project root.

-----

# Fake News Detection API

A high-performance, asynchronous REST API for detecting misinformation in news text. This service uses **FastAPI** to serve a **DistilBERT** model fine-tuned on the **PolitiKweli** dataset.

It features built-in **LRU caching** and utilizes **`uv`** for modern, lightning-fast Python dependency management.

## 🚀 Features

  * **BERT-based Inference:** Uses `lusamaki/distilbert_fine_tuned_fake_news_detection_model` for high-accuracy classification.
  * **Performance Optimized:**
      * **Model Lifespan:** Loads the heavy ML model only once at startup.
      * **Smart Caching:** Uses `async-lru` to instantly return results for repeated queries without hitting the model.
  * **Simple Interface:** Returns a clean JSON response with `label` (Real/Fake) and `confidence` score.
  * **Modern Stack:** Built with FastAPI and managed by `uv`.

## 🛠️ Tech Stack

  * **Framework:** FastAPI
  * **ML Engine:** Hugging Face Transformers / PyTorch
  * **Package Manager:** uv
  * **Caching:** async-lru

## 📋 Model & Dataset

This API serves a model trained on specific political fact-checking data:

  * **Model:** [DistilBERT Fine-tuned Fake News Detection](https://www.google.com/search?q=https://huggingface.co/lusamaki/distilbert_fine_tuned_fake_news_detection_model)
  * **Dataset:** **PolitiKweli** - A dataset for fake news detection in political contexts.
  * **Citation:** *Amol, 2023. PolitiKweli Dataset.*

## ⚡ Quick Start

This project uses `uv` for dependency management, so you do not need to manually create or activate virtual environments.

### 1\. Prerequisites

Ensure you have **uv** installed:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2\. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/fake-news-api.git
cd fake-news-api

# Install dependencies into the project's virtual environment
uv sync
```

*Note: If you are starting from scratch without a lockfile, run `uv add fastapi uvicorn transformers torch async-lru`.*

### 3\. Run the Server

Start the API using `uv run`. This automatically uses the correct environment.

```bash
uv run uvicorn main:app --reload
```

*The first run may take a few seconds to download the 260MB model file.*

## 🔌 API Usage

### Predict Endpoint

**POST** `/predict`

Accepts a JSON body containing the news text and returns the classification.

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The government has secretly announced that gravity will be turned off next Tuesday."}'
```

#### Response

```json
{
  "label": "Fake",
  "confidence": 0.99926
}
```

## 📂 Project Structure

```text
.
├── main.py            # Application entry point and logic
├── pyproject.toml     # Dependencies managed by uv
├── uv.lock            # Locked dependency versions
└── README.md          # Documentation
```

## 🧪 Testing

You can verify the API is running correctly using the health check endpoint:

```bash
curl http://127.0.0.1:8000/health
# Output: {"status": "running", "model_loaded": true}
```

## 📜 License

[MIT](https://choosealicense.com/licenses/mit/)
