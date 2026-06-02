# Nutrition-MTL-Inference-Engine

An AI-powered visual nutrition estimation system and end-to-end operational pipeline. This project provides a Multi-Task Learning (MTL) architecture to predict macro-nutrients and classify food ingredients directly from images, along with a production-ready FastAPI backend for serving the model.

## AI Architecture

The core model (`nutritionmtlmodel.ipynb`) implements a **Multi-Task Learning (MTL)** approach:
- **Backbone:** A `ConvNeXt` feature extractor acts as the visual backbone.
- **Dual Heads:**
  - *Classification Head:* Multi-label ingredient classification to identify components of the dish.
  - *Regression Head:* Continuous prediction of key macro-nutrients (Calories, Fat, Carbs, Protein, and Mass).

## End-to-End Operational Pipeline

The `app/` directory contains a modularized, production-ready FastAPI backend that serves the trained MTL model.

### Project Structure
- `app/server.py`: FastAPI application setup, routes, and endpoint definitions.
- `app/pipeline.py`: The inference pipeline that handles image preprocessing, model execution, and response formatting.
- `app/model.py`: PyTorch model definitions for the MTL architecture.
- `app/data.py`: Dataset utilities and data classes.
- `app/config.py` & `settings.json`: Configuration management.
- `run.py`: The entry point to start the `uvicorn` server.

### Running the Server

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server:**
   ```bash
   python run.py
   ```
   The server will start on `http://127.0.0.1:8000`. You can test endpoints via the Swagger UI at `http://127.0.0.1:8000/docs`.

## Setup & Weights (Important)

To keep the repository lightweight, heavy model weights and datasets are excluded from source control. **You must download these artifacts and place them in the correct directories before running the server.**

1. **Model Weights:**
   - Download the primary MTL weight file: `best_nutrition_mtl_model.pth`
   - Download the visual knowledge base: `visual_knowledge_base.pt`
   - Place both files in the root of the project directory (alongside `run.py`).

2. **Datasets (Optional for Training/Evaluation):**
   - The Nutrition5k dataset or custom CSVs (e.g., `Egyptian_Food_Combined_with_Users_Ratings.csv`) should be placed in `arabic_food_dataset/` or the designated data folder as configured in `settings.json`.
