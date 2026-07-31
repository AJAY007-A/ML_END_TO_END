# Car Price Prediction - Machine Learning Pipeline & Web App

A production-grade, end-to-end Machine Learning pipeline and Flask Web Application for predicting secondhand car prices based on vehicle specs, usage, and attributes.

## Project Architecture

```
ML_Project/
│
├── artifacts/              # Generated models, preprocessors, datasets, & metrics
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   ├── test.csv
│   ├── raw.csv
│   └── evaluation_metrics.json
│
├── Dataset/                # Raw source dataset
│   └── cardekho_dataset.csv
│
├── Notebook/               # Exploratory and training notebooks
│   ├── Model.ipynb
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   └── Experiments.ipynb
│
├── src/                    # Production codebase
│   ├── __init__.py
│   ├── exception.py        # Custom exception handler
│   ├── logger.py           # Logging module
│   ├── config.py           # Config dataclasses
│   ├── utils.py            # Helper utilities
│   │
│   ├── components/         # Modular pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   └── pipeline/           # Pipeline execution & prediction modules
│       ├── __init__.py
│       ├── training_pipeline.py
│       └── prediction_pipeline.py
│
├── templates/              # HTML Frontend
│   └── home.html
│
├── static/                 # Static styles & assets
│   └── css/
│       └── style.css
│
├── app.py                  # Flask Web Server
├── requirements.txt        # Python package dependencies
├── setup.py                # Package installation configuration
├── .env                    # Environment variables
├── README.md               # Documentation
└── LICENSE                 # MIT License
```

## Quick Start

### 1. Installation

Install dependencies and set up the package in editable mode:

```bash
pip install -r requirements.txt
```

### 2. Run the Training Pipeline

Execute the full machine learning training pipeline:

```bash
python src/pipeline/training_pipeline.py
```

This will ingest data, validate schema, transform features, benchmark models (Linear Regression, Ridge, Lasso, Random Forest), pick the best regressor, save artifacts in `artifacts/`, and write evaluation metrics.

### 3. Launch Flask Web Application

Start the local Flask development server:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000` to interact with the Car Price Predictor web UI.

## Model Performance Summary

- **Primary Regressor**: Random Forest Regressor
- **Preprocessors**: `StandardScaler` (numerical features) & `OneHotEncoder` (categorical features)
- **R2 Score**: ~0.90+
"# ML_END_TO_END" 
