# HeartWise ❤️

An educational heart disease prediction prototype built with Python, Streamlit, Pandas, Scikit-learn, and Plotly.

## Features
- Interactive health indicator form
- Machine learning prediction
- Dataset explorer
- Interactive charts
- Model evaluation metrics

## Setup (Windows PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download and prepare the dataset:

```powershell
python -m src.download_data
```

Train and evaluate the model:

```powershell
python -m src.train
```

Launch the website:

```powershell
python -m streamlit run app.py
```

## Dataset
UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/45/heart+disease

## Disclaimer
This is an educational project, not a validated medical device. Predictions must not be used for diagnosis or treatment decisions.
