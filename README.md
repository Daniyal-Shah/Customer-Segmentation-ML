# Mall Customer Segmentation

Customer segmentation pipeline for marketing campaign data. The project preprocesses customer features, applies clustering algorithms, and evaluates cluster quality with standard metrics.

## Dataset

The project uses the **Marketing Campaign** dataset (`data/marketing_campaign.csv`), which contains customer demographics, purchase behavior, and campaign response history. The file is tab-separated.

## Project Structure

```
.
├── data/
│   └── marketing_campaign.csv   # Raw dataset
├── notebooks/
│   └── experiments.ipynb        # Exploratory analysis and experiments
├── src/
│   ├── preprocessing.py         # Feature engineering and sklearn preprocessor
│   ├── train.py                 # Training, clustering, and evaluation
│   └── predict.py               # Inference (placeholder)
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train clustering models and print evaluation metrics:

```bash
python src/train.py
```

This script will:

1. Load and preprocess the marketing campaign data
2. Engineer features (e.g. `Age`, `Customer_Since`)
3. Scale numeric columns and one-hot encode categorical columns
4. Run the elbow method to explore K-Means cluster counts
5. Fit **K-Means** and **Agglomerative Clustering** (3 clusters)
6. Visualize K-Means clusters with PCA
7. Print silhouette, Calinski-Harabasz, and Davies-Bouldin scores

## Preprocessing

`src/preprocessing.py` handles:

- Dropping unused columns (`ID`, `Z_CostContact`, `Z_Revenue`, `Response`)
- Date parsing and feature engineering from `Dt_Customer` and `Year_Birth`
- Missing value imputation (median for numeric, mode for categorical)
- Building a `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`

## Models

| Model | Description |
|-------|-------------|
| K-Means | Partition-based clustering with 3 clusters |
| Agglomerative Clustering | Hierarchical clustering with 3 clusters |

Cluster quality is compared using:

- **Silhouette Score** — higher is better
- **Calinski-Harabasz Score** — higher is better
- **Davies-Bouldin Score** — lower is better

## Dependencies

- pandas
- scikit-learn
- matplotlib

## Notes

- Run commands from the project root so data paths resolve correctly.
- `plt.show()` in `train.py` opens an interactive plot window; use a local environment with a display to view it.
- `src/predict.py` is reserved for future inference logic.
