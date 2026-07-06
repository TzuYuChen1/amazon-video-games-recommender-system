# Amazon Video Games Recommender System — Matrix Factorization & Neural Collaborative Filtering

**Building and evaluating a rating-prediction recommender system on large-scale Amazon review data**

📍 University of Minnesota · Carlson School of Management · MSBA Program · MSBA 6351 AI for Personalized Recommendations

---

## Executive Summary

This project implements and compares two recommender system architectures — Matrix Factorization (MF) and Neural Collaborative Filtering (NCF) — on the Amazon Video Games review dataset, covering 500,000 user-product interactions across 400,000+ users and 63,000 video games.

The pipeline covers strategy memo, data processing, temporal train-test split, cold-start analysis, model development, evaluation, and segmentation-based deep dive. NCF substantially outperforms MF (RMSE 1.59 vs. 3.99), with both models' limitations largely driven by a severe cold-start problem: 96.8% of test instances involve users or items unseen during training.

**Key deliverables:**

- Strategy memo recommending feasible recommendation approaches for the dataset
- Temporal train-test split preserving chronological order
- Cold-start problem quantification across users and items
- Matrix Factorization implementation using TensorFlow/Keras embeddings
- Neural Collaborative Filtering implementation with deeper non-linear architecture
- Segmentation-based evaluation across user activity and item popularity tiers

---

## Results

| Model | RMSE | MAE |
|---|---|---|
| Matrix Factorization (MF) | 3.9878 | 3.6647 |
| **Neural Collaborative Filtering (NCF)** | **1.5905** | **1.2599** |

**Cold-start impact:** 96.8% of test instances involve unseen users or items, which is the primary driver of MF's elevated error. NCF's deeper architecture captures non-linear interaction patterns that generalize better under these conditions, though both models remain constrained by sparse training signal for cold-start cases.

---

## Repository Structure

```text
.
├── recommender_system.ipynb            # Main notebook — full pipeline from data to evaluation
├── data/
│   └── README.md                       # Dataset description & download instructions
└── README.md
```

---

## Dataset

**Amazon Reviews 2023 — Video Games**

| Attribute | Value |
|---|---|
| Source | [UCSD Amazon Reviews 2023](https://cseweb.ucsd.edu/~jmcauley/datasets.html) |
| Interactions | ~500,000 user-product ratings |
| Users | 400,000+ |
| Items | 63,000 video games |
| Rating scale | 1–5 stars |
| Additional fields | Timestamps, verified purchase flags, helpful votes |

Raw data is not included in this repository. See [`data/README.md`](data/README.md) for download instructions.

---

## Notebook Walkthrough

### Strategy Memo

Before building the models, a recommendation strategy memo is written from the perspective of an AI strategy consultant. Based on the dataset's rating-centric structure, the memo recommends a preference-based collaborative filtering approach and identifies the cold-start problem as the primary deployment risk.

### Data Processing

- Retains only the columns needed for recommendation: `user_id`, `parent_asin` (renamed to `item_id`), `rating`, `timestamp`
- Drops rows with missing values in key columns
- Encodes `user_id` and `item_id` to integer indices for embedding lookup

### Train-Test Split

Data is split temporally (80/20) to simulate a real deployment setting where the model trains on historical interactions and is evaluated on future ones. This preserves chronological order rather than randomly shuffling.

### Cold-Start Analysis

Quantifies the overlap between training and test sets:

- Users in test but not in training: **93,806**
- Items in test but not in training: **41,978**
- 96.8% of test instances involve at least one unseen user or item

This analysis contextualizes model performance — high RMSE is expected when the majority of test cases are structurally unanswerable by collaborative filtering alone.

### Model Development

**Matrix Factorization (MF)**

Implemented using TensorFlow/Keras. Each user and item is represented by a learned embedding vector; the predicted rating is the dot product of the two embeddings. Trained with MSE loss and Adam optimizer.

**Neural Collaborative Filtering (NCF)**

Extends MF by concatenating user and item embeddings and passing them through multiple dense layers with ReLU activations. The deeper architecture captures non-linear interaction patterns that MF's dot-product cannot represent.

### Evaluation

Both models are evaluated on RMSE and MAE on the held-out test set. Five sample predictions are generated alongside ground-truth ratings for qualitative inspection.

| Model | RMSE | MAE |
|---|---|---|
| MF | 3.9878 | 3.6647 |
| **NCF** | **1.5905** | **1.2599** |

MF's performance is dominated by cold-start failure — it cannot produce meaningful predictions for users and items with no training history. NCF's improvement reflects its stronger generalization from limited signal, though both models would benefit from hybrid approaches (e.g., content-based features) to address cold-start more directly.

### Segmentation Deep Dive

The test set is segmented along two dimensions to understand where each model succeeds and fails:

**User Activity Segments**

| Segment | Definition | Size |
|---|---|---|
| New Users | 0 training interactions | 93,806 |
| Average Users | 1–5 training interactions | 5,937 |
| Power Users | > 5 training interactions | 257 |

**Item Popularity Segments**

| Segment | Definition | Size |
|---|---|---|
| New Items | 0 training interactions | 41,978 |
| Average Items | 1–10 training interactions | 21,149 |
| Popular Items | > 10 training interactions | 36,873 |

Performance is evaluated separately on each segment to identify model strengths and weaknesses, and to draw business implications for how the system would serve different user and item tiers in deployment.

---

## Setup

```bash
pip install tensorflow pandas numpy scikit-learn
```

Update `data_path` in the notebook to point to your local copy of the dataset before running.

---

## Tools & Technologies

| Layer | Technology |
|---|---|
| Data processing | Python · Pandas · NumPy |
| Modeling | TensorFlow · Keras |
| Evaluation | scikit-learn (RMSE · MAE) |

---

## Usage and License Note

This repository is shared for academic and portfolio purposes. Please contact the author before reusing or redistributing the code.
