# 🎬 Movie Recommendation System

A collaborative-filtering project that recommends **unseen movies** to
users based on the MovieLens 100K ratings dataset. It compares a
popularity baseline with user-based collaborative filtering, item-based
collaborative filtering, and truncated Singular Value Decomposition
(SVD).

> **Project format:** Jupyter Notebook\
> **Dataset:** MovieLens 100K\
> **Primary evaluation metric:** Precision@K

## 📌 Overview

Movie recommendation systems help users discover films that match their
preferences. This project builds and evaluates several recommendation
approaches using historical user ratings.

The workflow includes data loading and exploration, per-user
train/validation/test splitting, user--item matrix construction, model
tuning on validation data, and a final comparison on a held-out test
set.

## ✨ Features

-   Loads the MovieLens 100K dataset from a local folder or downloads it
    from GroupLens when it is not found.
-   Explores rating distributions, user activity, movie activity, and
    matrix sparsity.
-   Splits ratings per user into training, validation, and test sets.
-   Excludes movies already rated by a user from recommendation results.
-   Compares four approaches:
    -   Popularity-based baseline with damped average ratings
    -   User-based collaborative filtering using cosine similarity
    -   Item-based collaborative filtering using adjusted cosine
        similarity
    -   Matrix factorization using truncated SVD
-   Tunes model hyperparameters on validation data.
-   Evaluates recommendations using Precision@K and Recall@K.
-   Produces personalized top-N recommendations and similar-movie
    results.

## 🧰 Tech Stack

-   **Python**
-   **Jupyter Notebook**
-   **NumPy** --- numerical computation
-   **pandas** --- data manipulation
-   **Matplotlib** --- exploratory plots and model comparisons
-   **SciPy** --- truncated SVD
-   **scikit-learn** --- cosine similarity

## 📊 Dataset

This project uses **MovieLens 100K**, which contains:

  Property           Value
  -------------- ---------
  Ratings          100,000
  Users                943
  Movies             1,682
  Rating scale        1--5

The notebook expects the `u.data` and `u.item` files from the `ml-100k`
dataset. It searches several common locations and downloads the official
GroupLens archive if the files cannot be found.

Dataset source: [GroupLens --- MovieLens
100K](https://grouplens.org/datasets/movielens/100k/)

Please review and comply with the dataset provider's terms before using
or redistributing the data.

## 🧠 Recommendation Methods

### 1. Popularity Baseline

Ranks movies using a damped average rating. Movies with fewer ratings
are pulled toward the global average, reducing the influence of a small
number of unusually high ratings. This baseline is not personalized.

### 2. User-Based Collaborative Filtering

Finds users with similar rating patterns using cosine similarity on
mean-centered ratings. Ratings from the most similar neighbors are
combined to estimate a user's preference for unseen movies.

### 3. Item-Based Collaborative Filtering

Computes movie-to-movie similarities using mean-centered ratings. It
predicts a user's potential rating for a movie from the user's ratings
of similar movies.

### 4. Matrix Factorization (SVD)

Uses truncated Singular Value Decomposition to approximate the
mean-centered user--item matrix with a lower-dimensional representation.
The number of latent factors is tuned using the validation set.

## 🧪 Evaluation Strategy

To reduce the risk of tuning against the final evaluation data, the
notebook uses separate per-user splits:

1.  **Test set:** 20% of each user's ratings are held out.
2.  **Validation set:** 10% of the remaining ratings are held out for
    hyperparameter tuning.
3.  **Training set:** the remaining ratings are used to fit models.

A held-out movie is considered **relevant** when its rating is **4 or
higher**.

### Metrics

-   **Precision@K:** fraction of the top-K recommended movies that are
    relevant.
-   **Recall@K:** fraction of the user's relevant held-out movies that
    appear in the top-K recommendations.

The notebook reports metrics at **K = 5, 10, and 20**. Compare
personalized models with the popularity baseline; absolute Precision@K
values depend on the dataset and evaluation setup.

## 🚀 Getting Started

### Prerequisites

Install Python 3.9 or later and Jupyter Notebook/JupyterLab. The
notebook uses the following libraries:

``` bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your repository URL and folder name.

### 2. Launch Jupyter

``` bash
jupyter notebook
```

### 3. Open and run the notebook

Open `movie_recommendation_system.ipynb` and run the cells from top to
bottom.

The notebook attempts to locate the dataset in common paths, including:

-   `./`
-   `./ml-100k/`
-   `./data/ml-100k/`
-   `./data/`

If the dataset is already downloaded, place the `ml-100k` directory next
to the notebook or in one of the supported locations. Otherwise, the
notebook attempts to download it from GroupLens, so an internet
connection is required for that first download.

## 🎯 Getting Recommendations

The notebook includes a `recommend_for_user` helper that returns the
highest-scoring unseen movies for a given user. It also demonstrates
side-by-side recommendations from the different models.

The example user can be changed through the notebook's `demo_user`
variable. MovieLens user IDs range from **1 to 943**.

## 📁 Project Structure

``` text
.
├── movie_recommendation_system.ipynb   # Data exploration, training, evaluation, recommendations
└── README.md                            # Project documentation
```

The dataset files may be stored locally in an `ml-100k/` or `data/`
directory; they do not need to be committed to the repository.

## ⚠️ Notes and Limitations

-   The user--item matrix is sparse: most users have not rated most
    movies.
-   The notebook's SVD approach fills missing entries with zero after
    mean-centering; this is a practical demonstration, but it is not
    equivalent to every specialized recommender-system
    matrix-factorization method.
-   Recommendations are based on ratings, not movie content. The current
    workflow does not use genres, descriptions, or user demographics as
    model features.
-   Results can vary with data splits and hyperparameter choices. The
    notebook uses a fixed random seed to improve reproducibility.
-   Exact metric values are intentionally not listed here because they
    depend on the results produced when the notebook is run.

## 🔮 Future Improvements

-   Add a hybrid recommender that combines ratings with movie genres or
    user demographics.
-   Explore methods such as ALS or an SGD-trained SVD model that
    optimize observed ratings directly.
-   Add ranking metrics such as NDCG@K or MAP@K.
-   Evaluate rating-prediction error with RMSE and MAE.
-   Package the recommendation functions into a reusable Python module
    or deploy them through an API or web interface.
