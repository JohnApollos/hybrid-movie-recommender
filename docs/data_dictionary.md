# 📚 MovieLens Dataset & Project Data Dictionary

This document details the schema, data types, descriptions, examples, and missing value profiles for all raw data files, generated splits, and model contract outputs in the Hybrid Movie Recommender System.

---

## 1. Raw MovieLens Dataset (`data/ml-latest-small/`)

The dataset comprises **100,836 ratings** and **3,683 tag applications** across **9,742 movies** created by **610 users** between March 29, 1996 and September 24, 2018.

### 1.1 `ratings.csv` (100,836 rows)
Records explicit movie ratings given by users.

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `ratings.csv` | `userId` | Integer | Unique identifier for each MovieLens user (anonymized) | `1` | 0 (0.0%) |
| `ratings.csv` | `movieId` | Integer | Unique identifier for each rated movie | `1` | 0 (0.0%) |
| `ratings.csv` | `rating` | Float | User's rating score on a 5-star scale with 0.5-star increments [0.5, 5.0] | `4.0` | 0 (0.0%) |
| `ratings.csv` | `timestamp` | Integer | Seconds since midnight UTC on January 1, 1970 (POSIX epoch) | `964982703` | 0 (0.0%) |

### 1.2 `movies.csv` (9,742 rows)
Catalog of all movies in the dataset with titles, release years, and genres.

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `movies.csv` | `movieId` | Integer | Unique identifier for each movie | `1` | 0 (0.0%) |
| `movies.csv` | `title` | String | Full title of the movie including release year in parentheses | `Toy Story (1995)` | 0 (0.0%) |
| `movies.csv` | `genres` | String | Pipe-delimited list of genres associated with the movie; `(no genres listed)` for unclassified | `Adventure\|Animation\|Children\|Comedy\|Fantasy` | 0 (0.0%)* (34 movies have `(no genres listed)`) |

### 1.3 `tags.csv` (3,683 rows)
Free-form user-applied tag metadata describing movie plot, themes, tone, or technical qualities.

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `tags.csv` | `userId` | Integer | Unique identifier for the user who added the tag | `2` | 0 (0.0%) |
| `tags.csv` | `movieId` | Integer | Unique identifier for the movie being tagged | `60756` | 0 (0.0%) |
| `tags.csv` | `tag` | String | User-generated descriptive keyword or phrase | `funny` | 0 (0.0%) |
| `tags.csv` | `timestamp` | Integer | Seconds since UTC epoch when the tag was applied | `1445714994` | 0 (0.0%) |

### 1.4 `links.csv` (9,742 rows)
Cross-reference identifiers linking MovieLens titles to external movie databases.

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `links.csv` | `movieId` | Integer | Unique identifier for each movie in MovieLens | `1` | 0 (0.0%) |
| `links.csv` | `imdbId` | String/Int | IMDb identifier (`imdb.com/title/tt<imdbId>`) | `0114709` | 0 (0.0%) |
| `links.csv` | `tmdbId` | Float/Int | The Movie Database identifier (`themoviedb.org/movie/<tmdbId>`) | `862` | 8 (0.08%) |

---

## 2. Experimental Data Splits (`data/splits/`)

### 2.1 Leave-One-Out Ranking Splits
- **`train_loo.csv` (100,226 rows):** All interactions except the chronological latest rating for each of the 610 users.
- **`test_loo.csv` (610 rows):** The single latest rating per user, used as the ground truth evaluation target. Relevance threshold: $\ge 4.0$ (363 users).

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `train_loo.csv` | `userId` | Integer | Anonymized user ID | `1` | 0 (0.0%) |
| `train_loo.csv` | `movieId` | Integer | Rated movie ID | `1` | 0 (0.0%) |
| `train_loo.csv` | `rating` | Float | Rating score [0.5, 5.0] | `4.0` | 0 (0.0%) |
| `train_loo.csv` | `timestamp` | Integer | POSIX timestamp of rating | `964982703` | 0 (0.0%) |
| `test_loo.csv` | `userId` | Integer | Anonymized user ID (one row per user) | `1` | 0 (0.0%) |
| `test_loo.csv` | `movieId` | Integer | Held-out latest movie ID | `2492` | 0 (0.0%) |
| `test_loo.csv` | `rating` | Float | Held-out true rating score | `4.0` | 0 (0.0%) |
| `test_loo.csv` | `timestamp` | Integer | Timestamp of latest interaction | `965719662` | 0 (0.0%) |

### 2.2 Random 80/20 Rating Splits (for RMSE)
- **`train_rmse.csv` (80,668 rows):** 80% random sample stratified with seed 42.
- **`test_rmse.csv` (20,168 rows):** 20% held-out test interactions for rating prediction error evaluation.

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `train_rmse.csv` | `userId` | Integer | Anonymized user ID | `462` | 0 (0.0%) |
| `train_rmse.csv` | `movieId` | Integer | Movie ID | `1721` | 0 (0.0%) |
| `train_rmse.csv` | `rating` | Float | Rating score | `4.0` | 0 (0.0%) |
| `train_rmse.csv` | `timestamp` | Integer | Rating timestamp | `1115867946` | 0 (0.0%) |
| `test_rmse.csv` | `userId` | Integer | Anonymized user ID | `288` | 0 (0.0%) |
| `test_rmse.csv` | `movieId` | Integer | Test movie ID | `2804` | 0 (0.0%) |
| `test_rmse.csv` | `rating` | Float | Test ground-truth rating | `4.5` | 0 (0.0%) |
| `test_rmse.csv` | `timestamp` | Integer | Rating timestamp | `978465565` | 0 (0.0%) |

---

## 3. Model Output Contracts (`outputs/`)

### 3.1 Model Top-10 Recommendations (`outputs/{model}_top10.csv`)
Generated by: `popularity`, `random`, `user_cf`, `item_cf`, `svd`, `content`, `hybrid`, `ncf`. Total rows: 6,100 (610 users $\times$ 10 ranks).

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `{model}_top10.csv` | `userId` | Integer | Target user ID | `1` | 0 (0.0%) |
| `{model}_top10.csv` | `movieId` | Integer | Recommended movie ID | `318` | 0 (0.0%) |
| `{model}_top10.csv` | `rank` | Integer | Recommendation rank position (1 to 10, 1 = best) | `1` | 0 (0.0%) |
| `{model}_top10.csv` | `score` | Float | Model-specific recommendation score or probability | `4.821` | 0 (0.0%) |

### 3.2 Evaluation Summary (`outputs/metrics.csv`)
Generated by `src/evaluate.py`.

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `metrics.csv` | `model` | String | Name of the recommender model | `hybrid` | 0 (0.0%) |
| `metrics.csv` | `precision_at_10` | Float | Proportion of top-10 recommendations that are relevant hits | `0.0096` | 0 (0.0%) |
| `metrics.csv` | `recall_at_10` | Float | Proportion of relevant held-out items recovered in top-10 | `0.0964` | 0 (0.0%) |
| `metrics.csv` | `ndcg_at_10` | Float | Normalized Discounted Cumulative Gain at rank 10 | `0.0521` | 0 (0.0%) |
| `metrics.csv` | `rmse` | Float | Root Mean Square Error on 80/20 test split (rating predictors only) | `0.8831` | Blank for ranking-only models |

### 3.3 Live Feedback Log (`feedback.csv`)
Appended dynamically by user interactions in the Streamlit application.

| File | Column | Type | Description | Example | Missing Values |
|---|---|---|---|---|---|
| `feedback.csv` | `timestamp` | String | ISO 8601 formatted datetime when feedback was submitted | `2026-11-16T10:45:00` | 0 (0.0%) |
| `feedback.csv` | `userId` | Integer | User ID who submitted feedback | `1` | 0 (0.0%) |
| `feedback.csv` | `movieId` | Integer | Movie ID evaluated | `318` | 0 (0.0%) |
| `feedback.csv` | `title` | String | Movie title evaluated | `Shawshank Redemption, The (1994)` | 0 (0.0%) |
| `feedback.csv` | `model` | String | Recommendation model that produced the item | `hybrid` | 0 (0.0%) |
| `feedback.csv` | `reaction` | String | User feedback rating (`LIKE` or `DISLIKE`) | `LIKE` | 0 (0.0%) |
| `feedback.csv` | `notes` | String | Optional context or notes | `User indicated positive interest` | 0 (0.0%) |
