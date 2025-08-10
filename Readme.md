# DSPI: Distribution-Aware Spatial Pivot Learned Index

## Introduction
This code repository contains the implementation of **DSPI** (Distribution-Aware Spatial Pivot Learned Index), a spatial learned index that leverages a self-attention mechanism to model the spatial distribution of data and uses a Transformer to jointly optimise pivot placement and pruning efficiency.  
DSPI supports **efficient range queries** and **k-nearest neighbour (kNN) queries** via a lightweight function mapping model, achieving high pruning rates and low query latency.

**Key Components in DSPI**:
- **Data-Aware Pivot Selection**: Automatically selects optimal pivots based on spatial data distribution.
- **Transformer-based Pruning Model**: Learns to prune irrelevant partitions in range and kNN queries.
- **Lightweight Mapping Model**: Efficiently maps query coordinates to candidate partitions.
- **Query Algorithms**:
  - **Encoding-based Range Query**: Uses pivot-encoded distances for early pruning.
  - **Adaptive kNN Query**: Dynamically adjusts search range based on candidate density.

---

## Getting Started

### Source Code Info
We implement DSPI in **Python 3.11** on **Ubuntu**.  
All dependencies are listed in `requirements.txt`.  
The code can be found in this repository, organised into modules for **data processing**, **model training**, and **query execution**.

---

## Content

**data/**  
- **raw/**: Raw datasets, e.g., [NYC Open Data -](https://opendata.cityofnewyork.us/)
- **processed/**: Preprocessed datasets ready for indexing, e.g., `Skewed_4d_100_processed_data.pkl`.  
- **query/**: Query files, e.g.,  `testpoint4d.txt`.

**scripts/**  

- **DSPI.py**: Main entry script for training, building, and querying DSPI.  

**src/dspi/**  
- **algorithms/**: Core query and update algorithms, including:
  - `rangequery.py`: Range query execution.
  - `knnquery.py`: kNN query execution.
  - `update.py`: Updating index structures.
- **core/**: Basic data structures like `Point.py` and `Reference.py`.
- **models/**: Learning models including:
  - `NN.py`: Neural network regression model.
  - `PolynomialRegression.py`: Polynomial regression baseline.
- **utils/**: Constants and helper functions.

**requirements.txt**  
- Lists Python dependencies for environment setup.

---

## Running Test Cases for DSPI

```bash
pip install -r requirements.txt

# Build index
python scripts/DSPI.py \
  --data data/processed/Skewed_4d_100_processed_data.pkl \
  --mode build \
  --out runs/dspi_index.pt

# Run range query
python scripts/DSPI.py \
  --data data/processed/Skewed_4d_100_processed_data.pkl \
  --mode range \
  --query data/query/testpoint4d.txt \
  --index runs/dspi_index.pt \
  --radius 0.05

# Run kNN query
python scripts/DSPI.py \
  --data data/processed/Skewed_4d_100_processed_data.pkl \
  --mode knn \
  --query data/query/testpoint4d.txt \
  --index runs/dspi_index.pt \
  --k 10
```

