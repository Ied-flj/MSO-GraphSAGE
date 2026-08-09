GraphSAGE for HRRSI Land Cover Classification

This repository provides a TensorFlow and StellarGraph implementation of GraphSAGE for High-Resolution Remote Sensing Image (HRRSI) object classification. It models spatial land cover data as a graph network to perform accurate node classification, and is specifically configured for baseline and ablation experiments.
Dependencies
Ensure you have the following libraries installed. It is recommended to use a virtual environment (e.g., Conda) with Python 3.7 - 3.9, given the StellarGraph requirements.
```bash
pip install tensorflow==2.x
pip install stellargraph
pip install pandas numpy scikit-learn matplotlib
```

Dataset Structure

The script expects the data to be placed in a designated directory (e.g., `E:\Bigpaper\Landcover\`). You will need to prepare two main CSV files:
Node Features & Labels (`Image2光谱归一化.csv`):
Must contain a `FID` column (used as the node index).
Must contain a `gridcode` column (the ground truth class labels).
Remaining columns should be the normalized spectral and spatial features.
Edge List (`Image2edge15.csv`):
A two-column CSV (no header required) representing the `source` and `target` nodes to construct the network.

Usage

Configure Data Paths: Update the data loading paths in the script to point to your local dataset directories.
Set GPU: The script defaults to using GPU `0`. Adjust `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` if you are running on a CPU or a different GPU.
Run the Script:
    ```bash
    python graphsage_classification.py
    ```

Outputs

Training History: Plots the loss and accuracy curves over the epochs.
Metrics: Outputs Accuracy, Precision, and Custom F1-Score on the test set.
Predictions: Exports the final test set predictions to a CSV file (e.g., `data2-GraphSAGE.csv`) for further analysis or ablation comparisons.
Extensibility
The codebase includes commented-out sections that can be easily enabled to:
Extract and save the intermediate GraphSAGE node embeddings (`GraphSAGEemb-64-linear.csv`).
Output predictions for the entire graph rather than just the test set.
