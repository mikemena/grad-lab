# grad-lab

# activate env

`source venv/bin/activate`

# Run data pre-processing:
# Run from project root: cd /Users/mike/Documents/red/grad-lab
`python src/data/prepare_data.py --config configs/pipeline_v1.yaml`

# Run train model - PyTorch
`python src/train_model.py --config configs/model_v1.yaml`

# Run train model - Tree-Based
`python src/train_tree_models.py --config configs/model_v1.yaml`

# Predict with custom input

`python src/predict.py --config configs/model_v1.yaml --input experiments/preprocessing/debug_splits/raw_test_split.xlsx`

# Compare runs
`python src/compare_runs.py --dir evaluation_results`

# Run streamlit model run comparison with streamlit
`streamlit run src/model_comparison_ui.py`

# Run mlflow
`mlflow ui`
