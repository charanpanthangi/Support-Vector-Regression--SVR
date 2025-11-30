# Support Vector Regression (SVR) Template

A beginner-friendly, end-to-end template for Support Vector Regression (SVR) using scikit-learn. The project demonstrates how to load data, preprocess features, train kernel-based SVR models, evaluate performance, and visualize predictions.

## What is SVR?
Support Vector Regression adapts the ideas of Support Vector Machines to regression tasks. Instead of trying to classify points, SVR finds a function that fits within an **epsilon-insensitive margin** around the data. Points inside the margin are considered correctly predicted, while points outside incur a loss.

### Why the ε-insensitive margin matters
- Predictions within ±ε of the target are treated as perfect, encouraging a simpler model.
- Points outside the tube influence the model through support vectors.
- The margin helps control overfitting and balances complexity with tolerance for small errors.

## Why kernels matter
SVR can model non-linear relationships using kernels:
- **Linear**: Fast baseline when relationships are mostly linear.
- **Polynomial**: Captures curved relationships using polynomial features.
- **RBF (Gaussian)**: Flexible default choice that fits many shapes by measuring similarity with distances.

## Why scaling is critical
SVR relies on distance calculations for kernels. Features with large scales dominate distance computations and harm performance. Scaling with `StandardScaler` (zero mean, unit variance) keeps all features balanced and stable for optimization.

## When to use SVR
- Medium-sized datasets with complex but smooth relationships
- When you want strong performance without extensive feature engineering
- When residuals show non-linear patterns that linear regression misses

## Dataset
The project uses the scikit-learn **California Housing** dataset, which predicts median house values from eight input features such as income, house age, and latitude/longitude.

## Project structure
```
app/
  data.py          # Load the California Housing dataset
  preprocess.py    # Train/test split and scaling helpers
  model.py         # Build kernel-based SVR models (RBF, poly, linear)
  evaluate.py      # Compute regression metrics (MSE, MAE, RMSE, R^2)
  visualize.py     # Plot actual vs. predicted values
  main.py          # Orchestrate the full training and evaluation pipeline
notebooks/
  demo_svr.ipynb   # Walkthrough notebook with explanations and plots
examples/
  README_examples.md  # Quick ideas for extending the template
tests/             # Lightweight pytest checks
requirements.txt   # Python dependencies
Dockerfile         # Container entrypoint that runs the pipeline
```

## How the pipeline works
1. **Load data** with `fetch_california_housing`.
2. **Split** into train/test sets.
3. **Scale** features using `StandardScaler` (mandatory for SVR).
4. **Build** an SVR model wrapped in a scikit-learn `Pipeline`.
5. **Train** on the training data.
6. **Evaluate** with MSE, MAE, RMSE, and R².
7. **Visualize** predictions against actual values (SVG plot).

## Quickstart
### Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

### Run tests
```bash
pytest
```

### Launch the notebook
```bash
jupyter notebook notebooks/demo_svr.ipynb
```

## Docker
Build and run the container:
```bash
docker build -t svr-template .
docker run --rm svr-template
```

## Future extensions
- Hyperparameter tuning (grid search for C, epsilon, gamma, and degree)
- Kernel experiments (sigmoid kernel, custom kernels)
- SVR vs. Linear Regression comparison and learning curves

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
