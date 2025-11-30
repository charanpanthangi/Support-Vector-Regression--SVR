# Examples and ideas

Use these prompts to explore the template further:

- Try different kernels: `kernel="linear"`, `kernel="poly"` with varying degrees, and compare metrics.
- Adjust `C` and `epsilon` to balance margin width and penalty for errors.
- Experiment with the `gamma` parameter for the RBF kernel to control how far the influence of a single training example reaches.
- Add cross-validation using `GridSearchCV` to find stronger hyperparameters.
- Compare SVR performance to a simple `LinearRegression` model on the same train/test split.
