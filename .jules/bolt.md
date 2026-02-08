## 2026-02-08 - Batch Prediction vs Single-Row Prediction
**Learning:** Massive performance overhead when predicting probabilities for single rows in a loop using `scikit-learn` and `pandas`.
**Action:** Always batch predictions for the entire dataset upfront if possible, especially when the model and features are static during the loop. The speedup was ~40,000x for the prediction function call. Also, using vectorized `np.clip` on the entire array is cleaner and faster than clipping inside a Python loop.
