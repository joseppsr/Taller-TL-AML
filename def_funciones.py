def compute_pearson_sq(y_proba, s_array):
    """Cuadrado de la correlación de Pearson entre y_pred y s."""
    y = np.asarray(y_proba, dtype="float64").reshape(-1)
    s = np.asarray(s_array, dtype="float64").reshape(-1)
    y_c = y - y.mean()
    s_c = s - s.mean()
    numerator = float(np.dot(y_c, s_c)) ** 2
    denominator = float(np.dot(y_c, y_c) * np.dot(s_c, s_c)) + 1e-8
    return numerator / denominator


def compute_cka_rbf_np(y_proba, s_array, gamma=0.5, max_samples=20000):
    """CKA RBF en numpy con submuestreo estratificado por s."""
    y = np.asarray(y_proba, dtype="float64").reshape(-1)
    s = np.asarray(s_array, dtype="float64").reshape(-1)
    if len(y) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        classes, counts = np.unique(s, return_counts=True)
        proportions = counts / counts.sum()
        n_per_class = np.round(proportions * max_samples).astype(int)
        n_per_class[-1] += max_samples - n_per_class.sum()
        idx = np.concatenate([
            rng.choice(np.where(s == c)[0], size=int(n), replace=False)
            for c, n in zip(classes, n_per_class)
        ])
        y, s = y[idx], s[idx]

    K = np.exp(-np.square(y[:, None] - y[None, :]) / (2.0 * gamma**2))
    L = np.exp(-np.square(s[:, None] - s[None, :]) / (2.0 * gamma**2))

    def center(matrix):
        return (
            matrix
            - matrix.mean(axis=1, keepdims=True)
            - matrix.mean(axis=0, keepdims=True)
            + matrix.mean()
        )

    K_c = center(K)
    L_c = center(L)
    d2 = (len(y) - 1) ** 2
    hsic_kl = (K_c * L_c).sum() / d2
    hsic_kk = (K_c * K_c).sum() / d2
    hsic_ll = (L_c * L_c).sum() / d2
    return float(max(hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-8), 0.0))


def evaluate_all(y_true_np, y_proba, s_np, threshold=0.5, cka_gamma=0.5):
    """Calcula métricas predictivas y de dependencia para cualquier split."""
    y_true_np = np.asarray(y_true_np).reshape(-1).astype(int)
    y_proba = np.asarray(y_proba).reshape(-1)
    s_np = np.asarray(s_np).reshape(-1)
    y_pred = (y_proba >= threshold).astype(int)

    group_0 = s_np == 0
    group_1 = s_np == 1

    return {
        "auc": roc_auc_score(y_true_np, y_proba),
        "accuracy": accuracy_score(y_true_np, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true_np, y_pred),
        "recall_class_1": recall_score(y_true_np, y_pred, zero_division=0),
        "precision_class_1": precision_score(y_true_np, y_pred, zero_division=0),
        "f1_class_1": f1_score(y_true_np, y_pred, zero_division=0),
        "pearson_sq": compute_pearson_sq(y_proba, s_np),
        "cka_rbf": compute_cka_rbf_np(y_proba, s_np, gamma=cka_gamma),
        "mean_pred_gender_0": float(np.mean(y_proba[group_0])) if np.any(group_0) else np.nan,
        "mean_pred_gender_1": float(np.mean(y_proba[group_1])) if np.any(group_1) else np.nan,
        "mean_pred_gap_abs": float(abs(np.mean(y_proba[group_0]) - np.mean(y_proba[group_1]))) if np.any(group_0) and np.any(group_1) else np.nan,
        "predicted_positive_rate_gap_abs": float(abs(np.mean(y_pred[group_0]) - np.mean(y_pred[group_1]))) if np.any(group_0) and np.any(group_1) else np.nan,
    }


def prefixed_metrics(metrics_dict, prefix):
    return {f"{prefix}_{key}": value for key, value in metrics_dict.items()}


s_val_np = s_val.values.astype("float64")
s_test_np = s_test.values.astype("float64")

y_val_proba_base = model.predict(
    {"scaled_features": X_val_scaled_np, "financial_raw": X_val_financial_raw},
    batch_size=1024,
    verbose=0,
).ravel()
y_test_proba_base = y_test_proba

base_val_metrics = evaluate_all(y_val_np, y_val_proba_base, s_val_np, cka_gamma=0.5)
base_test_metrics = evaluate_all(y_test_np, y_test_proba_base, s_test_np, cka_gamma=0.5)

base_metrics = {
    "lambda_fair": 0.0,
    **prefixed_metrics(base_val_metrics, "val"),
    **prefixed_metrics(base_test_metrics, "test"),
}

base_metrics_df = pd.DataFrame([{ "model_name": "base_punto_1", **base_metrics }])
print("Métricas modelo base (validation y test):")
display(base_metrics_df)
