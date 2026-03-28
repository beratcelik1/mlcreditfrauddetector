import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=5000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            linear = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        linear = np.dot(X, self.w) + self.b
        return self.sigmoid(linear)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


def confusion_counts(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def classification_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# load data
X_train = np.load("saved_models/X_train.npy")
y_train = np.load("saved_models/y_train.npy")
X_test = np.load("saved_models/X_test.npy")
y_test = np.load("saved_models/y_test.npy")

print("Data loaded successfully")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# train model
model = LogisticRegression(lr=0.01, n_iters=5000)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test, threshold=0.5)

# evaluate
results = classification_metrics(y_test, y_pred)

print("\nLogistic Regression Results")
print("Accuracy :", round(results["accuracy"], 4))
print("Precision:", round(results["precision"], 4))
print("Recall   :", round(results["recall"], 4))
print("F1 Score :", round(results["f1"], 4))
print("TP:", results["tp"], "TN:", results["tn"], "FP:", results["fp"], "FN:", results["fn"])


# =========================
# Extra Part: threshold tuning + hyperparameter tuning + AUC + save model
# =========================

def roc_curve_from_scratch(y_true, y_scores):
    # sort by predicted score descending
    desc_score_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tpr_list = [0.0]
    fpr_list = [0.0]

    tp = 0
    fp = 0
    prev_score = None

    for i in range(len(y_scores_sorted)):
        score = y_scores_sorted[i]
        label = y_true_sorted[i]

        if prev_score is not None and score != prev_score:
            tpr_list.append(tp / P if P > 0 else 0.0)
            fpr_list.append(fp / N if N > 0 else 0.0)

        if label == 1:
            tp += 1
        else:
            fp += 1

        prev_score = score

    tpr_list.append(tp / P if P > 0 else 0.0)
    fpr_list.append(fp / N if N > 0 else 0.0)

    return np.array(fpr_list), np.array(tpr_list)


def auc_from_scratch(fpr, tpr):
    auc = 0.0
    for i in range(1, len(fpr)):
        width = fpr[i] - fpr[i - 1]
        height = (tpr[i] + tpr[i - 1]) / 2
        auc += width * height
    return auc

def evaluate_with_thresholds(model, X_test, y_test, thresholds):
    probs = model.predict_proba(X_test)
    results = []

    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        metrics = classification_metrics(y_test, y_pred)
        metrics["threshold"] = threshold
        results.append(metrics)

    return probs, results


print("\n" + "=" * 50)
print("Hyperparameter Tuning for Logistic Regression")
print("=" * 50)

learning_rates = [0.001, 0.01, 0.1]
iteration_options = [1000, 5000, 10000]
thresholds = [0.3, 0.4, 0.5]

all_results = []
best_model = None
best_probs = None
best_result = None

# choose best model by F1 first, then recall
for lr in learning_rates:
    for n_iters in iteration_options:
        print(f"\nTraining model: lr={lr}, n_iters={n_iters}")

        temp_model = LogisticRegression(lr=lr, n_iters=n_iters)
        temp_model.fit(X_train, y_train)

        probs, threshold_results = evaluate_with_thresholds(
            temp_model, X_test, y_test, thresholds
        )

        for result in threshold_results:
            result["lr"] = lr
            result["n_iters"] = n_iters
            all_results.append(result)

            print(
                f"threshold={result['threshold']}, "
                f"precision={result['precision']:.4f}, "
                f"recall={result['recall']:.4f}, "
                f"f1={result['f1']:.4f}"
            )

            if best_result is None:
                best_result = result
                best_model = temp_model
                best_probs = probs
            else:
                if result["f1"] > best_result["f1"]:
                    best_result = result
                    best_model = temp_model
                    best_probs = probs
                elif result["f1"] == best_result["f1"] and result["recall"] > best_result["recall"]:
                    best_result = result
                    best_model = temp_model
                    best_probs = probs

print("\n" + "=" * 50)
print("Best Logistic Regression Setting")
print("=" * 50)
print("Learning rate :", best_result["lr"])
print("Iterations    :", best_result["n_iters"])
print("Threshold     :", best_result["threshold"])
print("Accuracy      :", round(best_result["accuracy"], 4))
print("Precision     :", round(best_result["precision"], 4))
print("Recall        :", round(best_result["recall"], 4))
print("F1 Score      :", round(best_result["f1"], 4))
print("TP:", best_result["tp"], "TN:", best_result["tn"], "FP:", best_result["fp"], "FN:", best_result["fn"])

# compute AUC using best model probabilities
fpr, tpr = roc_curve_from_scratch(y_test, best_probs)
auc_score = auc_from_scratch(fpr, tpr)

print("AUC-ROC       :", round(auc_score, 4))

# save best model parameters
np.save("saved_models/logreg_weights.npy", best_model.w)
np.save("saved_models/logreg_bias.npy", np.array([best_model.b]))
np.save("saved_models/logreg_best_probs.npy", best_probs)

print("\nBest model parameters saved:")
print("saved_models/logreg_weights.npy")
print("saved_models/logreg_bias.npy")
print("saved_models/logreg_best_probs.npy")

with open("saved_models/logreg_results.txt", "w") as f:
    f.write(f"Best LR: {best_result['lr']}\n")
    f.write(f"Iterations: {best_result['n_iters']}\n")
    f.write(f"Threshold: {best_result['threshold']}\n")
    f.write(f"Precision: {best_result['precision']}\n")
    f.write(f"Recall: {best_result['recall']}\n")
    f.write(f"F1: {best_result['f1']}\n")
    f.write(f"AUC: {auc_score}\n")