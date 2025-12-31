import numpy as np

def fit(X, y, categorical_features, var_smoothing=1e-6):
    # x -> features
    # y -> target
    # categorical_features -> gender, cholesterol, gluc, smoke, alco, active
    """
    Fit  Naive Bayes (numerical  + categorical)
    Returns model parameters
    """
    classes = np.unique(y) # classes to predict (yes \ no)
    n_features = X.shape[1]

    # get features with numeric values 
    numerical_features = []
    for i in range(n_features):
        if i not in categorical_features:
            numerical_features.append(i)

    
    mean = {}
    var = {}
    class_priors = {}
    cat_prob = {}

    # If a feature has very small variance (close to 0), dividing by will blow up the computation or produce infinite.
    #We add a tiny constant to the variance
    epsilon = var_smoothing 
    
    for c in classes:
        X_c = X[y == c]
        class_priors[c] = len(X_c) / len(X)
        
        # Continuous
        mean[c] = {}
        var[c] = {}
        cat_prob[c] = {}

        for f in numerical_features:
            mean[c][f] = np.mean(X_c[:, f])
            var[c][f] = np.var(X_c[:, f]) + epsilon

        # Categoricalfor f in categorical_features:
        for f in categorical_features:
            cat_prob[c][f] = {}
            values, counts = np.unique(X_c[:, f], return_counts=True)

            total = len(X_c)
            num_values = len(values)

            for v, cnt in zip(values, counts):
                cat_prob[c][f][v] = (cnt + 1) / (total + num_values)
                                # Laplace smoothing

    
    return classes, mean, var, cat_prob, class_priors, numerical_features

def gaussian_log_P(x, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)


def categorical_log_P(x, cls, categorical_features, cat_prob):
    log_p = 0.0

    for f in categorical_features:
        value = x[f]
        prob = cat_prob[cls][f].get(value, 1e-9)  
        log_p += np.log(prob)
    return log_p

def predict(X, classes, mean, var, cat_prob, class_priors,
            numerical_features, categorical_features):

    predictions = []

    for x in X:
        class_scores = {}

        for cls in classes:
            log_prob = np.log(class_priors[cls])

            # numerical features
            for f in numerical_features:
                log_prob += gaussian_log_P(x[f], mean[cls][f], var[cls][f])

            # categorical features
            log_prob += categorical_log_P(
                x, cls, categorical_features, cat_prob
            )

            class_scores[cls] = log_prob

        # choose class with max probability
        predictions.append(max(class_scores, key=class_scores.get))

    return np.array(predictions)

def score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_f1(y_true, y_pred, positive_class=1):
    tp = 0
    fp = 0
    fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yp == positive_class and yt == positive_class:
            tp += 1
        elif yp == positive_class and yt != positive_class:
            fp += 1
        elif yp != positive_class and yt == positive_class:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, f1

import numpy as np

def confusion_matrix_binary(y_true, y_pred, positive_class=1):
    tp = fp = tn = fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == positive_class and yp == positive_class:
            tp += 1
        elif yt != positive_class and yp == positive_class:
            fp += 1
        elif yt == positive_class and yp != positive_class:
            fn += 1
        else:
            tn += 1

    return np.array([[tn, fp],
                    [fn, tp]])

def plot_confusion_matrix(cm, classes=["0", "1"], title="Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.show()
    
import pandas as pd
import numpy as np

# LOAD TRAIN DATA 
train_df = pd.read_csv("train_data.csv")
train_df = train_df.drop(columns=["id"])
X_train = train_df.drop(columns=["cardio"]).values
y_train = train_df["cardio"].values

categorical_features = [1, 6, 7, 8, 9, 10]

categorical_columns = train_df.columns[categorical_features]
print(categorical_columns)

# TRAIN
classes, mean, var, cat_prob, class_priors, numerical_features = fit(
    X_train,
    y_train,
    categorical_features
)

# SAVE MODEL
np.savez(
    "naive_bayes_model.npz",
    classes=classes,
    mean=mean,
    var=var,
    cat_prob=cat_prob,
    class_priors=class_priors,
    numerical_features=numerical_features,
    categorical_features=categorical_features
)

print("Model trained and saved.")
import pandas as pd
import numpy as np

#  LOAD MODEL 
model = np.load("naive_bayes_model.npz", allow_pickle=True)

classes = model["classes"]
mean = model["mean"].item()
var = model["var"].item()
cat_prob = model["cat_prob"].item()
class_priors = model["class_priors"].item()
numerical_features = model["numerical_features"].tolist()
categorical_features = model["categorical_features"].tolist()

#  LOAD TEST DATA 
test_df = pd.read_csv("test_data.csv")
test_df = test_df.drop(columns=["id"])

X_test = test_df.drop(columns=["cardio"]).values
y_test = test_df["cardio"].values

#  PREDICT 
y_pred = predict(
    X_test,
    classes,
    mean,
    var,
    cat_prob,
    class_priors,
    numerical_features,
    categorical_features
)

print("Test Accuracy:", score(y_test, y_pred))

precision, f1 = precision_f1(y_test, y_pred)

print("Precision:", precision)
print("F1-score:", f1)
import matplotlib.pyplot as plt
import seaborn as sns


cm = confusion_matrix_binary(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plot_confusion_matrix(cm, classes=["No Cardio", "Cardio"])


