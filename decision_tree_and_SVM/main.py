import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np

# by Kacper Pach s27112 & Dawid Frontczak s29608
# rules & environment setup in readme (https://github.com/dawiffi/NAI_pjatk7sem/blob/main/adversarial_search/README.md)

def rate_model(X_test, y_test, classifier, name):
    """
    Shows accuracy and a simple classification report for a model.
    """
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"## Results for {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("---")

def convert_to_2d(X_data, mid_point):
    """
    Converts many features into 2 values by averaging:
    - the first half of the features
    - the second half of the features
    
    Used only for drawing 2D decision boundaries.
    """
    return np.column_stack([
        X_data[:, mid_point:].mean(axis=1),   # X-axis
        X_data[:, :mid_point].mean(axis=1),   # Y-axis
    ])

def prepare_and_test(parameter_set, result_set):
    """
    Trains and tests:
    - a Decision Tree
    - several SVM models with different kernels
    
    Steps:
    1. Split data
    2. Train Decision Tree
    3. Scale data for SVM
    4. Reduce data to 2D for plots
    5. Train SVM models and show results
    """

    # --- Split data into train/test (80/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        parameter_set, result_set, test_size=0.2, stratify=result_set
    )

    # 1. DECISION TREE
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)

    # Show the tree structure
    plot_tree(dt_classifier)
    plt.show()

    rate_model(X_test, y_test, dt_classifier, "Decision Tree")

    # 2. SCALE THE DATA FOR SVM
    # SVM works better when features have similar scales.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create 2D data for visualizing the SVM borders
    n_features = X_train_scaled.shape[1]
    mid_point = n_features // 2

    X_train_2d = convert_to_2d(X_train_scaled, mid_point)
    X_test_2d = convert_to_2d(X_test_scaled, mid_point)

    # Convert labels to numbers for coloring
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # 3. TRAIN SVM MODELS
    kernels = {"linear", "sigmoid", "poly", "rbf"}

    for kernel in kernels:
        svm_classifier = SVC(kernel=kernel, class_weight='balanced')
        svm_classifier.fit(X_train_2d, y_train)

        # Draw the decision boundary in 2D
        DecisionBoundaryDisplay.from_estimator(
            svm_classifier,
            X_train_2d,
            response_method="predict",
            alpha=0.5,
            xlabel="Avg of second half of features",
            ylabel="Avg of first half of features",
        )

        plt.scatter(
            X_train_2d[:, 0],
            X_train_2d[:, 1],
            c=y_train_encoded,
            edgecolor="k",
            s=20,
            cmap="viridis",
        )
        plt.title(f"SVM Boundary (kernel={kernel})")
        plt.show()

        rate_model(X_test_2d, y_test, svm_classifier, f"SVM kernel: {kernel}")

print("--- Sonar ---")

df = pd.read_csv('sonar.all-data')

# Last column = label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

prepare_and_test(X, y)

print("--- Potability of Water ---")

df = pd.read_csv('water_potability.csv')
df.fillna(0, inplace=True)   # Replace missing values with 0

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert categorical data (if any)
X_encoded = pd.get_dummies(X)

prepare_and_test(X_encoded, y)
