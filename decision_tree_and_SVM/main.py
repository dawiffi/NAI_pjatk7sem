import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np


# --- Ocena Modeli ---
def rate_model(X_test, y_test, classifier, name):
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"## Wyniki dla {name}")
    print(f"Dokładność: {accuracy:.4f}")
    print("Raport klasyfikacji:")
    print(classification_report(y_test, predictions))
    print("---")

def convert_to_2d(X_data, mid_point):
    return np.column_stack(
        [
            X_data[:, mid_point:].mean(axis=1),  # x: average of second half
            X_data[:, :mid_point].mean(axis=1),  # y: average of first half
        ]
    )

def prepare_and_test(parameter_set, result_set ):
    # Podział danych na zbiór treningowy i testowy
    # Używamy stratify=y, aby zachować proporcje klas w obu zbiorach
    X_train, X_test, y_train, y_test = train_test_split(parameter_set, result_set, test_size=0.2, stratify=result_set)

    # --- 1. Trening Drzewa Decyzyjnego ---
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    plot_tree(dt_classifier)
    plt.show()

    ## Drzewo Decyzyjne
    rate_model(X_test, y_test, dt_classifier, "Drzewo decyzyjne")

    # --- 2. Trening Maszyny Wektorów Wspierających (SVM) ---
    # Użycie jądra (kernel) RBF (Radial Basis Function) dla nieliniowej klasyfikacji
    # Używamy przeskalowanych danych

    
    # --- Przygotowanie dla SVM (Skalowanie) ---
    # SVM jest wrażliwe na skalę cech. Choć dane Sonar są w zakresie [0, 1],
    # warto je przeskalować standardowo dla optymalnej pracy jądra RBF.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
                                          
        # Visualize decision boundary by averaging features (2D projection)
    # Split features in half and average each half
    n_features = X_train_scaled.shape[1]
    mid_point = n_features // 2

    # Create 2D projection: x = avg of 2nd half, y = avg of 1st half
    X_train_2d = convert_to_2d(X_train_scaled, mid_point)
    X_test_2d = convert_to_2d(X_test_scaled, mid_point)

    # Encode labels to numeric for visualization
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    kernels = {"linear",  "sigmoid","poly", "rbf"}
    for kernel in kernels:
        svm_classifier = SVC(kernel=kernel, class_weight='balanced')
        svm_classifier.fit(X_train_2d, y_train)
        DecisionBoundaryDisplay.from_estimator(
            svm_classifier,
            X_train_2d,
            response_method="predict",
            alpha=0.5,
            xlabel=f"Average of features {mid_point + 1}-{n_features}",
            ylabel=f"Average of features 1-{mid_point}",
        )
        plt.scatter(
            X_train_2d[:, 0],
            X_train_2d[:, 1],
            c=y_train_encoded,
            edgecolor="k",
            s=20,
            cmap="viridis",
        )
        plt.title("SVC Decision Boundary (2D projection - averaged features)")
        plt.show()
        
        ## SVM
        rate_model(X_test_2d, y_test, svm_classifier, f"SVM kernel: {kernel}") 


df = pd.read_csv('sonar.all-data') 

# Ostatnia kolumna to etykieta, reszta to cechy
X = df.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej
y = df.iloc[:, -1]   # Ostatnia kolumna (klasa: 'R' lub 'M')

prepare_and_test(X,y)

# dataset 2 
print("--- Potability of water ---")
df = pd.read_csv('water_potability.csv') 
df.fillna(0, inplace=True)
X = df.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej
y = df.iloc[:, -1] # Ostatnia kolumna (Sales_Classification: 'Low' lub 'High')

X_encoded = pd.get_dummies(X)

prepare_and_test(X_encoded,y)
