import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# --- Ocena Modeli ---
def rate_model(X_test, y_test, classifier, name):
    dt_predictions = classifier.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    print(f"## Wyniki dla {name}")
    print(f"Dokładność: {dt_accuracy:.4f}")
    print("Raport klasyfikacji:")
    print(classification_report(y_test, dt_predictions))
    print("---")

df = pd.read_csv('sonar.all-data') 

# Ostatnia kolumna to etykieta, reszta to cechy
X = df.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej
y = df.iloc[:, -1]   # Ostatnia kolumna (klasa: 'R' lub 'M')

# Podział danych na zbiór treningowy i testowy
# Używamy stratify=y, aby zachować proporcje klas w obu zbiorach
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# --- Przygotowanie dla SVM (Skalowanie) ---
# SVM jest wrażliwe na skalę cech. Choć dane Sonar są w zakresie [0, 1],
# warto je przeskalować standardowo dla optymalnej pracy jądra RBF.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. Trening Drzewa Decyzyjnego ---
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# --- 2. Trening Maszyny Wektorów Wspierających (SVM) ---
# Użycie jądra (kernel) RBF (Radial Basis Function) dla nieliniowej klasyfikacji
# Używamy przeskalowanych danych
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train_scaled, y_train)

## Drzewo Decyzyjne
rate_model(X_test, y_test, dt_classifier, "Drzewo decyzyjne")

## SVM
rate_model(X_test, y_test, svm_classifier, "SVM")

# dataset 2 
print("--- BMW ---")
df = pd.read_csv('BMW_sales(2010-2024).csv', header=1) 
X = df.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej
y = df.iloc[:, -1] # Ostatnia kolumna (Sales_Classification: 'Low' lub 'High')

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, stratify=y)

# --- Przygotowanie dla SVM (Skalowanie) ---
# SVM jest wrażliwe na skalę cech. Choć dane Sonar są w zakresie [0, 1],
# warto je przeskalować standardowo dla optymalnej pracy jądra RBF.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1. Trening Drzewa Decyzyjnego ---
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# --- 2. Trening Maszyny Wektorów Wspierających (SVM) ---
# Użycie jądra (kernel) RBF (Radial Basis Function) dla nieliniowej klasyfikacji
# Używamy przeskalowanych danych
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train_scaled, y_train)

## Drzewo Decyzyjne
rate_model(X_test, y_test, dt_classifier, "Drzewo decyzyjne BMW")

## SVM
rate_model(X_test, y_test, svm_classifier, "SVM BMW")
