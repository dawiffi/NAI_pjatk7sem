import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- KROK 1: Ładowanie danych ---
df = pd.read_csv('sonar.all-data')

# Podgląd danych
print(f"Wymiary danych: {df.shape}") # Powinno być (208, 61)
print("Przykładowy wiersz (pierwsze 5 kolumn):")
print(df.iloc[0, :5].values)
print(f"Etykieta tego wiersza: {df.iloc[0, 60]}") # 'R' (Rock) lub 'M' (Mine)

# --- KROK 2: Preprocessing ---

# Podział na cechy (X) i etykiety (y)
X = df.iloc[:, :-1].values  # Wszystkie kolumny oprócz ostatniej (60 liczb - sygnały sonarowe)
y = df.iloc[:, -1].values   # Tylko ostatnia kolumna (klasa: 'R' lub 'M')

# Zamiana liter na liczby (Kodowanie etykiet)
# Sieć neuronowa potrzebuje liczb (0 i 1), a nie liter ('R' i 'M')
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# Teraz: 'M' (Mina) -> 0, 'R' (Skała) -> 1 (lub odwrotnie, sprawdzimy to na końcu)
print(f"\nKlasy zakodowane jako: {encoder.classes_} -> [0, 1]")

# Podział na zbiór treningowy i testowy
# Test_size=0.2 oznacza, że 20% danych odkładamy na bok do sprawdzenia
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# WAŻNE: Skalowanie danych (StandardScaler)
# Dane sonarowe są już w zakresie 0-1, ale sieci neuronowe działają lepiej, 
# gdy dane mają średnią 0 i odchylenie standardowe 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- KROK 3: Definicja i Trening Sieci ---
print("\nRozpoczynam trening sieci...")

# Ponieważ mamy tylko 208 przykładów (bardzo mało!), sieć musi być prosta,
# żeby nie "nauczyła się danych na pamięć" (overfitting).
mlp = MLPClassifier(
    hidden_layer_sizes=(30, 30),  # Dwie warstwy po 30 neuronów
    activation='relu',
    solver='adam',                # 'adam' jest standardem, ale przy małych danych 'lbfgs' bywa lepszy
    max_iter=1000,                # Więcej epok, bo dane są trudne do rozdzielenia
    random_state=42
)

mlp.fit(X_train, y_train)
print("Trening zakończony.")

# --- KROK 4: Ewaluacja ---
y_pred = mlp.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nDokładność modelu: {acc * 100:.2f}%")

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

