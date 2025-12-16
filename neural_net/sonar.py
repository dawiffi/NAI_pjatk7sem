import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# by Kacper Pach s27112 & Dawid Frontczak s29608
# environment setup i dokumentacja w readme (https://github.com/dawiffi/NAI_pjatk7sem/blob/main/neural_net/README.md)

df = pd.read_csv('sonar.all-data')

# Podgląd danych
print(f"Wymiary danych: {df.shape}") # Powinno być (208, 61)
print("Przykładowy wiersz (pierwsze 5 kolumn):")
print(df.iloc[0, :5].values)
print(f"Etykieta tego wiersza: {df.iloc[0, 60]}") # 'R' (Rock) lub 'M' (Mine)

# Podział na cechy (X) i etykiety (y)
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values 

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nRozpoczynam trening sieci...")

mlp = MLPClassifier(
    hidden_layer_sizes=(30, 30),  # Dwie warstwy po 30 neuronów
    )

mlp.fit(X_train, y_train)
print("Trening zakończony.")

# --- KROK 4: Ewaluacja ---
y_pred = mlp.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nDokładność modelu: {acc * 100:.2f}%")

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))
