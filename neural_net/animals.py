import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.datasets import cifar10
import time

# --- KROK 1: Ładowanie danych ---
print("Ładowanie zbioru CIFAR-10...")
# Używamy helpera z Keras tylko do pobrania danych (jest najszybszy/najstabilniejszy)
(X_train_raw, y_train), (X_test_raw, y_test) = cifar10.load_data()

# CIFAR-10 klasy (dla czytelności wyników)
class_names = ['Samolot', 'Samochód', 'Ptak', 'Kot', 'Jeleń', 
               'Pies', 'Żaba', 'Koń', 'Statek', 'Ciężarówka']

# Spłaszczamy etykiety (z tablicy 2D na 1D)
y_train = y_train.flatten()
y_test = y_test.flatten()

# --- KROK 2: Preprocessing (Wstępne przetwarzanie) ---
# Obrazy w CIFAR-10 mają wymiary 32x32 piksele i 3 kanały kolorów (RGB).
# Scikit-learn MLP wymaga wektora 1D (płaskiej listy liczb), nie obrazka 3D.

# Zmiana kształtu: (50000, 32, 32, 3) -> (50000, 3072)
# 32 * 32 * 3 = 3072 cechy na wejściu
X_train = X_train_raw.reshape(50000, 32 * 32 * 3)
X_test = X_test_raw.reshape(10000, 32 * 32 * 3)

# Normalizacja: Wartości pikseli są od 0 do 255. Skalujemy je do zakresu 0-1.
# To kluczowe dla zbieżności sieci neuronowej!
X_train = X_train / 255.0
X_test = X_test / 255.0

# OGRANICZENIE DANYCH (Dla celów edukacyjnych)
# MLP w Scikit-learn działa na procesorze (CPU) i jest wolny przy dużych danych.
# Weźmiemy tylko 5000 obrazków do treningu, żeby kod wykonał się w minutę, a nie godzinę.
subset_size = 5000
X_train_small = X_train[:subset_size]
y_train_small = y_train[:subset_size]

print(f"Dane przygotowane. Trenujemy na {subset_size} przykładach.")

# --- KROK 3: Definicja i Trening Sieci Neuronowej ---
print("Rozpoczynam trening sieci MLP (to może chwilę potrwać)...")

# MLPClassifier:
# hidden_layer_sizes=(128, 64) -> Dwie warstwy ukryte: pierwsza ma 128 neuronów, druga 64.
# activation='relu' -> Standardowa funkcja aktywacji.
# solver='adam' -> Optymalizator (najlepszy ogólny wybór).
# max_iter=100 -> Maksymalna liczba epok.

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), 
    activation='relu', 
    solver='adam', 
    alpha=1e-4, 
    batch_size=64, 
    learning_rate_init=0.001, 
    max_iter=100, 
    random_state=42,
    verbose=True,      # Wypisuje postęp w konsoli
    early_stopping=True # Zatrzymuje trening, jeśli sieć przestaje się uczyć
)

start_time = time.time()
mlp.fit(X_train_small, y_train_small)
end_time = time.time()

print(f"Trening zakończony w {end_time - start_time:.2f} sekund.")

# --- KROK 4: Ewaluacja ---
print("Przewidywanie na zbiorze testowym...")
y_pred = mlp.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nDokładność modelu (Accuracy): {acc * 100:.2f}%")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- KROK 5: Wizualizacja wyników ---
def show_predictions(X_img, y_true, y_pred, classes, num_samples=5):
    plt.figure(figsize=(10, 4))
    indices = np.random.choice(len(X_img), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        # Musimy odwrócić spłaszczenie, aby wyświetlić obrazek
        plt.imshow(X_img[idx], interpolation='nearest')
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        title = f"Prawda: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]}"
        
        plt.title(title, color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Wyświetlamy 5 losowych przykładów ze zbioru testowego
# Uwaga: Przekazujemy X_test_raw (niespłaszczone) do wyświetlania
show_predictions(X_test_raw, y_test, y_pred, class_names)