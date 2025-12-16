from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.datasets import cifar10
from visualize import visualize_predictions
import time

# by Kacper Pach s27112 & Dawid Frontczak s29608
# environment setup i dokumentacja w readme (https://github.com/dawiffi/NAI_pjatk7sem/blob/main/neural_net/README.md)

print("Ładowanie zbioru CIFAR-10...")
(X_train_raw, y_train), (X_test_raw, y_test) = cifar10.load_data()

# CIFAR-10 klasy (dla czytelności wyników)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Spłaszczamy etykiety (z tablicy 2D na 1D)
y_train = y_train.flatten()
y_test = y_test.flatten()

# 32 * 32 * 3 = 3072 cechy na wejściu
X_train = X_train_raw.reshape(len(X_train_raw), -1)
X_test = X_test_raw.reshape(len(X_test_raw), -1)

# Normalizacja: Wartości pikseli są od 0 do 255. Skalujemy je do zakresu 0-1.
X_train = X_train / 255.0
X_test = X_test / 255.0

subset_size = 5000
X_train_small = X_train[:subset_size]
y_train_small = y_train[:subset_size]

print(f"Dane przygotowane. Trenujemy na {subset_size} przykładach.")

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16), 
    max_iter=100, 
    verbose=True,      # Wypisuje postęp w konsoli
    #early_stopping=True # Zatrzymuje trening, jeśli sieć przestaje się uczyć
)

start_time = time.time()
mlp.fit(X_train_small, y_train_small)
end_time = time.time()

print(f"Trening zakończony w {end_time - start_time:.2f} sekund.")

print("Przewidywanie na zbiorze testowym...")
y_pred = mlp.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nDokładność modelu (Accuracy): {acc * 100:.2f}%")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=class_names))

visualize_predictions("Animals",X_test_raw, y_test, y_pred, class_names)