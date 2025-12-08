import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import fashion_mnist

# --- KROK 1: Ładowanie danych ---
print("Ładowanie zbioru Fashion-MNIST...")
# Pobieramy dane (automatycznie pobierze się z serwerów Google/Zalando)
(X_train_raw, y_train), (X_test_raw, y_test) = fashion_mnist.load_data()

# Nazwy klas zgodnie z dokumentacją Zalando
class_names = [
    'T-shirt/Top', 'Spodnie', 'Sweter', 'Sukienka', 'Płaszcz',
    'Sandał', 'Koszula', 'Trampki', 'Torba', 'Botki'
]

# --- KROK 2: Preprocessing ---
# Obrazy w Fashion-MNIST mają wymiar 28x28 pikseli (1 kanał - szary).
# Musimy je spłaszczyć do wektora o długości 784 (28 * 28).

X_train = X_train_raw.reshape(X_train_raw.shape[0], 28 * 28)
X_test = X_test_raw.reshape(X_test_raw.shape[0], 28 * 28)

# Normalizacja (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Opcjonalnie: Zmniejszenie zbioru dla szybszego testu na słabszym komputerze
# Pełny zbiór to 60,000 obrazków. Weźmiemy 10,000 - to wystarczy, by mieć dobry wynik.
subset_size = 10000
X_train_subset = X_train[:subset_size]
y_train_subset = y_train[:subset_size]

print(f"Trening na {subset_size} przykładach (wymiar wejścia: 784 cechy).")

# --- KROK 3: Konfiguracja i Trening Sieci ---
# Tu możemy użyć nieco mniejszej sieci niż przy CIFAR, bo dane są prostsze.
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), # Dwie warstwy ukryte
    activation='relu',
    solver='adam',
    max_iter=50,                  # Mniej epok wystarczy, bo zbiór jest prostszy
    random_state=42,
    verbose=True                  # Widzimy postęp (funkcja straty maleje)
)

print("\nRozpoczynam trening...")
mlp.fit(X_train_subset, y_train_subset)


# --- KROK 4: Ewaluacja ---
print("\nSprawdzanie na pełnym zbiorze testowym (10 000 obrazków)...")
y_pred = mlp.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Dokładność (Accuracy): {acc * 100:.2f}%")

# Wyświetlamy szczegółowy raport dla każdej części garderoby
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=class_names))

# --- KROK 5: Wizualizacja Błędów i Sukcesów ---
def visualize_results(X_original, y_true, y_pred, classes):
    plt.figure(figsize=(12, 6))
    
    # Wybieramy losowe indeksy
    indices = np.random.choice(len(X_original), 10, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_original[idx], cmap='gray') # Ważne: cmap='gray' dla czarno-białych
        
        is_correct = y_true[idx] == y_pred[idx]
        color = 'green' if is_correct else 'red'
        
        # Jeśli błąd, pokażemy: Prawda (X) -> Predykcja (Y)
        label = classes[y_pred[idx]]
        if not is_correct:
            label = f"{classes[y_true[idx]]}\n-> {classes[y_pred[idx]]}"
            
        plt.title(label, color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle("Przykładowe predykcje (Zielone = OK, Czerwone = Błąd)", fontsize=14)
    plt.tight_layout()
    plt.show()

visualize_results(X_test_raw, y_test, y_pred, class_names)