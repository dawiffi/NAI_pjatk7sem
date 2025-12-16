from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.datasets import fashion_mnist
from visualize import visualize_predictions

# by Kacper Pach s27112 & Dawid Frontczak s29608
# environment setup i dokumentacja w readme (https://github.com/dawiffi/NAI_pjatk7sem/blob/main/neural_net/README.md)

# Ładowanie danych
print("Ładowanie zbioru Fashion-MNIST")
(X_train_raw, y_train), (X_test_raw, y_test) = fashion_mnist.load_data()

# Nazwy klas zgodnie z dokumentacją Zalando (https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file)
class_names = [
    'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Obrazy w Fashion-MNIST mają wymiar 28x28 pikseli (1 kanał - szary).
# reshape(liczba_probek, -1) automatycznie spłaszcza wymiary.

X_train = X_train_raw.reshape(len(X_train_raw), -1)
X_test = X_test_raw.reshape(len(X_test_raw), -1)

# Normalizacja (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Tu możemy użyć mniejszej sieci niż przy CIFAR, bo dane są prostsze.
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), # Dwie warstwy ukryte
    max_iter=50,                  # Mniej epok wystarczy, bo zbiór jest prostszy
    verbose=True                  
)

print("\nRozpoczynam trening...")
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Dokładność (Accuracy): {acc * 100:.2f}%")

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=class_names))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()

visualize_predictions("clothes",X_test_raw, y_test, y_pred, class_names)