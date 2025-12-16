import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from visualize import visualize_predictions
import kagglehub

# by Kacper Pach s27112 & Dawid Frontczak s29608
# environment setup i dokumentacja w readme (https://github.com/dawiffi/NAI_pjatk7sem/blob/main/neural_net/README.md)

DATA_DIR = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")
IMG_SIZE = (64, 64) 
MAX_TOTAL_IMAGES = 200 

def load_dataset(base_path, split_name):
    """
    Wczytuje obrazy. Zwraca tablicę (N, 64, 64, 3).
    """
    data_path = os.path.join(base_path, split_name)
    classes = ['chihuahua', 'muffin'] 
    
    images = []
    labels = []
    
    print(f"\n--- Ładowanie zbioru: {split_name} (Limit: {MAX_TOTAL_IMAGES*2} obrazów) ---")
    
    for label_idx, class_name in enumerate(classes):
        current_count = 0
        class_folder = os.path.join(data_path, class_name)

        if not os.path.exists(class_folder):
            continue
            
        files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, file_name in enumerate(files):
            if current_count >= MAX_TOTAL_IMAGES:
                break
            
            file_path = os.path.join(class_folder, file_name)
            
            try:
                img = imread(file_path)
                img_resized = resize(img, IMG_SIZE, anti_aliasing=True)

                # Wymuszamy 3 kanały (RGB)
                if img_resized.shape[-1] == 4:
                    img_3ch = img_resized[:, :, :3]
                elif img_resized.shape[-1] == 3:
                    img_3ch = img_resized
                else:
                    continue
                
                images.append(img_3ch)
                labels.append(label_idx)
                current_count += 1
                
            except Exception as e:
                print(f"Błąd: {e}")
                continue

            print(f"Postęp: {current_count}/{MAX_TOTAL_IMAGES} [{class_name}]", end='\r')
        print()
    print(f"Ładowanie zakończone. Wczytano {MAX_TOTAL_IMAGES*2} obrazów.")
    return np.array(images), np.array(labels), classes

# Ładowanie danych
X_train, y_train, class_names = load_dataset(DATA_DIR, 'train')
X_test, y_test, _ = load_dataset(DATA_DIR, 'test')

print(f"\nDane załadowane. Kształt X_train: {X_train.shape}") 
# Oczekiwany kształt: (liczba_zdjęć, 64, 64, 3)

# Przygotowanie danych dla MLP 
# MLPClassifier wymaga 2D (samples, features), a my mamy 4D (samples, h, w, c).
# Musimy spłaszczyć dane na moment treningu/predykcji.

# reshape(liczba_probek, -1) automatycznie oblicza rozmiar drugiej osi (64*64*3 = 12288)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Trening Sieci MLP
print("\nRozpoczynam trening MLPClassifier...")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64)
)
mlp.fit(X_train_flat, y_train) 

y_pred = mlp.predict(X_test_flat)

acc = accuracy_score(y_test, y_pred)
print(f"\nDokładność (Accuracy): {acc*100:.2f}%")

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=class_names, labels=[0, 1]))
visualize_predictions("muffin vs chihuahua", X_test, y_test, y_pred, class_names)