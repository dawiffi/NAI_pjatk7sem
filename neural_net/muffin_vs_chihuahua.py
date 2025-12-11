import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
import time
import kagglehub

# ==========================================
# KONFIGURACJA
# ==========================================
# ZMIEŃ TO na ścieżkę do folderu głównego (tam, gdzie są 'train' i 'test')

# --- KONFIGURACJA ---
# ZMIEŃ TO na swoją ścieżkę
DATA_DIR = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")
IMG_SIZE = (64, 64) 
MAX_TOTAL_IMAGES = 200 

def load_dataset_simple_progress(base_path, split_name):
    """
    Wczytuje obrazy z folderu z wyświetlaniem prostego licznika postępu.
    Poprawiono błąd 'ValueError: setting an array element with a sequence' 
    poprzez wymuszenie 3 kanałów (RGB) z użyciem samego numpy.
    """
    data_path = os.path.join(base_path, split_name)
    classes = ['chihuahua', 'muffin'] 
    
    images = []
    labels = []
    
    print(f"\n--- Ładowanie zbioru: {split_name} (Limit: {MAX_TOTAL_IMAGES} obrazów) ---")
    
    for label_idx, class_name in enumerate(classes):
        current_count = 0

        class_folder = os.path.join(data_path, class_name)
        print(class_folder, os.path.exists(class_folder))

        if not os.path.exists(class_folder):
            continue
            
        files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, file_name in enumerate(files):
            if current_count >= MAX_TOTAL_IMAGES:
                break
            
            file_path = os.path.join(class_folder, file_name)
            
            try:
                # 1. Wczytanie obrazu
                img = imread(file_path)
                
                # --- KLUCZOWA POPRAWKA BEZ NOWEGO IMPORTU ---
                
                # Zmiana rozmiaru na docelowy (64x64), by znormalizować wymiary
                img_resized = resize(img, IMG_SIZE, anti_aliasing=True)

                # Wymuszamy 3 kanały (RGB) używając funkcji NumPy:
                if img_resized.ndim == 2:
                    # Obraz jest czarno-biały (tylko 2 wymiary H x W). 
                    # Tworzymy 3 identyczne kanały (H x W x 3) za pomocą np.stack
                    img_3ch = np.stack([img_resized, img_resized, img_resized], axis=-1)
                elif img_resized.shape[-1] == 4:
                    # Obraz ma 4 kanały (RGBA). Ucinamy ostatni (Alpha/przezroczystość)
                    img_3ch = img_resized[:, :, :3]
                elif img_resized.shape[-1] == 3:
                    # Obraz ma 3 kanały (RGB). W porządku.
                    img_3ch = img_resized
                else:
                    # Inna nieoczekiwana liczba kanałów. Pomijamy.
                    continue
                
                # 2. Spłaszcz (Flatten): 64x64x3 -> wektor 12288 liczb
                img_flat = img_3ch.flatten()
                
                # --- KONIEC KLUCZOWEJ POPRAWKI ---

                images.append(img_flat)
                labels.append(label_idx)
                current_count += 1
                
            except Exception as e:
                # Złapanie błędu wczytywania (np. uszkodzony plik)
                print(f"Błąd przy przetwarzaniu {file_name}: {e}")
                continue

            # 3. Wyświetlanie postępu (X/Y)
            print(f"Postęp: {current_count}/{MAX_TOTAL_IMAGES} [{class_name}]", end='\r')

            
    # Na koniec procesu ładowania, drukujemy pustą linię
    print(" " * 60, end='\r')
    print(f"Ładowanie zakończone. Wczytano {current_count} obrazów.")
    return np.array(images), np.array(labels), classes # Teraz ta linia już nie powinna rzucać błędem

# ==========================================
# GŁÓWNA CZĘŚĆ KODU (jak poprzednio)
# ==========================================

# KROK 1: Ładowanie danych z prostym wskaźnikiem
X_train, y_train, class_names = load_dataset_simple_progress(DATA_DIR, 'train')
X_test, y_test, _ = load_dataset_simple_progress(DATA_DIR, 'test')

# Mieszanie danych treningowych
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print(f"\nDane załadowane. Trening: {X_train.shape[0]}, Test: {X_test.shape[0]}")

if len(X_train) == 0:
    print("BŁĄD: Zbiór treningowy jest pusty. Sprawdź ścieżkę DATA_DIR!")
    exit()

# KROK 2: Trening Sieci MLP
print("\nRozpoczynam trening MLPClassifier...")
start_time = time.time()

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    # Wyłączamy verbose, aby nie kolidował z naszym licznikiem postępu
    verbose=False
)

mlp.fit(X_train, y_train)
end_time = time.time()
print(f"Trening zakończony w {end_time - start_time:.2f} sekund.")

# KROK 3: Wyniki
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nDokładność (Accuracy): {acc*100:.2f}%")

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=class_names, labels=[0, 1]))
# Możesz dodać funkcję wizualizacji z poprzedniego kroku, jeśli chcesz zobaczyć błędy.
def visualize_predictions(X, y_true, y_pred, classes, count=6):
    plt.figure(figsize=(12, 6))
    indices = np.random.choice(len(X), count, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i+1)
        
        # Musimy "odwrócić" spłaszczenie, żeby wyświetlić obrazek
        # X[idx] to wektor, reshape zamienia go z powrotem w (64, 64, 3)
        img_reshaped = X[idx].reshape(IMG_SIZE[0], IMG_SIZE[1], 3)
        
        plt.imshow(img_reshaped)
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        title = f"Prawda: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]}"
        
        plt.title(title, color=color, fontsize=10, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle("Przykładowe wyniki (Muffin vs Chihuahua)", fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_predictions(X_test, y_test, y_pred, class_names)