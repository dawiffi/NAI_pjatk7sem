import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(plotname,X , y_true, y_pred, classes):
    """
    Wizualizuje losowo wybrane predykcje modelu, porównując je z prawdą.
    Wyświetla próbki w siatce i koloruje tytuły (Zielony = OK, Czerwony = Błąd).

    Parametry:
    ----------
    plotname : 
        Główny tytuł całego wykresu.
    X :
        Tablica z obrazami wejściowymi.
    y_true : 
        Wektor prawdziwych etykiet (indeksy klas).
    y_pred : 
        Wektor etykiet przewidzianych przez model.
    classes :
        Lista nazw klas odpowiadających indeksom (np. ['chihuahua', 'muffin']).
    """
    plt.figure(figsize=(12, 6))
    indices = np.random.choice(len(X), 6, replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i+1)
        
        img= X[idx]
        
        plt.imshow(img)
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        title = f"Prawda: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]}"
        
        plt.title(title, color=color, fontsize=10, fontweight='bold')
        plt.axis('off')
    
    plt.suptitle(plotname, fontsize=16)
    plt.tight_layout()
    plt.show()
