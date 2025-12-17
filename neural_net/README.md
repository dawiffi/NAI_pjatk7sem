Przed uruchomieniem upewnij się, że masz: 
`pip install numpy scikit-image scikit-learn kagglehub pandas tensorflow`

1. Wykorzystać jeden z zbiorów danych z poprzednich ćwiczeń i naucz sieć neuronową.
Porównaj skuteczność obu podejść. Dodaj logi/print screen do repozytorium.
`python .\sonar.py`
wynik za pomocą sieci neuronowej:
<img width="505" height="253" alt="Zrzut ekranu 2025-12-16 215450" src="https://github.com/user-attachments/assets/88451040-bffc-4b12-ad5f-3fb2c08bfdcd" />
wynik z poprzedniego zadania:
<img width="592" height="603" alt="image" src="https://github.com/user-attachments/assets/b7741721-7f48-4b13-9de2-4f3839901608" />
w przypadku małego prostego datasetu jakim jest sonar drzewo decyzyjne daje dokładniejsze wyniki

2. Naucz sieć rozpoznawać zwierzęta, np. z zbioru CIFAR10
`python .\animals.py`
<img width="1497" height="831" alt="image" src="https://github.com/user-attachments/assets/2ea3dc5f-2792-47ae-a0c3-5ca5672cee63" />

3. Naucz sieć rozpoznawać ubrania. np. GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database.
`python .\clothes.py`
<img width="1493" height="827" alt="image" src="https://github.com/user-attachments/assets/b3ae62a7-2eee-41e2-a8ca-bea227c6fcbe" />

5. Zaskocz mnie. Zaproponuj własny przypadek użycia sieci neuronowych do problemu klasyfikacji.
zbiór muffin vs Chihuahua (https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification)
`python .\muffin_vs_chihuahua.py`
<img width="1497" height="833" alt="image" src="https://github.com/user-attachments/assets/5a9d6cbd-ea2d-4469-9326-d1878a7f2f4d" />

Dla jednego z punktu narysuj confussion matrix. Dodaj logi/print screen do repozytorium.
(wykonane dla fashion-mnist)
<img width="1366" height="1100" alt="image" src="https://github.com/user-attachments/assets/51c0bd3f-977b-44e0-90d3-3adf6ac11fa4" />

Do jednego z punktów użyj dwóch rozmiarów sieci neuronowych. Porównaj wyniki. Dodaj logi/print screen do repozytorium
rozmiar 1 (128, 64):
<img width="523" height="492" alt="image" src="https://github.com/user-attachments/assets/18d21d9a-7794-4e49-8d94-ebe549de5c5d" />
rozmiar 2 (256, 128, 32):
<img width="535" height="497" alt="image" src="https://github.com/user-attachments/assets/78c568c6-3118-433b-8268-bdceac5bab8e" />
w tym przypadku trening na większym rozmiarze sieci nie przyniósł dużych korzyści, jednak zwiększył mocno czas trenowania.

