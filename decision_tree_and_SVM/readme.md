## instalacja potrzebnych bibliotek:
`pip install pandas scikit-learn matplotlib numpy`
## start programu 
`python main.py`

## Wywołanie 
konsola:
metryki jakości klasyfikacji 
<img width="1028" height="1542" alt="image" src="https://github.com/user-attachments/assets/6f74d6f3-f661-4358-a951-f67ffed04f22" />
<img width="695" height="1522" alt="image" src="https://github.com/user-attachments/assets/d16b79b6-fbdd-47f8-a9e1-f6281f4b376c" />
dla każdego kenela SVM oraz decision tree jest pokazywana wizualizacja
<img width="689" height="503" alt="image" src="https://github.com/user-attachments/assets/37ecd3ef-5aa5-4836-9bd6-8ac150e2490e" />
<img width="781" height="576" alt="image" src="https://github.com/user-attachments/assets/e515de38-fefd-4225-8d1c-8a0dbc17f00d" />

## Wnioski ze stosowania różnych kernel function 
Różne kernele SVC wpływają na wyniki tak, że RBF najczęściej daje najlepszą skuteczność, bo dobrze radzi sobie z danymi nieliniowymi. Linear działa dobrze tylko przy prostych, liniowo separowalnych danych. Poly może dać dobre wyniki, ale zależą one mocno od parametrów. Sigmoid zwykle wypada najsłabiej i jest najmniej stabilny.
<img width="673" height="1027" alt="image" src="https://github.com/user-attachments/assets/51adb58a-744f-4443-9b00-0f146c2b6224" />
<img width="707" height="956" alt="image" src="https://github.com/user-attachments/assets/20450eb0-2e73-4fcf-8424-6a06a30b8cd3" />
