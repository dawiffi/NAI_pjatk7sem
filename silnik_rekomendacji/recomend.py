import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests

# by Kacper Pach s27112 & Dawid Frontczak s29608
# rules & environment setup in readme (https://github.com/dawiffi/NAI_pjatk7sem/blob/main/silnik_rekomendacji/README.md)

# --- Konfiguracja ---
file_path = "formatted_data.csv"
TARGET_USER = 'Dawid Frontczak'
NUM_CLUSTERS = 3
N_RECOMMENDATIONS = 5

# UZUPEŁNIJ SWÓJ KLUCZ API TMDB TUTAJ!
TMDB_API_KEY = "YOUR_KEY" 
BASE_URL = "https://api.themoviedb.org/3/"
# --------------------

def fetch_movie_details(title):
    """
    Wyszukuje film/serial po tytule w TMDB i zwraca podstawowe szczegóły.
    Wymaga zainstalowania biblioteki 'requests' i ważnego klucza TMDB_API_KEY.
    """
    try:
        # Używamy endpointu multi, by znaleźć i filmy, i seriale
        search_url = f"{BASE_URL}search/multi"
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
            'language': 'pl-PL'
        }
        
        response = requests.get(search_url, params=params, timeout=5)
        response.raise_for_status()
        
        results = response.json().get('results')
        
        if results and len(results) > 0:
            best_match = results[0]
            
            # Weryfikacja typu treści
            media_type = best_match.get('media_type')
            
            if media_type == 'movie':
                main_title = best_match.get('title')
                release_date = best_match.get('release_date', 'Brak daty')
            elif media_type == 'tv':
                main_title = best_match.get('name')
                release_date = best_match.get('first_air_date', 'Brak daty')
                
            return {
                'Tytuł': main_title or title,
                'Typ': media_type.upper(),
                'Opis (Skrót)': best_match.get('overview', 'Brak opisu.')[:150] + '...',
                'Rok wydania': release_date.split('-')[0] if release_date else 'N/A'
            }
        
        return None
        
    except requests.exceptions.RequestException as e:
        return {'Tytuł': title, 'Opis (Skrót)': f"Błąd API: {e}", 'Rok wydania': 'N/A', 'Typ': 'Błąd'}

def load_and_transform_data(file_path):
    """
    Wczytuje dane z formatu długiego (UserID, MovieID, Rating) z nagłówkiem.
    Czyści dane, zapewniając poprawne typy kolumn.
    """

    data = pd.read_csv(file_path)
    
    # Konwersja ocen na float, z obsługą błędów (coercing)
    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    
    # Usunięcie wierszy z brakującymi lub niepoprawnymi wartościami
    data_final = data.dropna(subset=['UserID', 'MovieID', 'Rating'])
    
    # Finalna konwersja typów: ID na string, Oceny na float
    data_final['UserID'] = data_final['UserID'].astype(str)
    data_final['MovieID'] = data_final['MovieID'].astype(str)
    
    return data_final

def get_recommendations(user_id, matrix, n=5):
    """Generuje top N rekomendacji i antyrekomendacji na podstawie średnich ocen w klastrze."""
    if user_id not in matrix.index:
        return ["Brak użytkownika w macierzy"], ["Brak użytkownika w macierzy"]

    target_cluster = matrix.loc[user_id, 'Cluster']
    cluster_data = matrix[matrix['Cluster'] == target_cluster].drop(columns=['Cluster'])

    # Filmy nieoglądane przez użytkownika
    user_ratings = matrix.loc[user_id].drop('Cluster', errors='ignore')
    unseen_movies = user_ratings[user_ratings == 0].index.tolist()
    
    if not unseen_movies:
        return ["Brak nieoglądanych filmów"], ["Brak nieoglądanych filmów"]

    # Obliczanie średniej oceny dla nieoglądanych filmów w klastrze
    cluster_means = cluster_data[unseen_movies].mean()
    
    # Rekomendacje: Top N z najwyższą średnią
    # ascending=False (od najwyższej do najniższej)
    recommendation_series = cluster_means.sort_values(ascending=False)
    top_recommendations = recommendation_series.head(n).index.tolist()
    
    # Antyrekomendacje: Top N z najniższą średnią
    # ascending=True (od najniższej do najwyższej)
    antirecommendation_series = cluster_means.sort_values(ascending=True)
    top_antirecommendations = antirecommendation_series.head(n).index.tolist()
            
    return top_recommendations, top_antirecommendations

def main():
    """Główna funkcja uruchamiająca silnik rekomendacji."""
    print("--- 1. Wczytywanie i Klasteryzacja Danych ---")
    data_final = load_and_transform_data(file_path)

    user_movie_matrix = data_final.pivot_table(
        index='UserID',
        columns='MovieID',
        values='Rating'
    ).fillna(0)

    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_movie_matrix)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS)
    user_movie_matrix['Cluster'] = kmeans.fit_predict(user_features_scaled)
    
    # Weryfikacja, czy użytkownik istnieje
    if TARGET_USER not in user_movie_matrix.index:
        print(f"Błąd: Użytkownik '{TARGET_USER}' nie istnieje w danych.")
        return

    target_cluster = user_movie_matrix.loc[TARGET_USER, 'Cluster']
    
    print(f"Użytkownik '{TARGET_USER}' należy do klastra: {target_cluster}")
    print(f"Liczba filmów/seriali: {user_movie_matrix.shape[1] - 1}")
    
    # 2. Generowanie rekomendacji i antyrekomendacji
    recommendations, antirecommendations = get_recommendations(TARGET_USER, user_movie_matrix, N_RECOMMENDATIONS)

    # 3. Wyświetlanie wyników z użyciem API
    print(f"\n==============================================")
    print(f"⭐️ TOP {N_RECOMMENDATIONS} REKOMENDACJI dla {TARGET_USER}:")
    for i, title in enumerate(recommendations):
        details = fetch_movie_details(title)
        if details:
            print(f"--- {i+1}. {details['Tytuł']} ({details['Rok wydania']} | {details['Typ']}) ---")
            print(f"   Opis: {details['Opis (Skrót)']}")
        else:
            print(f"--- {i+1}. {title} (Brak szczegółów w TMDB) ---")

    print(f"\n❌ TOP {N_RECOMMENDATIONS} ANTYREKOMENDACJI (Nie polecane):")
    for i, movie in enumerate(antirecommendations):
        print(f"{i+1}. {movie}")
    print(f"==============================================")

if __name__ == "__main__":
    main()