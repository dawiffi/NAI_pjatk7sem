before running you need following the following: (install with pip install)
`scikit-learn, pandas, requests`

for the api integration you need to supply your own TMDB API key you can get it from: 
https://developer.themoviedb.org/reference/getting-started
the key need to be filled in the `recomend.py` as TMDB_API_KEY = "YOUR_KEY" 

parameters like TARGET_USER, file_path, NUM_CLUSTERS, N_RECOMMENDATIONS can be changed changed by the user in script directly

to run:
`python ./recomend.py`

if you want to supply your own data it needs to be formatted in the same way as in `formatted_data.csv` with the same column names.