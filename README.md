# Idol-Group-Classification

## Data Source:  
https://www.kaggle.com/datasets/sberj127/kpop-hits-through-the-years  
The dataset contains songs included in Apple Music's annual K-Pop Hits playlists. Musical features for each song were obtained from Spotify using the Spotipy library.  
The documentation for the Spotify audio features can be found here:  
https://developer.spotify.com/documentation/web-api/reference/get-audio-features  

## Project Goal:  
We build several models with the task of classifying whether a song is attributed to an idol group or non-idol group artist(s). For each model, we would like to share explanations, background information, and key concepts in constructing the model. Each model should be accompanied with interactive visualizations that contribute to the viewer's understanding of the model. 

## Feature/Target Engineering:  
### Idol Group Classification:  
Songs were classified as either attributed to an idol group or non-idol group artist(s). Songs by soloists or solo members of idol groups were classified as attributed to non-idol group artist(s). Songs by idol group subunits were classified as attributed to an idol group. Songs by an idol group featuring non-affiliated artists were classified as attributed to an idol group. 
### Artist Gender:  
Artists were classified as either male, female, or mixed. Male and female artists include both soloists and groups. Mixed gender artists are groups comprising of artists of more than one gender, including features and collaborations. 

## How to Run the Dashboard:  
### dashboard.py
1. Download the files
2. Run the script dashboard.py
3. Open the Dash application running locally on your computer
### dashboard2.ipynb
1. Download the files
2. Run the notebook dashboard2.ipynb
3. The Dash application should open directly in your notebook
