import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
import base64
from flask import Flask, request, redirect, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load environment variables
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

# Initialize Flask app
app = Flask(__name__)

# Initialize Spotify OAuth
sp_oauth = SpotifyOAuth(client_id=CLIENT_ID,
                        client_secret=CLIENT_SECRET,
                        redirect_uri=REDIRECT_URI,
                        scope="user-library-read user-read-recently-played user-top-read")

# Fibonacci sequence generator
def generate_fibonacci_sequence(n):
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

# Fetch user playlists
def get_user_playlists(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get("https://api.spotify.com/v1/me/playlists", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch playlists: {response.status_code}")

# Fetch track features
def get_track_features(access_token, track_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"https://api.spotify.com/v1/audio-features/{track_id}", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch track features: {response.status_code}")

# Train a predictive model
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")
    return model

# Refresh access token
def refresh_access_token(refresh_token):
    TOKEN_URL = 'https://accounts.spotify.com/api/token'
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode('utf-8')).decode('utf-8')
    headers = {
        'Authorization': f'Basic {auth_header}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }
    response = requests.post(TOKEN_URL, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        raise Exception(f"Token refresh failed: {response.status_code}")

# Flask routes
@app.route('/')
def index():
    auth_url = sp_oauth.get_authorize_url()
    return f'<h2><a href="{auth_url}">Authenticate with Spotify</a></h2>'

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return "Authorization code is missing.", 400
    
    try:
        token_info = sp_oauth.get_access_token(code)
        if token_info:
            access_token = token_info['access_token']
            refresh_token = token_info.get('refresh_token')
            return f"Access token: {access_token}"
        else:
            return "Failed to retrieve access token.", 500
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/generate_playlist', methods=['GET'])
def generate_playlist():
    access_token = request.args.get('access_token')
    if not access_token:
        return "Access token is required.", 400
    
    try:
        # Use Fibonacci sequence to determine playlist length
        fibonacci_sequence = generate_fibonacci_sequence(10)
        playlist_length = fibonacci_sequence[-1]  # Use the last Fibonacci number
        
        # Fetch user playlists
        playlists = get_user_playlists(access_token)
        playlist_id = playlists['items'][0]['id']  # Use the first playlist
        
        # Fetch tracks from the playlist
        headers = {"Authorization": f"Bearer {access_token}"}
        playlist_tracks = requests.get(f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks", headers=headers).json()
        track_ids = [track['track']['id'] for track in playlist_tracks['items']]
        
        # Fetch track features
        features = []
        for track_id in track_ids[:playlist_length]:  # Limit to Fibonacci length
            track_features = get_track_features(access_token, track_id)
            features.append([
                track_features['danceability'],
                track_features['energy'],
                track_features['tempo']
            ])
        
        # Train a model (example: predict danceability > 0.5)
        labels = [1 if f[0] > 0.5 else 0 for f in features]  # Binary classification
        model = train_model(np.array(features), np.array(labels))
        
        return jsonify({
            "playlist_length": playlist_length,
            "track_features": features,
            "model_accuracy": "Check console for accuracy"
        })
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000)