from flask import Flask, request, render_template
from apscheduler.schedulers.background import BackgroundScheduler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import os
import requests
import urllib3
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__, template_folder='.')

def ping():
    print("Pinging the server...")

# Set up APScheduler to run the ping function every 10 minutes
scheduler = BackgroundScheduler()
scheduler.add_job(ping, 'interval', minutes=10)  # This will run `ping()` every 10 minutes
scheduler.start()

# Load Spotify API credentials from a configuration file
CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']

# logic from ref - https://github.com/spotipy-dev/spotipy/issues/766#issuecomment-1005102900

# Create a requests session with retry logic
session = requests.Session()

retry = urllib3.Retry(
    total=3,  # Retry up to 3 times
    connect=None,
    read=3,  # Retry on read failures (including rate limiting)
    allowed_methods=frozenset(['GET', 'POST', 'PUT', 'DELETE']),
    status=3,  # Retry on certain HTTP status codes
    backoff_factor=0.5,  # Delay between retries
    status_forcelist=(429, 500, 502, 503, 504),  # These are retryable errors
    respect_retry_after_header=True  # Automatically respect the 'Retry-After' header
)

adapter = requests.adapters.HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Initialize Spotify client with custom session
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=client_credentials_manager, requests_session=session)

if not os.path.exists('static'):
    os.makedirs('static')

def run_fuzzy_simulation(mood_input, intensity_level, time):

    mood = ctrl.Antecedent(np.arange(0, 11, 1), 'mood')
    time_of_day = ctrl.Antecedent(np.arange(0, 24, 1), 'time_of_day')
    intensity = ctrl.Antecedent(np.arange(0, 11, 1), 'intensity')

    mood['horrible'] = fuzz.trimf(mood.universe, [0, 0, 4])
    mood['neutral'] = fuzz.trimf(mood.universe, [3, 5, 7])
    mood['fantastic'] = fuzz.trimf(mood.universe, [6, 10, 10])

    time_of_day['early'] = fuzz.trimf(time_of_day.universe, [0, 0, 7])
    time_of_day['mid_morning'] = fuzz.trimf(time_of_day.universe, [6, 9, 12])
    time_of_day['afternoon'] = fuzz.trimf(time_of_day.universe, [11, 13, 16])
    time_of_day['evening'] = fuzz.trimf(time_of_day.universe, [15, 19, 21])
    time_of_day['night'] = fuzz.trimf(time_of_day.universe, [20, 23, 23])

    intensity['calm'] = fuzz.trimf(intensity.universe, [0, 0, 4])
    intensity['neutral'] = fuzz.trimf(intensity.universe, [3, 5, 7])
    intensity['lively'] = fuzz.trimf(intensity.universe, [6, 10, 10])

    plt.figure()
    mood.view()
    plt.savefig('static/mood.png')
    plt.close('all')

    plt.figure()
    time_of_day.view()
    plt.savefig('static/time_of_day.png')
    plt.close('all')

    plt.figure()
    intensity.view()
    plt.savefig('static/intensity.png')
    plt.close('all')

    acousticness = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'acousticness')
    danceability = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'danceability')
    energy = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'energy')
    valence = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'valence')

    # Set fuzzy membership functions for outputs
    acousticness['low'] = fuzz.trimf(acousticness.universe, [0.0, 0.0, 0.4])
    acousticness['medium'] = fuzz.trimf(acousticness.universe, [0.3, 0.5, 0.7])
    acousticness['high'] = fuzz.trimf(acousticness.universe, [0.6, 1.0, 1.0])

    danceability['low'] = fuzz.trimf(danceability.universe, [0.0, 0.0, 0.4])
    danceability['medium'] = fuzz.trimf(danceability.universe, [0.3, 0.5, 0.7])
    danceability['high'] = fuzz.trimf(danceability.universe, [0.6, 1.0, 1.0])

    energy['low'] = fuzz.trimf(energy.universe, [0.0, 0.0, 0.4])
    energy['medium'] = fuzz.trimf(energy.universe, [0.3, 0.5, 0.7])
    energy['high'] = fuzz.trimf(energy.universe, [0.6, 1.0, 1.0])

    valence['low'] = fuzz.trimf(valence.universe, [0.0, 0.0, 0.4])
    valence['medium'] = fuzz.trimf(valence.universe, [0.3, 0.5, 0.7])
    valence['high'] = fuzz.trimf(valence.universe, [0.6, 1.0, 1.0])

    plt.figure()
    acousticness.view()
    plt.savefig('static/acousticness.png')
    plt.close('all')

    plt.figure()
    danceability.view()
    plt.savefig('static/danceability.png')
    plt.close('all')

    plt.figure()
    energy.view()
    plt.savefig('static/energy.png')
    plt.close('all')

    plt.figure()
    valence.view()
    plt.savefig('static/valence.png')
    plt.close('all')

    rule1 = ctrl.Rule(intensity['lively'], acousticness['low'])
    rule2 = ctrl.Rule((intensity['neutral'] | mood['fantastic']), acousticness['medium'])
    rule3 = ctrl.Rule((intensity['calm'] | mood['horrible']), acousticness['high'])

    # rules for danceability
    rule4 = ctrl.Rule((mood['horrible'] | time_of_day['early'] | time_of_day['night']), danceability['low'])
    rule5 = ctrl.Rule((mood['neutral'] | time_of_day['mid_morning']), danceability['medium'])
    rule6 = ctrl.Rule((mood['fantastic'] | time_of_day['afternoon'] | time_of_day['evening']), danceability['high'])

    # rules for energy
    rule7 = ctrl.Rule((intensity['calm'] | time_of_day['early'] | time_of_day['night']), energy['low'])
    rule8 = ctrl.Rule((intensity['neutral'] | time_of_day['mid_morning']), energy['medium'])
    rule9 = ctrl.Rule((intensity['lively'] | time_of_day['afternoon'] | time_of_day['evening']), energy['high'])

    # rules for valence
    rule10 = ctrl.Rule(mood['horrible'], valence['low'])
    rule11 = ctrl.Rule(mood['neutral'], valence['medium'])
    rule12 = ctrl.Rule(mood['fantastic'], valence['high'])

    acousticness_control = ctrl.ControlSystem([rule1, rule2, rule3])
    danceability_control = ctrl.ControlSystem([rule4, rule5, rule6])
    energy_control = ctrl.ControlSystem([rule7, rule8, rule9])
    valence_control = ctrl.ControlSystem([rule10, rule11, rule12])

    acousticness_inference = ctrl.ControlSystemSimulation(acousticness_control)
    danceability_inference = ctrl.ControlSystemSimulation(danceability_control)
    energy_inference = ctrl.ControlSystemSimulation(energy_control)
    valence_inference = ctrl.ControlSystemSimulation(valence_control)

    acousticness_inference.input['mood'] = mood_input
    acousticness_inference.input['intensity'] = intensity_level

    danceability_inference.input['mood'] = mood_input
    danceability_inference.input['time_of_day'] = time

    energy_inference.input['intensity'] = intensity_level
    energy_inference.input['time_of_day'] = time

    valence_inference.input['mood'] = mood_input

    acousticness_inference.compute()
    danceability_inference.compute()
    energy_inference.compute()
    valence_inference.compute()

    # Save output graphs
    plt.figure()
    acousticness.view(sim=acousticness_inference)
    plt.savefig('static/acousticness_result.png')
    plt.close('all')

    plt.figure()
    danceability.view(sim=danceability_inference)
    plt.savefig('static/danceability_result.png')
    plt.close('all')

    plt.figure()
    energy.view(sim=energy_inference)
    plt.savefig('static/energy_result.png')
    plt.close('all')

    plt.figure()
    valence.view(sim=valence_inference)
    plt.savefig('static/valence_result.png')
    plt.close('all')

    return {
        'acousticness': round(acousticness_inference.output['acousticness'], 3),
        'danceability': round(danceability_inference.output['danceability'], 3),
        'energy': round(energy_inference.output['energy'], 3),
        'valence': round(valence_inference.output['valence'], 3)
    }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_song', methods=['GET'])
def search_song():
    query = request.args.get('query', '')
    if query:
        try:
            results = sp.search(q=query, type='track', limit=5)
            songs = [
                {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name']
                }
                for track in results['tracks']['items']
            ]
            return {'songs': songs}
        except Exception as e:
            return {'error': str(e)}, 500
    return {'songs': []}

@app.route('/search_genre', methods=['GET'])
def search_genre():
    query = request.args.get('query', '')
    try:
        genres = sp.recommendation_genre_seeds()  # Fetch all genres from the Spotify API
        return {'genres': genres['genres']}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/results', methods=['POST'])
def results():
    # Get user inputs
    mood_input = float(request.form['mood'])
    intensity_level = float(request.form['intensity'])
    time = float(request.form['time'])
    seed_song_id = request.form.get('seed_song_id')
    seed_genre = request.form.get('seed_genre')

    seed_song_url = None
    seed_song_name = None
    seed_song_artist = None

    if seed_song_id:
        try:
            seed_song_url = f"https://open.spotify.com/track/{seed_song_id}"
            track_details = sp.track(seed_song_id)
            seed_song_name = track_details['name']
            seed_song_artist = track_details['artists'][0]['name']
        except Exception as e:
            return {'error': f"Error fetching song details: {str(e)}"}, 500

    output_values = run_fuzzy_simulation(mood_input, intensity_level, time)

    # Prioritize genre over song for recommendations
    try:
        if seed_genre:
            recommendations = sp.recommendations(
                seed_genres=[seed_genre],
                limit=10,
                target_acousticness=output_values['acousticness'],
                target_danceability=output_values['danceability'],
                target_energy=output_values['energy'],
                target_valence=output_values['valence'])
        else:
            recommendations = sp.recommendations(
                seed_tracks=[seed_song_id],
                limit=10,
                target_acousticness=output_values['acousticness'],
                target_danceability=output_values['danceability'],
                target_energy=output_values['energy'],
                target_valence=output_values['valence'])
            
    except Exception as e:
        return {'error': f"Error fetching recommendations: {str(e)}"}, 500

    return render_template('results.html',
                           mood_input=mood_input,
                           intensity_level=intensity_level,
                           time=time,
                           seed_song_url=seed_song_url,
                           seed_song_name=seed_song_name,
                           seed_song_artist=seed_song_artist,
                           seed_song_genre = seed_genre,
                           output_values=output_values,
                           tracks=recommendations['tracks'])

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
