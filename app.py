from flask import Flask, request, render_template
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yaml
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load Spotify API credentials from a configuration file
with open('spotify_api.yaml', 'r') as file:
    config = yaml.safe_load(file)

CLIENT_ID = config['CLIENT_ID']
CLIENT_SECRET = config['CLIENT_SECRET']

# Initialize Spotify client
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

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
    plt.close()

    plt.figure()
    time_of_day.view()
    plt.savefig('static/time_of_day.png')
    plt.close()

    plt.figure()
    intensity.view()
    plt.savefig('static/intensity.png')
    plt.close()

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
    plt.close()

    plt.figure()
    danceability.view()
    plt.savefig('static/danceability.png')
    plt.close()

    plt.figure()
    energy.view()
    plt.savefig('static/energy.png')
    plt.close()

    plt.figure()
    valence.view()
    plt.savefig('static/valence.png')
    plt.close()

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
    plt.close()

    plt.figure()
    danceability.view(sim=danceability_inference)
    plt.savefig('static/danceability_result.png')
    plt.close()

    plt.figure()
    energy.view(sim=energy_inference)
    plt.savefig('static/energy_result.png')
    plt.close()

    plt.figure()
    valence.view(sim=valence_inference)
    plt.savefig('static/valence_result.png')
    plt.close()

    return {
        'acousticness': acousticness_inference.output['acousticness'],
        'danceability': danceability_inference.output['danceability'],
        'energy': energy_inference.output['energy'],
        'valence': valence_inference.output['valence']
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    # Get user inputs
    mood_input = float(request.form['mood'])
    intensity_level = float(request.form['intensity'])
    time = float(request.form['time'])
    seed_song_url = request.form['seed_song_url']

    # Run the fuzzy inference simulation and get the results
    output_values = run_fuzzy_simulation(mood_input, intensity_level, time)

    recommendations = sp.recommendations(
        seed_tracks=[seed_song_url.split('/')[-1].split('?')[0]],  # Extract track ID from URL
        limit=10,
        target_acousticness=output_values['acousticness'],
        target_danceability=output_values['danceability'],
        target_energy=output_values['energy'],
        target_valence=output_values['valence']
    )

    return render_template('results.html', 
                           mood_input=mood_input, 
                           intensity_level=intensity_level, 
                           time=time, 
                           seed_song_url=seed_song_url, 
                           output_values=output_values,
                           tracks=recommendations['tracks'])

if __name__ == '__main__':
    app.run(debug=True, threaded=False) 