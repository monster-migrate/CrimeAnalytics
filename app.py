from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from geopy.distance import great_circle
import joblib

app = Flask(__name__)

# Load your data (replace this with the path to your actual dataset)
dw_caw_latlong = pd.read_csv('./dw_caw_latlong.csv')  # Replace with the path to your dataset

crime_features = dw_caw_latlong[['Rape', 'Kidnapping and Abduction', 'Dowry Deaths',
                                 'Assault on women with intent to outrage her modesty',
                                 'Insult to modesty of Women', 'Cruelty by Husband or his Relatives',
                                 'Importation of Girls']]
scaler = StandardScaler()
crime_features_scaled = scaler.fit_transform(crime_features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
clusters = kmeans.fit_predict(crime_features_scaled)

# Add the cluster labels back to the original DataFrame
dw_caw_latlong['Cluster'] = clusters

# Define the function to find the nearest district
def find_nearest_district(lat, lon, df):
    distances = df.apply(lambda row: great_circle((lat, lon), (row['Latitude'], row['Longitude'])).km, axis=1)
    nearest_index = distances.idxmin()
    return df.loc[nearest_index, 'Cluster']

# Define the API endpoint
@app.route('/safety_rating', methods=['GET'])
def get_safety_rating():
    # Extract latitude and longitude from query parameters
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    # Validate the input
    if lat is None or lon is None:
        return jsonify({'error': 'Invalid or missing latitude and longitude parameters'}), 400
    
    # Find the safety rating (cluster)
    safety_rating = find_nearest_district(lat, lon, dw_caw_latlong)
    
    # Convert the safety rating to a native Python int
    safety_rating = int(safety_rating)
    
    # Return the result as JSON
    return jsonify({'safety_rating': safety_rating})

if __name__ == '__main__':
    app.run(debug=True)