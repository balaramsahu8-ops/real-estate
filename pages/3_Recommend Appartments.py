import streamlit as st
import ast
import pickle
from pathlib import Path
import re
import pandas as pd

st.set_page_config(page_title="Recommend Appartments")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / 'models'

with open(MODELS_DIR / 'loaction_distance.pkl', 'rb') as file:
    location_df = pickle.load(file)

with open(MODELS_DIR / 'cosine_sim1.pkl', 'rb') as file:
    cosine_sim1 = pickle.load(file)

with open(MODELS_DIR / 'cosine_sim2.pkl', 'rb') as file:
    cosine_sim2 = pickle.load(file)

with open(MODELS_DIR / 'cosine_sim3.pkl', 'rb') as file:
    cosine_sim3 = pickle.load(file)

property_names = location_df['PropertyName'].astype(str).tolist()


def parse_location_advantages(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed_value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        return parsed_value if isinstance(parsed_value, dict) else {}
    return {}


location_df['ParsedLocationAdvantages'] = location_df['LocationAdvantages'].apply(
    parse_location_advantages
)


def distance_in_km(distance):
    match = re.search(r'([\d.]+)\s*(km|kilometer|kilometers|m|meter|meters)', str(distance).lower())
    if not match:
        return None

    value = float(match.group(1))
    return value / 1000 if match.group(2).startswith(('m', 'meter')) else value


def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3
    property_index = property_names.index(property_name)
    sim_scores = list(enumerate(cosine_sim_matrix[property_index]))

    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    top_properties = [property_names[index] for index in top_indices]

    # Create a dataframe with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return recommendations_df


st.title('Select Location and Radius')

location_options = sorted({
    location
    for advantages in location_df['ParsedLocationAdvantages']
    for location in advantages
})
selected_location = st.selectbox('Location', location_options)

radius = st.number_input('Radius in Kms', min_value=0.0, value=5.0)

if st.button('Search'):
    nearby_properties = []
    for _, property_row in location_df.iterrows():
        distance = property_row['ParsedLocationAdvantages'].get(selected_location)
        distance_km = distance_in_km(distance)
        if distance_km is not None and distance_km <= radius:
            nearby_properties.append({
                'PropertyName': property_row['PropertyName'],
                'Distance (km)': round(distance_km, 2),
            })

    st.dataframe(pd.DataFrame(nearby_properties), hide_index=True)

st.title('Recommend Appartments')
selected_appartment = st.selectbox('Select an appartment', property_names)

if st.button('Recommend'):
    recommendation_df = recommend_properties_with_scores(selected_appartment)

    st.dataframe(recommendation_df)