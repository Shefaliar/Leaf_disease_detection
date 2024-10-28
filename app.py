import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

# Paths
data_dir = 'dataset/Arecanut_dataset/Arecanut_dataset'
model_path = 'leaf_disease_model.h5'
segmentation_model_path = 'segmentation_model.h5'

# Load models
model = load_model(model_path)
segmentation_model = load_model(segmentation_model_path)

# Class names and their display names
class_names = [
    'Healthy_Leaf', 'Healthy_Nut', 'Healthy_Trunk', 'Mahali_Koleroga', 
    'Stem_bleeding', 'black_pepper_healthy', 'black_pepper_leaf_blight', 
    'black_pepper_yellow_mottle_virus', 'bud borer', 'healthy_foot', 
    'leaf spot disease', 'stem cracking', 'yellow leaf disease'
]

display_names = {
    'yellow leaf disease': 'Yellow Leaf Disease in Arecanut',
    'leaf spot disease': 'Leaf Spot Disease in Arecanut',
    'black_pepper_yellow_mottle_virus': 'Yellow Mottle Virus in Black Pepper',
    'black_pepper_leaf_blight': 'Leaf Blight in Black Pepper',
    'Healthy_Leaf': 'Healthy Arecanut Leaf',
    'Healthy_Nut': 'Healthy Nut',
    'Healthy_Trunk': 'Healthy Trunk',
    'Mahali_Koleroga': 'Mahali Koleroga',
    'Stem_bleeding': 'Stem Bleeding',
    'black_pepper_healthy': 'Healthy Black Pepper Leaf',
    'bud borer': 'Bud Borer',
    'healthy_foot': 'Healthy Foot',
    'stem cracking': 'Stem Cracking'
}

# Load and encode background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    .title {{
        color: #228B22;
        font-weight: bold;
        font-size: 3em;
    }}
    .subheader {{
        color: #000000;
        font-size: 1.5em;
    }}
    .severity-low {{
        color: green;
        font-weight: bold;
    }}
    .severity-medium {{
        color: orange;
        font-weight: bold;
    }}
    .severity-high {{
        color: red;
        font-weight: bold;
    }}
    .classification-result {{
        color: black;
        font-weight: bold;
        font-size: 1.5em;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background image
set_background('back.jpg')

# App content
st.markdown('<h1 class="title"> Disease Detection & Severity Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subheader">Upload an image of a leaf to classify and analyze its condition.</h3>', unsafe_allow_html=True)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
disease_causes = {
"leaf spot disease": "Fungus *Colletotrichum*",
"black_pepper_leaf_blight": "Bacteria's belonging to species *Xanthomonas*, *Pseudomonas*",
"black_pepper_yellow_mottle_virus": "*Piper yellow mottle virus*",
"yellow leaf disease": "*Areca palm velarivirus 1*"
}

def classify_image(img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    predicted_class = class_names[class_idx]
    
    # Only show specific messages for healthy leaves
    if predicted_class == 'Healthy_Leaf':
        return 'Healthy Arecanut Leaf'
    elif predicted_class == 'black_pepper_healthy':
        return 'Healthy Black Pepper Leaf'
    else:
        return predicted_class
def segment_and_calculate_severity(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    mask = segmentation_model.predict(img_resized)
    mask = (mask > 0.05).astype(np.uint8)
    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    affected_pixels = np.sum(mask)
    total_pixels = mask.size
    severity = (affected_pixels / total_pixels) * 100
    adjusted_severity = min(severity * 2.5, 100)
    return mask, adjusted_severity



if uploaded_file is not None:
    # Create 'uploads' directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Save and display the uploaded image
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(file_path, caption='Uploaded Leaf Image', use_column_width=True)

    # Classify and display result
    result = classify_image(file_path)
    display_result = display_names.get(result, result)  # Get the display name from the mapping

    if "Healthy" in display_result:
        st.markdown(f"### <span style='color: black; font-weight: bold;'>Result: {display_result}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"### <span style='color: black; font-weight: bold;'>Predicted Disease: {display_result}</span>", unsafe_allow_html=True)
        
        # Display cause of the disease if it exists in the dictionary
        if result in disease_causes:
            cause = disease_causes[result]
            st.markdown(f"### <span style='color: black;'>Cause of the Disease: <span style='color: black;'>{cause}</span>", unsafe_allow_html=True)
    
    # Only show severity analysis and control measures if it’s a disease
    if result in ["leaf spot disease", "yellow leaf disease", "black_pepper_yellow_mottle_virus", "black_pepper_leaf_blight"]:
        mask, severity = segment_and_calculate_severity(file_path)
        
        # Define severity level and styling
        if severity < 5:
            severity_level = "low"
            severity_style = "severity-low"
        elif severity < 15:
            severity_level = "medium"
            severity_style = "severity-medium"
        else:
            severity_level = "high"
            severity_style = "severity-high"

        # Display severity and severity level in the appropriate color
        st.markdown(f"### <span style='color: black;'>Severity of the disease: <span class='{severity_style}'>{severity:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"### <span style='color: black;'>Severity Level : <span class='{severity_style}'>{severity_level.capitalize()}</span>", unsafe_allow_html=True)
        st.image(mask * 255, caption='Segmented Mask', use_column_width=True, clamp=True)
# Display result and manage display names


def segment_and_calculate_severity(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    mask = segmentation_model.predict(img_resized)
    mask = (mask > 0.05).astype(np.uint8)
    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    affected_pixels = np.sum(mask)
    total_pixels = mask.size
    severity = (affected_pixels / total_pixels) * 100
    adjusted_severity = min(severity * 2.5, 100)
    return mask, adjusted_severity



# Control measures dictionary template for all diseases with severity levels
control_measures = {
    "black_pepper_leaf_blight": {
        "low": [
            "Separate the plants with mild infection.",
            "Remove affected leaves and destroy them to reduce the spread to healthy vines."
        ],
        "medium": [
            "Spray Bordeaux mixture (1%) on affected plants.",
            "Remove any severely affected vines.",
            "Ensure phytosanitation ie. removal and burning of affected vines practices to reduce spread."
        ],
        "high": [
            "1% Bordeaux mixture can be applied during monsoon seaso.",
            "metalaxyl and fosetyl are also effective fugicides.",                  
            "Consider applying systemic fungicides, especially during the monsoon season in the future.",
            "Remove and destroy all affected vines to prevent spread." 
            
        ]
    },
    "yellow leaf disease": {
        "low": [
            "Regular application of fertilizer at the rate of 100g N, 40g P₂O₅, and 140g K₂O per palm per year.",
            "Additional superphosphate application (1 kg/palm) alone or with lime (1 kg/palm).",
            "Irrigation every four days during summer months.",
            
        ],
        "medium": [
            "Manuring with green leaf and compost at 12 kg per palm.",
            "Soil application of Blue Copper 50 + Thimet 10 G (100 g each/palm).",
            
        ],
        "high": [
            
            "Removal of diseased plants to prevent the spread of disease."
            "Replant with the Mangala variety of palm, known to be less susceptible to Yellow Leaf Disease.",
            "Use planting material from plants that have shown resistancein the future to the disease to increase disease tolerance",
            "Source seed materials from disease-free areas to reduce the risk of future infections."
            
        ]
    },
    "leaf spot disease": {
        "low": [
            "Phytosanitation: Remove and burn mildly infected leaves.",
            "Fungicides: Spray affected leaves with 0.3% Mancozeb or 0.2% Foltaf.",
            "Nutrient Management: Apply soil-test-based balanced nutrients to maintain healthy palms."
            
        ],
        "medium": [
            "Phytosanitation: Remove and burn severely infected leaves to reduce inoculum.",
            "Fungicides: First round: Spray with Propiconazole 25% EC (1 ml per liter), Tebuconazole (1 ml per liter), or Hexaconazole (1 ml per liter). Follow-up: After 25-30 days, spray with Propineb 70% WP (2 g per liter of water).",

            "Nutrient Management: Use soil-based nutrient applications to support plant resilience."
           
        ],
        "high": [
            "Phytosanitation: Implement community-level removal and burning of severely infected plant parts across affected areas.",
            "Conduct community-level spraying with Propiconazole, Tebuconazole, or Hexaconazole, followed by Propineb as described above."
            "Avoid moving planting materials from diseased areas to prevent further spread."
            "Community Approach: Adopt a community-level spraying initiative for effective disease management.",

        ]
    },
    "black_pepper_yellow_mottle_virus": {
        "low": [
            "Isolate infected plants to prevent virus spread.",
            "Remove affected leaves if virus symptoms are mild."
        ],
        "medium": [
            "Remove heavily infected parts to minimize virus spread.",
            "Monitor nearby plants closely for early symptoms."
        ],
        "high": [
            "Destroy severely infected plants to protect nearby crops.",
            "Implement vector control measures to prevent virus spread by insects."
        ]
    }
}

# Display control measures as bullet points for each disease based on severity
if result in control_measures:
    # Severity calculations and display logic
    mask, severity = segment_and_calculate_severity(file_path)
    
    # Define severity level and styling
    if severity < 5:
        severity_level = "low"
        severity_style = "severity-low"
    elif severity < 20:
        severity_level = "medium"
        severity_style = "severity-medium"
    else:
        severity_level = "high"
        severity_style = "severity-high"


    
    # Display control measures as bullet points
    st.markdown(f"### <span style='color: black; font-weight: bold;'>Control Measures:</span>", unsafe_allow_html=True)
    for measure in control_measures[result].get(severity_level, ["No specific control measures available."]):
        st.markdown(f"###<span style='color: black;'>- {measure}", unsafe_allow_html=True)
