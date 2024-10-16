
import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths
data_dir = 'dataset/Arecanut_dataset/Arecanut_dataset'
model_path = 'leaf_disease_model.h5'
segmentation_model_path = 'segmentation_model.h5'

# Load models
model = load_model(model_path)
segmentation_model = load_model(segmentation_model_path)

# Class names
class_names = ['Healthy_Leaf', 'Healthy_Nut', 'Healthy_Trunk', 'Mahali_Koleroga', 'Stem_bleeding', 'black_pepper_healthy', 'black_pepper_leaf_blight', 'black_pepper_yellow_mottle_virus', 'bud borer', 'healthy_foot', 'leaf spot disease', 'stem cracking', 'yellow leaf disease']
# Function to preprocess and classify the image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    st.write(f"Predicted class index: {class_idx}")  # For debugging
    return class_names[class_idx]

# Function to segment the image and calculate severity
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
    st.write(f"Affected Pixels: {affected_pixels}")  # For debugging
    st.write(f"Total Pixels: {total_pixels}")        # For debugging
    st.write(f"Severity: {severity:.2f}%")          # For debugging
    st.write(f"Adjusted Severity: {adjusted_severity:.2f}%")  # For debugging
    return mask, adjusted_severity

disease_guidelines = {
    "yellow leaf disease": {
        "cause": "FPhytoplasma",
        "guidelines": {
            "low_severity": {
                "phytosanitation": "Remove and destroy the infected leaves.",
                "treatment": "Spray with Propiconazole 25% EC (1 ml per litre of water).",
                "precautions": ["Avoid overhead irrigation.", "Monitor leaf conditions regularly.", "Avoid moving planting materials from affected areas."]
            },
            "mid_severity": {
                "phytosanitation": "Remove and burn infected leaves.",
                "treatment": "Spray with Tebuconazole (1 ml per litre of water) or Hexaconazole (1 ml per litre) every 20 days.",
                "nutrient_management": "Apply soil-test-based fertilizers for rejuvenation."
            },
            "high_severity": {
                "phytosanitation": "Remove and destroy severely infected plants to prevent the spreading of disease.",
                "treatment": "Spray with Propineb 70% WP (2 g per litre) every 25-30 days. Add sticker solution.",
                "community_approach": "Adopt community-level spraying. Avoid moving planting materials from affected areas."
            }
        }
    },
    "black_pepper_yellow_mottle_virus": {
        "cause": "Viral infection spread through insect vectors",
        "guidelines": {
            "low_severity": {
                "phytosanitation": "Remove and destroy infected leaves.",
                "treatment": "Apply insecticides to control vectors.",
                "precautions": ["Quarantine planting material from infected areas."]
            },
            "mid_severity": {
                "phytosanitation": "Prune infected areas.",
                "treatment": "Apply neem oil-based pesticides.",
                "nutrient_management": "Apply micronutrient foliar sprays (e.g., zinc, boron)."
            },
            "high_severity": {
                "phytosanitation": "Burn heavily infected parts.",
                "treatment": "Apply systemic insecticides for vector control.",
                "community_approach": "Coordinate with neighboring farmers to manage vector activity."
            }
        }
    },
    "leaf spot disease": {
        "cause": "Fungal pathogens",
        "guidelines": {
            "low_severity": {
                "phytosanitation": "Remove affected leaves and debris.",
                "treatment": "Apply fungicides like Copper Oxychloride.",
                "precautions": ["Avoid overhead irrigation.", "Do not work in the field when foliage is wet."]
            },
            "mid_severity": {
                "phytosanitation": "Increase the frequency of leaf inspections.",
                "treatment": "Use systemic fungicides such as Propiconazole.",
                "nutrient_management": "Apply balanced fertilizers to strengthen plant health."
            },
            "high_severity": {
                "phytosanitation": "Destroy severely affected plants to halt spread.",
                "treatment": "Use broad-spectrum fungicides; repeat applications every 7-10 days.",
                "community_approach": "Encourage neighboring farms to adopt similar treatments."
            }
        }
    },
    "black_pepper_leaf_blight": {
        "cause": "Fungal infections due to excess moisture",
        "guidelines": {
            "low_severity": {
                "phytosanitation": "Trim back affected areas.",
                "treatment": "Apply fungicides containing Mancozeb.",
                "precautions": ["Avoid late evening watering to reduce humidity."]
            },
            "mid_severity": {
                "phytosanitation": "Remove all infected leaves.",
                "treatment": "Utilize fungicides such as Thiophanate-methyl.",
                "nutrient_management": "Enhance potassium levels in soil to bolster plant health."
            },
            "high_severity": {
                "phytosanitation": "Completely remove and destroy infected plants.",
                "treatment": "Intensive fungicide treatment and rotation of fungicide classes.",
                "community_approach": "Establish a cooperative approach for controlling moisture levels in the area."
            }
        }
    }
    # Add other disease guidelines here as needed
}

# Streamlit UI
# Streamlit UI
st.title("Leaf Disease Classification and Severity Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(file_path, caption='Uploaded Image', use_column_width=True)

    result = classify_image(file_path)
    st.write(f"The leaf is classified as: {result}")

    if result in ["leaf spot disease", "yellow leaf disease", "black_pepper_yellow_mottle_virus", "black_pepper_leaf_blight"]:
        mask, severity = segment_and_calculate_severity(file_path)
        st.write(f"Severity of the disease is: {severity:.2f}%")
        
        if severity < 5:
            severity_level = "low_severity"
        elif severity < 20:
            severity_level = "mid_severity"
        else:
            severity_level = "high_severity"
        
        st.write(f"Severity Level: {severity_level.replace('_', ' ').title()}")
        
        if result in disease_guidelines:
            if severity_level in disease_guidelines[result]["guidelines"]:
                guidelines = disease_guidelines[result]["guidelines"][severity_level]
                
                for key, value in guidelines.items():
                    if isinstance(value, list):
                        st.write(f"**{key.replace('_', ' ').title()}**:")
                        for item in value:
                            st.write(f" - {item}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}**: {value}")
            else:
                st.write("Severity level not found in the guidelines.")
        else:
            st.write("Disease not recognized in the guideline list.")
        
        st.image(mask * 255, caption='Segmented Mask', use_column_width=True)
    else:
        st.write("Disease not recognized in the guideline list.")
