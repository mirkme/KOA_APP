import numpy as np
import streamlit as st
import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.applications import DenseNet121, ResNet50, Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
import plotly.express as px



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



#streamlit page setting
icon = Image.open("mdc.png")
st.set_page_config(page_title="Knee Osteoarthritis", page_icon=icon)



#Apply CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #D6EAF8, #EAF4FC);
    }
    [data-testid="stSidebar"] {
        background-color: #F5F5F5; /* grey */
    }
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #000000 !important;  /* Black */
    }
    .welcome-container {
        text-align: center;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8); /* semi transparent white box */
        border-radius: 10px;
        margin: 20px 0;
    }
    .placeholder-image {
        opacity: 0;
        animation: fadeIn 1s ease-in forwards;
        max-width: 200px;
        margin: 20px auto;
        display: block;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 0.7; }
    }
    .process-steps {
        margin-top: 20px;
        font-size: 14px;
        color: #555;
    }
    .custom-header {
        text-align: center;
        font-size: 140px;
        font-weight: bold;
        color: #000000;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)



#class labels
class_names = ["Healthy", "Minimal", "Doubtful", "Moderate", "Severe"]



#constants for models
target_size1 = (224, 224)
img_shape1 = (224, 224, 3)
target_size2 = (150, 150)



#MODELs-binary classifier,densenet, resnet, xception


#binary classifier
model = load_model('C:\\Users\\Asus\\Desktop\\KOA_APP\\src\\models\\knee_xray_classifier1.keras', custom_objects={'InputLayer': InputLayer})


# densnet121
# Base model
base_densenet = DenseNet121(
    input_shape=img_shape1,
    include_top=False,
    weights="imagenet"
)
# Fine-tune
for layer in base_densenet.layers:
    layer.trainable = True
# Add classifier on top
model_densenet = Sequential([
    base_densenet,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(5, activation="softmax")
])
# Load weights
model_densenet.load_weights("C:\\Users\\Asus\\Desktop\\KOA_APP\\src\\models\\100 epochs org dataset DenseNet121_ft.hdf5")


# resnet50
# Base model
base_resnet = ResNet50(
    input_shape=img_shape1,
    include_top=False,
    weights='imagenet'
)
# Make all layers trainable
for layer in base_resnet.layers:
    layer.trainable = True
# Build final model
model_resnet = Sequential([
    base_resnet,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(5, activation="softmax")
])
#Load weights
model_resnet.load_weights("C:\\Users\\Asus\\Desktop\\KOA_APP\\src\\models\\model_ResNet50_ft.hdf5")


#Xception
# Base model
base_xception = Xception(
    input_shape=img_shape1,
    include_top=False,
    weights="imagenet"
)
for layer in base_xception.layers:
    layer.trainable = True
model_xception = Sequential([
    base_xception,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(5, activation="softmax")
])
model_xception.load_weights("C:\\Users\\Asus\\Desktop\\KOA_APP\\src\\models\\model_Xception_ft.hdf5")



# Knee X-ray classifier
def classify_if_xray(img_array,model):
    # preprocess
    img_array_proc = np.expand_dims(img_array, axis=0) 
    img_array_proc = img_array_proc / 255.0             

    # prediction
    pred = model.predict(img_array_proc, verbose=0)[0][0]

    # decision
    if pred >= 0.5:                  # NOT a Knee X‑ray
        # Optionally log/print confidence if desired
        # print(f"NOT a Knee X-ray (confidence {pred:.2%})")
        return  False, pred, img_array
    else:                            # Knee X‑ray
        # print(f"Knee X-ray (confidence {(1 - pred):.2%})")
        return True, 1 - pred, img_array



def run_ensemble_prediction(img_array):
    try:
        # Preprocess
        img_array_xception = tf.keras.applications.xception.preprocess_input(np.copy(img_array))
        img_array_densenet = tf.keras.applications.densenet.preprocess_input(np.copy(img_array))
        img_array_resnet = tf.keras.applications.resnet50.preprocess_input(np.copy(img_array))

        # Prediction
        pred_xception = model_xception.predict(img_array_xception)
        pred_densenet = model_densenet.predict(img_array_densenet)
        pred_resnet = model_resnet.predict(img_array_resnet)

        # Ensemble
        y_pred = (pred_xception + pred_densenet + pred_resnet) / 3
        return y_pred[0] 
    except Exception as e:
        print("Prediction error:", e)
        return None



# Precautions and Medications
precautions = {
    "Doubtful": [
        "Avoid high-impact activities",
        "Maintain a healthy weight",
        "Strengthen muscles around the knee",
        "Practice low-impact exercises",
        "Use supportive footwear"
    ]
}

medications = {
    "Minimal": [
        "Over-the-counter pain relievers",
        "Physical therapy",
        "Light exercises",
        "Warm compresses for stiffness",
        "Knee support if needed"
    ],
    "Moderate": [
        "Prescription pain medication",
        "Weight management",
        "Corticosteroid injections",
        "Physical therapy",
        "Joint support and braces"
    ],
    "Severe": [
        "Joint replacement consultation",
        "Strong prescription medications",
        "Intra-articular hyaluronic acid injections",
        "Customized physiotherapy",
        "Pain management plan"
    ]
}



#User Interface
with st.sidebar:
    st.image(icon)
    st.subheader("Group 4")
    st.markdown("**AI-Powered Knee X-Ray Analysis**")
    st.markdown("Upload a knee X-ray to get an AI-powered osteoarthritis grading.")
    st.subheader(":arrow_up: Upload Image")
    uploaded_file = st.file_uploader("Choose X-ray image")

#title
st.markdown('<div class="custom-header">OsteoGrade</div>',unsafe_allow_html=True)

#welcome message
if uploaded_file is None:
    st.markdown(
        """
        <div class="welcome-container">
            <h3>Welcome to AI-Powered Knee Osteoarthritis Grading</h3>
            <p>Upload a knee X-ray image to get started.</p>
            <p>Powered by advanced AI models for accurate grading.</p>
            <img src="https://img.icons8.com/ios/200/knee-joint.png" class="placeholder-image" alt="Knee Joint Placeholder">
            <div class="process-steps">
                <p><b>How it works:</b> 1. Upload your X-ray → 2. AI analyzes the image → 3. Get your grading results.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Create two-column layout for input and analysis
col1, col2 = st.columns(2)
y_pred = None

if uploaded_file is not None:
    with col1:
        st.subheader("Input")
        st.image(uploaded_file, width=300)# Display uploaded image

        # Load and preprocess image for binary classifier
        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=target_size2)
        img = tf.keras.preprocessing.image.img_to_array(img)

        # Verify if image
        is_knee, conf_knee, img_pass=classify_if_xray(img,model)
        if not is_knee:
            st.error(f"This doesn't look like a knee X-ray "
                f"(confidence{(1-conf_knee)*100:.1f}%). "
                "Please upload a valid knee X-ray image.")
            st.stop()

        # Load and preprocess image for ensemble prediction
        img_ensemble = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=target_size1) 
        img_ensemble = tf.keras.preprocessing.image.img_to_array(img_ensemble)
        img_array_ensemble = np.expand_dims(img_ensemble, axis=0).astype(np.float32)

        # Run prediction on button click
        if st.button(":arrows_counterclockwise: Predict"):
            with st.spinner("Running ensemble prediction..."):
                y_pred = run_ensemble_prediction(img_array_ensemble)

        # Process prediction
        if y_pred is not None:
            probability = np.max(y_pred) * 100
            predicted_class = class_names[np.argmax(y_pred)]
            st.success("Prediction")
            st.metric(
                 label="Prediction",
                value=f"{predicted_class} - {probability:.2f}%",
            )
    
    #display analysis
    if y_pred is not None:
        with col2:
            st.subheader(":bar_chart: Analysis")
            # Convert y_pred to percentages
            y_pred_percent = [p * 100 for p in y_pred]
            # bar chart
            fig = px.bar(
                x=y_pred_percent,
                y=class_names,
                orientation='h',
                text=[f"{p:.2f}%" for p in y_pred_percent],  # Display percentages on bars
                range_x=[0, 120],
            )
            fig.update_traces(textposition='outside')  # Position text outside the bars
            fig.update_layout(
                xaxis_title="",
                yaxis_title="",
                xaxis=dict(tickvals=list(range(0, 101, 20))),  # Match the x-axis ticks
                showlegend=False,
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show recommendations based on predicted class
            if predicted_class == "Doubtful":
                st.subheader(":pill: Recommended Precautions for Doubtful Condition")
                for item in precautions["Doubtful"]:
                    st.write("- " + item)
            elif predicted_class in medications:
                st.subheader(":pill: Recommended Medications for " + predicted_class)
                for item in medications[predicted_class]:
                    st.write("- " + item)
    else:
        st.error("Press Prediction Button to get the Results.")
