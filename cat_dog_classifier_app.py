"""
Cat vs Dog Image Classifier
A deep learning application using MobileNetV2 for binary image classification
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .confidence-high {
        color: #00C851;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffbb33;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = tf.keras.models.load_model('models/cat_dog_model.keras')
        return model
    except:
        st.error("⚠️ Model not found! Please download the model first.")
        st.info("See setup instructions in README.md")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize to model input size
    img = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model, image):
    """Make prediction on image"""
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)
    return prediction[0][0]

def create_confidence_chart(confidence):
    """Create confidence visualization"""
    cat_conf = (1 - confidence) * 100
    dog_conf = confidence * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=[cat_conf, dog_conf],
            y=['Cat 🐱', 'Dog 🐶'],
            orientation='h',
            marker=dict(
                color=['#FF6B6B', '#4ECDC4'],
                line=dict(color='white', width=2)
            ),
            text=[f'{cat_conf:.1f}%', f'{dog_conf:.1f}%'],
            textposition='inside',
            textfont=dict(size=16, color='white')
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores",
        xaxis_title="Confidence (%)",
        height=250,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14)
    )
    
    fig.update_xaxis(range=[0, 100])
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🐾 Cat vs Dog Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>
            Upload an image of a cat or dog, and let AI predict which one it is!
        </p>
        <p style='color: #666;'>
            Powered by MobileNetV2 with 98.9% accuracy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a cat or dog"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Your Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
            # Make prediction
            with st.spinner('Analyzing...'):
                confidence = predict(model, image)
            
            # Determine prediction
            if confidence > 0.5:
                predicted_class = "Dog"
                emoji = "🐶"
                conf_pct = confidence * 100
            else:
                predicted_class = "Cat"
                emoji = "🐱"
                conf_pct = (1 - confidence) * 100
            
            # Confidence level
            if conf_pct >= 90:
                conf_class = "confidence-high"
                conf_text = "Very Confident"
            elif conf_pct >= 70:
                conf_class = "confidence-medium"
                conf_text = "Confident"
            else:
                conf_class = "confidence-low"
                conf_text = "Uncertain"
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style='margin: 0;'>{emoji} It's a {predicted_class}!</h2>
                <p style='font-size: 2rem; margin: 1rem 0;' class='{conf_class}'>
                    {conf_pct:.1f}% Confidence
                </p>
                <p style='margin: 0; opacity: 0.9;'>{conf_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence chart
            fig = create_confidence_chart(confidence)
            st.plotly_chart(fig, use_container_width=True)
        
        # Fun facts
        st.markdown("---")
        st.subheader("🎉 Fun Facts!")
        
        if predicted_class == "Dog":
            st.info("""
            **Did you know?**
            - Dogs have about 300 million olfactory receptors (humans have 6 million!)
            - A dog's sense of smell is 10,000 to 100,000 times stronger than humans
            - The world's oldest dog lived to 29 years old!
            - Dogs can understand up to 250 words and gestures
            """)
        else:
            st.info("""
            **Did you know?**
            - Cats spend 70% of their lives sleeping (12-16 hours per day!)
            - A cat's purr vibrates at 25-150 Hz, which can help heal bones
            - Cats have 32 muscles in each ear!
            - A group of cats is called a "clowder"
            """)
    
    else:
        # Example section
        st.markdown("---")
        st.subheader("📖 How to Use")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 1️⃣ Upload")
            st.write("Click the upload button and select a cat or dog image")
        
        with col2:
            st.markdown("### 2️⃣ Analyze")
            st.write("AI analyzes the image instantly")
        
        with col3:
            st.markdown("### 3️⃣ Results")
            st.write("Get prediction with confidence score")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p><strong>Technology:</strong> MobileNetV2 Transfer Learning | TensorFlow | Keras</p>
            <p><strong>Accuracy:</strong> 98.9% on validation set</p>
            <p>Built as a machine learning portfolio project</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
