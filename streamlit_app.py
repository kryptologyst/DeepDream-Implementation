# Streamlit Web UI for Advanced DeepDream
# Modern, interactive interface for DeepDream generation

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
from advanced_deepdream import AdvancedDeepDream, ImageDatabase
import time

# Page configuration
st.set_page_config(
    page_title="🎨 Advanced DeepDream Generator",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_deepdream_model(model_name="inception_v3"):
    """Load DeepDream model with caching"""
    return AdvancedDeepDream(model_name=model_name)

@st.cache_resource
def load_image_database():
    """Load image database with caching"""
    return ImageDatabase()

def create_sample_images():
    """Create sample images for the database"""
    db = load_image_database()
    sample_images = []
    
    for category_images in db.get_all_images():
        for img_info in category_images:
            image_path = db.create_sample_image(img_info['name'])
            sample_images.append({
                'path': image_path,
                'name': img_info['name'],
                'description': img_info['description'],
                'category': img_info['category']
            })
    
    return sample_images

def display_image_comparison(original, dream, title="DeepDream Result"):
    """Display side-by-side comparison of original and dream images"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(original, use_column_width=True)
    
    with col2:
        st.subheader("💭 DeepDream Result")
        st.image(dream, use_column_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Advanced DeepDream Generator</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Welcome to the Advanced DeepDream Generator!</strong><br>
    This tool uses state-of-the-art neural networks to create surreal, dream-like images by amplifying patterns 
    that the AI "sees" in your photos. Choose from multiple models, layers, and advanced techniques.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Model selection
    model_options = {
        'InceptionV3': 'inception_v3',
        'ResNet50': 'resnet50', 
        'VGG16': 'vgg16'
    }
    
    selected_model = st.sidebar.selectbox(
        "🤖 Choose AI Model",
        options=list(model_options.keys()),
        help="Different models produce different artistic styles"
    )
    
    # Method selection
    method_options = {
        'Octave Scaling': 'octave',
        'Layer Blending': 'blend',
        'Guided DeepDream': 'guided'
    }
    
    selected_method = st.sidebar.selectbox(
        "🎭 DeepDream Method",
        options=list(method_options.keys()),
        help="Different methods create different visual effects"
    )
    
    # Layer selection
    layer_options = {
        'Early Layers (Simple patterns)': 'early',
        'Mid Layers (Complex shapes)': 'mid',
        'Late Layers (Abstract concepts)': 'late'
    }
    
    selected_layer = st.sidebar.selectbox(
        "🔍 Neural Layer",
        options=list(layer_options.keys()),
        help="Different layers capture different levels of visual complexity"
    )
    
    # Advanced parameters
    st.sidebar.markdown("### 🔧 Advanced Parameters")
    
    iterations = st.sidebar.slider(
        "Iterations",
        min_value=5,
        max_value=50,
        value=20,
        help="More iterations = more detailed dreams (but slower)"
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        help="Higher values = more dramatic changes"
    )
    
    # Method-specific parameters
    if selected_method == 'Octave Scaling':
        octave_n = st.sidebar.slider(
            "Number of Octaves",
            min_value=2,
            max_value=6,
            value=4,
            help="More octaves = better detail preservation"
        )
        
        octave_scale = st.sidebar.slider(
            "Octave Scale Factor",
            min_value=1.1,
            max_value=2.0,
            value=1.4,
            step=0.1,
            help="Scale factor between octaves"
        )
    
    elif selected_method == 'Layer Blending':
        st.sidebar.markdown("**Layer Weights:**")
        early_weight = st.sidebar.slider("Early Layer Weight", 0.0, 1.0, 0.2)
        mid_weight = st.sidebar.slider("Mid Layer Weight", 0.0, 1.0, 0.6)
        late_weight = st.sidebar.slider("Late Layer Weight", 0.0, 1.0, 0.2)
        
        # Normalize weights
        total_weight = early_weight + mid_weight + late_weight
        if total_weight > 0:
            early_weight /= total_weight
            mid_weight /= total_weight
            late_weight /= total_weight
    
    elif selected_method == 'Guided DeepDream':
        guide_weight = st.sidebar.slider(
            "Guide Influence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            help="How much the guide image influences the result"
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎨 Generate Dreams", "📚 Sample Gallery", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🎨 Generate Your DeepDream</h2>', unsafe_allow_html=True)
        
        # Image upload
        uploaded_file = st.file_uploader(
            "📁 Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload your own image to transform into a dream"
        )
        
        # Sample images
        st.markdown("**Or choose from sample images:**")
        sample_images = create_sample_images()
        
        col1, col2, col3 = st.columns(3)
        selected_sample = None
        
        for i, img_info in enumerate(sample_images[:6]):  # Show first 6 samples
            with [col1, col2, col3][i % 3]:
                if st.button(f"📸 {img_info['name']}", key=f"sample_{i}"):
                    selected_sample = img_info['path']
        
        # Guide image for guided DeepDream
        guide_file = None
        if selected_method == 'Guided DeepDream':
            st.markdown("**🎯 Upload a guide image:**")
            guide_file = st.file_uploader(
                "Guide Image",
                type=['png', 'jpg', 'jpeg'],
                help="This image will guide the style of your DeepDream"
            )
        
        # Generate button
        if st.button("🚀 Generate DeepDream", type="primary"):
            if uploaded_file is not None or selected_sample is not None:
                
                # Determine which image to use
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    image_name = uploaded_file.name
                else:
                    image = Image.open(selected_sample)
                    image_name = os.path.basename(selected_sample)
                
                # Save uploaded image temporarily
                temp_path = f"temp_{image_name}"
                image.save(temp_path)
                
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Load model
                    status_text.text("🤖 Loading AI model...")
                    progress_bar.progress(20)
                    
                    deepdream = load_deepdream_model(model_options[selected_model])
                    
                    # Prepare parameters
                    params = {
                        'iterations': iterations,
                        'lr': learning_rate
                    }
                    
                    if selected_method == 'Octave Scaling':
                        params.update({
                            'octave_n': octave_n,
                            'octave_scale': octave_scale
                        })
                    elif selected_method == 'Layer Blending':
                        params['layer_weights'] = {
                            'early': early_weight,
                            'mid': mid_weight,
                            'late': late_weight
                        }
                    elif selected_method == 'Guided DeepDream':
                        if guide_file is not None:
                            guide_image = Image.open(guide_file)
                            guide_path = f"temp_guide_{guide_file.name}"
                            guide_image.save(guide_path)
                            params['guide_path'] = guide_path
                        else:
                            st.error("Please upload a guide image for Guided DeepDream")
                            return
                    
                    # Generate DeepDream
                    status_text.text("💭 Generating DeepDream...")
                    progress_bar.progress(50)
                    
                    dream_result = deepdream.generate_dream(
                        temp_path,
                        method=method_options[selected_method],
                        layer=layer_options[selected_layer],
                        **params
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("✨ Finalizing result...")
                    
                    # Display results
                    display_image_comparison(image, dream_result, f"DeepDream: {image_name}")
                    
                    # Download button
                    dream_pil = Image.fromarray((dream_result * 255).astype(np.uint8))
                    
                    # Convert to bytes for download
                    img_buffer = io.BytesIO()
                    dream_pil.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="💾 Download DeepDream",
                        data=img_buffer.getvalue(),
                        file_name=f"deepdream_{image_name}",
                        mime="image/png"
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    st.markdown("""
                    <div class="success-box">
                    <strong>🎉 DeepDream generated successfully!</strong><br>
                    Your surreal creation is ready. Try different parameters to explore various artistic styles.
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error generating DeepDream: {str(e)}")
                    st.markdown("""
                    <div class="warning-box">
                    <strong>💡 Troubleshooting Tips:</strong><br>
                    • Try reducing the number of iterations<br>
                    • Use a smaller learning rate<br>
                    • Make sure your image is in a supported format (PNG, JPG)<br>
                    • Try a different model or layer
                    </div>
                    """, unsafe_allow_html=True)
                
                finally:
                    # Clean up temporary files
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if 'guide_path' in locals() and os.path.exists(guide_path):
                        os.remove(guide_path)
            
            else:
                st.warning("⚠️ Please upload an image or select a sample image")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📚 Sample Image Gallery</h2>', unsafe_allow_html=True)
        
        # Display sample images by category
        categories = {}
        for img_info in sample_images:
            category = img_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(img_info)
        
        for category, images in categories.items():
            st.markdown(f"### {category.title()} Images")
            
            cols = st.columns(min(len(images), 3))
            for i, img_info in enumerate(images):
                with cols[i % 3]:
                    st.image(img_info['path'], caption=img_info['description'], use_column_width=True)
                    st.caption(f"📁 {img_info['name']}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">ℹ️ About DeepDream</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🧠 What is DeepDream?
        
        DeepDream is a computer vision technique that uses deep neural networks to find and enhance patterns in images. 
        It works by maximizing the activation of specific neurons in a pre-trained network, creating surreal, 
        dream-like visuals that reveal what the AI "sees" in your images.
        
        ## 🎨 Features of This Implementation
        
        ### **Advanced Techniques:**
        - **Octave Scaling**: Multi-scale processing for better detail preservation
        - **Layer Blending**: Combine multiple neural layers for complex effects
        - **Guided DeepDream**: Use a reference image to guide the style
        
        ### **Multiple Models:**
        - **InceptionV3**: Great for organic, flowing patterns
        - **ResNet50**: Good balance of detail and abstraction
        - **VGG16**: Excellent for texture and pattern enhancement
        
        ### **Layer Selection:**
        - **Early Layers**: Simple patterns, edges, textures
        - **Mid Layers**: Complex shapes, objects, structures
        - **Late Layers**: Abstract concepts, high-level features
        
        ## 🔧 Technical Details
        
        This implementation uses modern PyTorch with the latest best practices:
        - Updated model loading (no deprecated `pretrained=True`)
        - GPU acceleration when available
        - Proper image preprocessing and normalization
        - Advanced optimization techniques
        
        ## 🚀 Getting Started
        
        1. **Upload an Image**: Choose any photo you'd like to transform
        2. **Select Parameters**: Experiment with different models, layers, and methods
        3. **Generate**: Click the generate button and watch the magic happen
        4. **Download**: Save your unique DeepDream creation
        
        ## 💡 Tips for Best Results
        
        - **Nature photos** work great with InceptionV3
        - **Architecture** looks amazing with ResNet50
        - **Abstract art** benefits from VGG16
        - **Higher iterations** = more detailed dreams
        - **Lower learning rates** = smoother transitions
        """)
        
        st.markdown("""
        <div class="info-box">
        <strong>🔬 Technical Implementation:</strong><br>
        This project demonstrates advanced computer vision techniques including gradient ascent, 
        feature visualization, and neural network interpretability. The code is production-ready 
        and follows modern Python best practices.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
