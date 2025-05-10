import streamlit as st
import requests
import base64
import io
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="👗",
    layout="wide"
)

# API configuration
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

def check_api_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get("status") == "healthy", health_data
        return False, None
    except:
        return False, None

def get_recommendations(image_file, num_recommendations=5):
    """Get recommendations from the API"""
    try:
        # Prepare the file for upload
        files = {"file": image_file}
        params = {"num_recommendations": num_recommendations}
        
        # Make API request
        response = requests.post(
            f"{API_URL}/upload-file",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def display_recommendations(recommendations):
    """Display recommendation results in a grid"""
    if not recommendations or "recommendations" not in recommendations:
        st.warning("No recommendations received")
        return
    
    items = recommendations["recommendations"]
    
    # Create a grid of recommendations
    cols_per_row = 3
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(items):
                item = items[i + j]
                
                with col:
                    # Display image
                    image_url = f"{API_URL}{item['image_url']}" if item['image_url'].startswith('/') else item['image_url']
                    
                    try:
                        st.image(image_url, use_column_width=True)
                    except:
                        st.write("🖼️ Image not available")
                    
                    # Display product info
                    st.write(f"**{item.get('product_name', 'Unknown Product')}**")
                    
                    if 'article_type' in item:
                        st.write(f"Type: {item['article_type']}")
                    
                    if 'color' in item:
                        st.write(f"Color: {item['color']}")
                    
                    # Display similarity score as progress bar
                    similarity = item.get('similarity', 0)
                    st.progress(similarity)
                    st.write(f"Similarity: {similarity:.1%}")

def main():
    # Title and description
    st.title("Fashion Recommendation System")
    st.write("Upload an image to find similar fashion items")
    
    # Sidebar: API health check
    with st.sidebar:
        st.header("System Status")
        is_healthy, health_data = check_api_health()
        
        if is_healthy:
            st.success("✅ API is online")
            if health_data:
                st.write(f"Products loaded: {health_data.get('products_count', 0)}")
        else:
            st.error("❌ API is offline")
            st.write("Please check if the API server is running")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # File upload section
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a fashion item image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a fashion item"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Number of recommendations slider
            num_recommendations = st.slider(
                "Number of recommendations",
                min_value=1,
                max_value=10,
                value=5
            )
            
            # Get recommendations button
            if st.button("Find Similar Items", type="primary"):
                with st.spinner("Finding similar fashion items..."):
                    # Reset file pointer for upload
                    uploaded_file.seek(0)
                    
                    # Get recommendations
                    recommendations = get_recommendations(uploaded_file, num_recommendations)
                    
                    # Store in session state to persist
                    if recommendations:
                        st.session_state['recommendations'] = recommendations
    
    with col2:
        # Display recommendations
        st.subheader("Recommendations")
        
        # Show recommendations from session state
        if 'recommendations' in st.session_state:
            display_recommendations(st.session_state['recommendations'])
        else:
            st.info("Upload an image and click 'Find Similar Items' to see recommendations")
    
    # Footer with information
    st.markdown("---")
    st.markdown("""
    ### About
    This fashion recommendation system uses deep learning to find visually similar items.
    - **Model**: DINOv2 for feature extraction
    - **Algorithm**: K-Nearest Neighbors for similarity search
    - **Dataset**: Fashion product images
    """)

if __name__ == "__main__":
    main()