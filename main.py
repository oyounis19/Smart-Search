import streamlit as st
import pandas as pd
from utils import load_ner_model, parse_query_with_ner, search_products, load_data

# Page config
st.set_page_config(
    page_title="Smart Search - OLX Products",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Smart Search - OLX Products")
st.markdown("Search for products using natural language in Arabic or English")

# Load model and data
@st.cache_resource
def initialize_app():
    model = load_ner_model()
    return model

# Initialize
model = initialize_app()
df = load_data()

if df.empty:
    st.stop()

# Initialize session state for query
if 'search_query' not in st.session_state:
    st.session_state.search_query = "Iphone 13 pro 256GB less than 60k"

# Search interface
st.markdown("### Enter your search query:")

query = st.text_input(
    "Search Query:",
    value=st.session_state.search_query,
    placeholder="Type your search here or use examples from sidebar...",
    help="You can search in Arabic or English"
)

if query:
    with st.spinner("Parsing query and searching..."):
        # Parse the query
        parsed = parse_query_with_ner(query, model)
        
        # Search products
        results, applied_filters = search_products(df, parsed)
    
    # Display results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🔍 Query Analysis")
        
        # Show parsed entities
        st.markdown("### **Extracted Filters:**")
        if parsed['brand']:
            st.write(f"**📱 Brand: {parsed['brand']}**")
        if parsed['model']:
            st.write(f"**📋 Model: {parsed['model']}**")
        if parsed['storage']:
            st.write(f"**💾 Storage: {parsed['storage']}**")
        if parsed['ram']:
            st.write(f"**🧠 RAM: {parsed['ram']}**")
        if parsed['color']:
            st.write(f"**🎨 Color: {parsed['color']}**")
        if parsed['price_max']:
            st.write(f"**💰 Max Price: {parsed['price_max']:,} EGP**")
        if parsed['battery_min']:
            st.write(f"**🔋 Min Battery: {parsed['battery_min']}%**")
        if parsed['condition']:
            st.write(f"**⚙️ Condition: {parsed['condition']}**")

        st.markdown(f"Found {len(results)} products")
    
    with col2:
        st.markdown("### 📱 Search Results")
        
        if len(results) == 0:
            st.warning("No products found matching your criteria. Try adjusting your search.")
        else:
            # Display results
            for idx, row in results.head(10).iterrows():  # Show top 10
                with st.container():
                    result_col1, result_col2 = st.columns([1, 3])
                    
                    with result_col1:
                        # Display image
                        if pd.notna(row['main_image']) and row['main_image']:
                            try:
                                st.image(row['main_image'], width=120)
                            except:
                                st.write("🖼️ Image not available")
                        else:
                            st.write("🖼️ No image")
                    
                    with result_col2:
                        # Product details
                        st.markdown(f"**{row['title']}**")
                        
                        # Price with emphasis
                        st.markdown(f"💰 **{row['price']:,.0f} EGP**")
                        
                        # Location and brand
                        info_parts = []
                        if pd.notna(row['location']):
                            info_parts.append(f"📍 {row['location']}")
                        if pd.notna(row['brand']):
                            info_parts.append(f"🏷️ {row['brand']}")
                        if pd.notna(row['seller_name']):
                            info_parts.append(f"👤 {row['seller_name']}")
                        
                        if info_parts:
                            st.write(" | ".join(info_parts))
                        
                        # Description preview
                        if pd.notna(row['description']):
                            desc = str(row['description'])[:150]
                            if len(str(row['description'])) > 150:
                                desc += "..."
                            st.write(f"📝 {desc}")
                        
                        # Link to OLX
                        if pd.notna(row['product_url']):
                            st.markdown(f"[🔗 View on OLX]({row['product_url']})")
                        
                        st.divider()

# Sidebar with info
with st.sidebar:
    st.markdown("### 🎯 Example Searches")
    st.markdown("Click any example to try it:")
    
    # Example search queries as buttons
    examples = [
        "Iphone 13 pro 256GB under 70,000 80% battery",
        "ايفون 14 برو بطاريه 80%",
        "Samsung A54 less than 15k",
        "iPhone 11 128GB white",
        "هواوي P50 اقل من 8 الف",
        "Samsung Galaxy S23 256GB",
        "iPhone 14 Pro Max sealed",
        "شاومي ريدمي نوت12",
        "OPPO Reno 8 128GB",
        "Samsung S23 8GB RAM"
    ]
    
    for example in examples:
        if st.button(example, key=f"btn_{example}", use_container_width=True):
            st.session_state.search_query = example
            st.rerun()
    
    st.divider()
    
    st.markdown("### 🛠️ Powered by")
    st.markdown("[NAMAA-Space/gliner_arabic-v2.1](https://huggingface.co/NAMAA-Space/gliner_arabic-v2.1)")

    st.markdown("### 📄 Note")
    st.markdown("""
        **If we fine-tune the model on domain-specific data, we will improve its accuracy for specific queries.**
    """)
