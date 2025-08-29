import streamlit as st
import pandas as pd
import numpy as np
from simple_search import SimpleSmartSearch
import re

# Page configuration
st.set_page_config(
    page_title="Smart Phone Search",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    .search-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .product-card {
        background: black;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .price {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }
    .similarity-score {
        background: #e3f2fd;
        color: #1976d2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    .filter-applied {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize search system
@st.cache_resource
def load_search_system():
    return SimpleSmartSearch()

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📱 Smart Phone Search</h1>
        <p>AI-powered search for OLX/Dubizzle phone listings</p>
        <p>Search in Arabic, English, or both! Try: "iPhone 13 under 15k 128GB" or "أيفون 256 جيجا"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load search system
    try:
        search = load_search_system()
    except Exception as e:
        st.error(f"Error loading search system: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🔍 Search Options")
    
    # Search settings
    max_results = st.sidebar.slider("Max Results", 5, 50, 15)
    show_stats = st.sidebar.checkbox("Show Statistics", True)
    show_filters = st.sidebar.checkbox("Show Applied Filters", True)
    
    # Sample queries
    st.sidebar.header("💡 Sample Queries")
    sample_queries = search.get_sample_queries()
    
    for i, query in enumerate(sample_queries[:6]):
        if st.sidebar.button(f"Try: {query[:30]}...", key=f"sample_{i}"):
            st.session_state.search_query = query
    
    # Main search area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search box
        search_query = st.text_input(
            "🔍 Search for phones",
            value=st.session_state.get('search_query', ''),
            placeholder="e.g., iPhone 13 under 15k with 128GB storage",
            help="Search in Arabic, English, or both!"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Perform search
    if search_button and search_query:
        with st.spinner("🔍 Searching..."):
            # Perform search
            result = search.smart_search(search_query, top_k=max_results)
            
            # Display results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📱 Results Found", result['total_found'])
            with col2:
                st.metric("📊 Initial Matches", result['original_count'])
            with col3:
                filter_count = len(result['filters'])
                st.metric("🔧 Filters Applied", filter_count)
            
            # Show applied filters
            if show_filters and result['filters']:
                st.markdown("### 🔧 Applied Filters")
                filter_html = ""
                for key, value in result['filters'].items():
                    filter_html += f'<span class="filter-applied">{key}: {value}</span> '
                st.markdown(filter_html, unsafe_allow_html=True)
            
            # Search statistics
            if show_stats:
                with st.expander("📈 Search Statistics", expanded=False):
                    stats = search.search_stats(search_query)
                    
                    if stats:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'price' in stats:
                                st.subheader("💰 Price Statistics")
                                st.write(f"Min: {stats['price']['min']:,.0f} EGP")
                                st.write(f"Max: {stats['price']['max']:,.0f} EGP")
                                st.write(f"Average: {stats['price']['mean']:,.0f} EGP")
                        
                        with col2:
                            if 'brands' in stats:
                                st.subheader("🏷️ Brand Distribution")
                                brand_df = pd.DataFrame(list(stats['brands'].items()), 
                                                      columns=['Brand', 'Count'])
                                st.dataframe(brand_df, hide_index=True)
                    else:
                        st.info("No statistics available for this search.")
            
            # Results
            if result['total_found'] > 0:
                st.markdown("### 📱 Search Results")
                
                results_df = result['results']
                
                for idx, (_, row) in enumerate(results_df.iterrows()):
                    # Product card
                    with st.container():
                        st.markdown(f"""
                        <div class="product-card">
                            <div style="display: flex; justify-content: between; align-items: start;">
                                <div style="flex: 1;">
                                    <h4 style="margin: 0 0 10px 0;">{row['title']}</h4>
                                    <div style="margin-bottom: 10px;">
                                        <span class="price">{row['price'] if pd.notna(row['price']) else 'Price not specified'}</span>
                                        <span style="margin-left: 15px;">📍 {row['location_clean']}</span>
                                    </div>
                                </div>
                                <div>
                                    <span class="similarity-score">Match: {row['similarity_score']:.3f}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Details in expandable section
                        with st.expander(f"Details for: {row['title'][:50]}..."):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**📱 Phone Details**")
                                st.write(f"Brand: {row['brand']}")
                                if pd.notna(row['phone_model']):
                                    st.write(f"Model: {row['phone_model']}")
                                if pd.notna(row['storage_gb']):
                                    st.write(f"Storage: {row['storage_gb']}GB")
                                if pd.notna(row['ram_gb']):
                                    st.write(f"RAM: {row['ram_gb']}GB")
                            
                            with col2:
                                st.write("**🔋 Condition & Battery**")
                                st.write(f"Condition: {row['condition_extracted']}")
                                if pd.notna(row['battery_percent']):
                                    st.write(f"Battery: {row['battery_percent']}%")
                                st.write(f"Negotiable: {'Yes' if row['Negotiable'] == 1 else 'No'}")
                            
                            with col3:
                                st.write("**📞 Seller Info**")
                                if pd.notna(row['seller_name']):
                                    st.write(f"Seller: {row['seller_name']}")
                                if pd.notna(row['seller_joined']):
                                    st.write(f"Joined: {row['seller_joined']}")
                                if pd.notna(row['product_url']):
                                    st.markdown(f"[View Original Ad]({row['product_url']})")
                            
                            # Description
                            if pd.notna(row['description_clean']):
                                st.write("**📝 Description**")
                                description = row['description_clean'][:300]
                                if len(row['description_clean']) > 300:
                                    description += "..."
                                st.write(description)
            else:
                st.warning("No results found. Try a different search query or check your filters.")
    
    elif not search_query and search_button:
        st.warning("Please enter a search query.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 30px;">
        <p>🚀 Smart Search Demo | Built with Streamlit & scikit-learn</p>
        <p>Search across 9,900+ phone listings from OLX Egypt</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
