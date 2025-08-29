import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata

class SmartPhoneSearch:
    def __init__(self, data_path: str = 'olx_products_cleaned.csv'):
        """Initialize the smart search system"""
        print("Loading cleaned data...")
        self.df = pd.read_csv(data_path)
        
        # Initialize models
        print("Loading multilingual embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize TF-IDF for baseline comparison
        print("Initializing TF-IDF vectorizer...")
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # Keep Arabic and English
            lowercase=True
        )
        
        # Precompute embeddings and TF-IDF
        self._precompute_vectors()
        
        # Arabic numerals mapping
        self.arabic_numerals = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
        
        print("Smart search system ready!")

    def _precompute_vectors(self):
        """Precompute embeddings and TF-IDF vectors for all products"""
        print("Precomputing embeddings for all products...")
        
        # Get search texts (combining title, description, specs)
        search_texts = self.df['search_text'].fillna('').tolist()
        
        # Compute embeddings
        self.embeddings = self.embedder.encode(search_texts, show_progress_bar=True)
        
        # Compute TF-IDF
        print("Computing TF-IDF vectors...")
        self.tfidf_matrix = self.tfidf.fit_transform(search_texts)
        
        print(f"Precomputed vectors for {len(search_texts)} products")

    def normalize_query(self, query: str) -> str:
        """Normalize search query"""
        # Convert Arabic numerals to English
        query = query.translate(self.arabic_numerals)
        # Normalize Unicode
        query = unicodedata.normalize('NFKC', query)
        # Clean spaces
        query = re.sub(r'\s+', ' ', query.strip())
        
        return query.lower()

    def extract_filters_from_query(self, query: str) -> Dict:
        """Extract structured filters from natural language query"""
        filters = {}
        
        # Price filters
        price_patterns = [
            r'(?:under|below|اقل من|تحت)\s*(\d+)k?',
            r'(?:max|maximum|حد اقصى)\s*(\d+)k?',
            r'(?:above|over|اكثر من|فوق)\s*(\d+)k?'
        ]
        
        for i, pattern in enumerate(price_patterns):
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                price = int(match.group(1))
                if 'k' in match.group(0):
                    price *= 1000
                
                if i < 2:  # under/below/max
                    filters['max_price'] = price
                else:  # above/over
                    filters['min_price'] = price
        
        # Storage filters
        storage_pattern = r'(\d+)\s*(?:gb|جيجا|giga)'
        storage_matches = re.findall(storage_pattern, query, re.IGNORECASE)
        if storage_matches:
            # Take the largest storage mentioned
            filters['storage_gb'] = max([int(s) for s in storage_matches])
        
        # RAM filters
        ram_pattern = r'(?:ram|رامات?)\s*(\d+)\s*(?:gb|جيجا)?'
        ram_match = re.search(ram_pattern, query, re.IGNORECASE)
        if ram_match:
            filters['ram_gb'] = int(ram_match.group(1))
        
        # Battery filters
        battery_pattern = r'(?:battery|بطار[يةه])\s*(\d+)\s*%?'
        battery_match = re.search(battery_pattern, query, re.IGNORECASE)
        if battery_match:
            filters['min_battery'] = int(battery_match.group(1))
        
        # Brand filters
        brands = ['iphone', 'apple', 'samsung', 'xiaomi', 'redmi', 'oppo', 'realme', 'huawei']
        for brand in brands:
            if brand in query.lower():
                if brand in ['iphone', 'apple']:
                    filters['brand'] = 'Apple - iPhone'
                else:
                    filters['brand'] = brand.title()
        
        # Condition filters
        if any(word in query.lower() for word in ['new', 'sealed', 'جديد', 'كسر زيرو']):
            filters['condition'] = 'new'
        elif any(word in query.lower() for word in ['excellent', 'ممتاز', 'نضيف']):
            filters['condition'] = 'excellent'
        
        return filters

    def apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply structured filters to dataframe"""
        filtered_df = df.copy()
        
        # Price filters
        if 'max_price' in filters:
            filtered_df = filtered_df[
                (filtered_df['price_clean'].isna()) | 
                (filtered_df['price_clean'] <= filters['max_price'])
            ]
        
        if 'min_price' in filters:
            filtered_df = filtered_df[
                (filtered_df['price_clean'].isna()) | 
                (filtered_df['price_clean'] >= filters['min_price'])
            ]
        
        # Storage filter
        if 'storage_gb' in filters:
            filtered_df = filtered_df[
                (filtered_df['storage_gb'].isna()) | 
                (filtered_df['storage_gb'] >= filters['storage_gb'] * 0.8)  # Allow some tolerance
            ]
        
        # RAM filter
        if 'ram_gb' in filters:
            filtered_df = filtered_df[
                (filtered_df['ram_gb'].isna()) | 
                (filtered_df['ram_gb'] >= filters['ram_gb'])
            ]
        
        # Battery filter
        if 'min_battery' in filters:
            filtered_df = filtered_df[
                (filtered_df['battery_percent'].isna()) | 
                (filtered_df['battery_percent'] >= filters['min_battery'])
            ]
        
        # Brand filter
        if 'brand' in filters:
            filtered_df = filtered_df[
                filtered_df['brand'].str.contains(filters['brand'], case=False, na=False)
            ]
        
        # Condition filter
        if 'condition' in filters:
            filtered_df = filtered_df[
                filtered_df['condition_extracted'] == filters['condition']
            ]
        
        return filtered_df

    def semantic_search(self, query: str, top_k: int = 20) -> Tuple[pd.DataFrame, np.ndarray]:
        """Perform semantic search using embeddings"""
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        return results, similarities[top_indices]

    def keyword_search(self, query: str, top_k: int = 20) -> Tuple[pd.DataFrame, np.ndarray]:
        """Perform baseline keyword search using TF-IDF"""
        # Transform query
        query_tfidf = self.tfidf.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        return results, similarities[top_indices]

    def smart_search(self, query: str, top_k: int = 20, method: str = 'semantic') -> Dict:
        """Main search function combining semantic search with filters"""
        
        # Normalize query
        normalized_query = self.normalize_query(query)
        
        # Extract filters
        filters = self.extract_filters_from_query(normalized_query)
        
        # Perform search
        if method == 'semantic':
            results, scores = self.semantic_search(normalized_query, top_k=100)  # Get more initially
        else:
            results, scores = self.keyword_search(normalized_query, top_k=100)
        
        # Apply filters
        if filters:
            original_count = len(results)
            results = self.apply_filters(results, filters)
            print(f"Filters applied: {filters}")
            print(f"Results after filtering: {len(results)}/{original_count}")
        
        # Take top-k after filtering
        results = results.head(top_k)
        
        return {
            'results': results,
            'query': query,
            'normalized_query': normalized_query,
            'filters': filters,
            'method': method,
            'total_found': len(results)
        }

    def compare_search_methods(self, query: str, top_k: int = 10) -> Dict:
        """Compare semantic vs keyword search"""
        semantic_results = self.smart_search(query, top_k, method='semantic')
        keyword_results = self.smart_search(query, top_k, method='keyword')
        
        return {
            'query': query,
            'semantic': semantic_results,
            'keyword': keyword_results
        }

    def get_sample_queries(self) -> List[str]:
        """Get sample queries for testing"""
        return [
            "iPhone 13 under 15k with 128GB storage",
            "أيفون استعمال خفيف 256 جيجا",
            "Samsung A52 good battery life",
            "سامسونج نوت جديد بالضمان",
            "iPhone with good battery above 80%",
            "Xiaomi Redmi 8GB RAM",
            "OPPO phone under 10000 EGP",
            "هاتف 256 جيجا بطارية جيدة",
            "iPhone 14 Pro Max purple",
            "Samsung Galaxy S23 sealed new"
        ]

# Test the search system
if __name__ == "__main__":
    # Initialize search system
    search = SmartPhoneSearch()
    
    # Test with sample queries
    print("\n" + "="*50)
    print("TESTING SMART SEARCH SYSTEM")
    print("="*50)
    
    sample_queries = search.get_sample_queries()
    
    for i, query in enumerate(sample_queries[:3]):  # Test first 3 queries
        print(f"\n--- Query {i+1}: {query} ---")
        
        result = search.smart_search(query, top_k=5)
        
        print(f"Found {result['total_found']} results")
        print(f"Filters detected: {result['filters']}")
        
        print("\nTop 3 results:")
        for j, (_, row) in enumerate(result['results'].head(3).iterrows()):
            print(f"{j+1}. {row['title']}")
            print(f"   Price: {row['price']} | Storage: {row['storage_gb']}GB | Battery: {row['battery_percent']}%")
            print(f"   Similarity: {row['similarity_score']:.3f}")
        
        print("-" * 50)
