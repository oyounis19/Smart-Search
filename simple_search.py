import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata

class SimpleSmartSearch:
    def __init__(self, data_path: str = 'olx_products_cleaned.csv'):
        """Initialize the search system with TF-IDF (no deep learning for now)"""
        print("Loading cleaned data...")
        self.df = pd.read_csv(data_path)
        
        # Initialize TF-IDF vectorizer with Arabic/English support
        print("Initializing TF-IDF vectorizer...")
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # Include trigrams for better phrase matching
            stop_words=None,     # Keep all words for Arabic/English mix
            lowercase=True,
            min_df=2,           # Ignore very rare terms
            max_df=0.95         # Ignore very common terms
        )
        
        # Precompute TF-IDF vectors
        self._precompute_vectors()
        
        # Arabic numerals mapping
        self.arabic_numerals = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
        
        print("Search system ready!")

    def _precompute_vectors(self):
        """Precompute TF-IDF vectors for all products"""
        print("Computing TF-IDF vectors for all products...")
        
        # Get search texts (combining title, description, specs)
        search_texts = self.df['search_text'].fillna('').tolist()
        
        # Compute TF-IDF matrix
        self.tfidf_matrix = self.tfidf.fit_transform(search_texts)
        
        print(f"Computed TF-IDF vectors for {len(search_texts)} products")

    def normalize_query(self, query: str) -> str:
        """Normalize search query"""
        # Convert Arabic numerals to English
        query = query.translate(self.arabic_numerals)
        # Normalize Unicode
        query = unicodedata.normalize('NFKC', query)
        # Clean up extra spaces
        query = re.sub(r'\s+', ' ', query.strip())
        
        return query.lower()

    def extract_filters_from_query(self, query: str) -> Dict:
        """Extract structured filters from natural language query"""
        filters = {}
        
        # Price filters
        price_patterns = [
            (r'(?:under|below|اقل من|تحت)\s*(\d+)k?', 'max_price'),
            (r'(?:max|maximum|حد اقصى)\s*(\d+)k?', 'max_price'),
            (r'(?:above|over|اكثر من|فوق)\s*(\d+)k?', 'min_price')
        ]
        
        for pattern, filter_type in price_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                price = int(match.group(1))
                if 'k' in match.group(0).lower():
                    price *= 1000
                filters[filter_type] = price
        
        # Storage filters
        storage_pattern = r'(\d+)\s*(?:gb|جيجا|giga)'
        storage_matches = re.findall(storage_pattern, query, re.IGNORECASE)
        if storage_matches:
            # Take the largest storage mentioned (likely the target)
            filters['storage_gb'] = max([int(s) for s in storage_matches])
        
        # RAM filters
        ram_pattern = r'(?:ram|رامات?)\s*(\d+)\s*(?:gb|جيجا)?'
        ram_match = re.search(ram_pattern, query, re.IGNORECASE)
        if ram_match:
            filters['ram_gb'] = int(ram_match.group(1))
        
        # Battery filters
        battery_pattern = r'(?:battery|بطار[يةه])\s*(?:above|over|اكثر من)?\s*(\d+)\s*%?'
        battery_match = re.search(battery_pattern, query, re.IGNORECASE)
        if battery_match:
            filters['min_battery'] = int(battery_match.group(1))
        
        # Brand filters
        brand_mapping = {
            'iphone': 'Apple - iPhone',
            'apple': 'Apple - iPhone',
            'samsung': 'Samsung',
            'xiaomi': 'Xiaomi',
            'redmi': 'Xiaomi',
            'oppo': 'OPPO',
            'realme': 'Realme',
            'huawei': 'Huawei'
        }
        
        for brand_keyword, brand_name in brand_mapping.items():
            if brand_keyword in query.lower():
                filters['brand'] = brand_name
                break
        
        # Condition filters
        condition_mapping = [
            (['new', 'sealed', 'جديد', 'كسر زيرو'], 'new'),
            (['excellent', 'ممتاز', 'نضيف', 'نظيف'], 'excellent'),
            (['good', 'جيد', 'خفيف'], 'good')
        ]
        
        for keywords, condition in condition_mapping:
            if any(keyword in query.lower() for keyword in keywords):
                filters['condition'] = condition
                break
        
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
        
        # Storage filter (with tolerance)
        if 'storage_gb' in filters:
            target_storage = filters['storage_gb']
            filtered_df = filtered_df[
                (filtered_df['storage_gb'].isna()) | 
                (filtered_df['storage_gb'] >= target_storage * 0.8)  # 80% tolerance
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

    def keyword_search(self, query: str, top_k: int = 20) -> Tuple[pd.DataFrame, np.ndarray]:
        """Perform keyword search using TF-IDF"""
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

    def smart_search(self, query: str, top_k: int = 20) -> Dict:
        """Main search function combining TF-IDF search with filters"""
        
        # Normalize query
        normalized_query = self.normalize_query(query)
        
        # Extract filters
        filters = self.extract_filters_from_query(normalized_query)
        
        # Perform TF-IDF search
        results, scores = self.keyword_search(normalized_query, top_k=100)  # Get more initially
        
        # Apply filters
        original_count = len(results)
        if filters:
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
            'total_found': len(results),
            'original_count': original_count
        }

    def search_stats(self, query: str) -> Dict:
        """Get search statistics and insights"""
        result = self.smart_search(query, top_k=50)
        
        results_df = result['results']
        stats = {}
        
        if len(results_df) > 0:
            # Price statistics
            valid_prices = results_df['price_clean'].dropna()
            if len(valid_prices) > 0:
                stats['price'] = {
                    'min': valid_prices.min(),
                    'max': valid_prices.max(),
                    'mean': valid_prices.mean(),
                    'median': valid_prices.median()
                }
            
            # Brand distribution
            stats['brands'] = results_df['brand'].value_counts().head(5).to_dict()
            
            # Storage distribution
            storage_dist = results_df['storage_gb'].dropna()
            if len(storage_dist) > 0:
                stats['storage'] = storage_dist.value_counts().sort_index().to_dict()
            
            # Battery statistics
            battery_dist = results_df['battery_percent'].dropna()
            if len(battery_dist) > 0:
                stats['battery'] = {
                    'min': battery_dist.min(),
                    'max': battery_dist.max(),
                    'mean': battery_dist.mean()
                }
        
        return stats

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
    search = SimpleSmartSearch()
    
    # Test with sample queries
    print("\n" + "="*60)
    print("TESTING SMART SEARCH SYSTEM")
    print("="*60)
    
    sample_queries = search.get_sample_queries()
    
    for i, query in enumerate(sample_queries[:5]):  # Test first 5 queries
        print(f"\n{'='*20} Query {i+1} {'='*20}")
        print(f"Query: {query}")
        print("-" * 50)
        
        result = search.smart_search(query, top_k=5)
        
        print(f"Found {result['total_found']} results (from {result['original_count']} initial matches)")
        print(f"Filters detected: {result['filters']}")
        
        print(f"\nTop {len(result['results'])} results:")
        for j, (_, row) in enumerate(result['results'].iterrows()):
            print(f"\n{j+1}. {row['title']}")
            print(f"   💰 Price: {row['price']} | 💾 Storage: {row['storage_gb']}GB | 🔋 Battery: {row['battery_percent']}%")
            print(f"   📍 Location: {row['location_clean']} | 🏷️ Brand: {row['brand']}")
            print(f"   📊 Similarity: {row['similarity_score']:.3f}")
        
        # Get search statistics
        stats = search.search_stats(query)
        if stats:
            print(f"\n📈 Search Statistics:")
            if 'price' in stats:
                print(f"   Price range: {stats['price']['min']:,.0f} - {stats['price']['max']:,.0f} EGP")
            if 'brands' in stats:
                print(f"   Top brands: {', '.join(stats['brands'].keys())}")
        
        print("\n" + "="*60)
