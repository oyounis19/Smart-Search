import pandas as pd
import re
import unicodedata
from langdetect import detect, DetectorFactory
import numpy as np
from typing import Dict, List, Tuple, Optional

# Set seed for consistent language detection
DetectorFactory.seed = 0

class PhoneDataCleaner:
    def __init__(self):
        # Arabic numerals to English mapping
        self.arabic_numerals = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
        
        # Storage patterns (GB, TB)
        self.storage_patterns = {
            'gb': r'(\d+)\s*(?:gb|جيجا|جيجابايت|giga)',
            'tb': r'(\d+)\s*(?:tb|تيرا|تيرابايت|tera)'
        }
        
        # RAM patterns
        self.ram_patterns = r'(?:ram|رامات?|ذاكرة)\s*[:/]?\s*(\d+)\s*(?:gb|جيجا)?'
        
        # Battery patterns
        self.battery_patterns = r'(?:بطار[يةه]|battery)\s*[:/]?\s*(\d+)\s*%?'
        
        # Price patterns
        self.price_patterns = r'(?:egp|جنيه?|ج\.م|£)\s*([0-9,]+)'
        
        # Condition keywords
        self.condition_keywords = {
            'new': ['جديد', 'new', 'sealed', 'كسر زيرو', 'zero', 'مغلف'],
            'excellent': ['ممتاز', 'excellent', 'فوق الممتاز', 'نضيف', 'نظيف'],
            'good': ['جيد', 'good', 'استعمال خفيف', 'light usage'],
            'fair': ['مقبول', 'fair', 'استعمال عادي'],
            'poor': ['ضعيف', 'poor', 'محتاج صيانة', 'needs repair']
        }
        
        # Phone models mapping
        self.phone_models = {
            'iphone': r'iphone?\s*(\d+(?:\s*pro)?(?:\s*max)?)',
            'samsung': r'(?:samsung\s*)?(?:galaxy\s*)?([as]\d+(?:\s*ultra)?(?:\s*plus)?)',
            'xiaomi': r'(?:xiaomi\s*)?(?:redmi\s*)?(?:note\s*)?(\d+(?:\s*pro)?)',
            'oppo': r'oppo\s*([a-z]?\d+)',
            'realme': r'realme\s*(\d+(?:\s*pro)?)',
            'huawei': r'huawei\s*([a-z]?\d+(?:\s*pro)?)'
        }

    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text and convert Arabic numerals to English"""
        if pd.isna(text):
            return ""
        
        # Convert to string and normalize
        text = str(text)
        # Remove diacritics and normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        # Convert Arabic numerals to English
        text = text.translate(self.arabic_numerals)
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Arabic or English"""
        try:
            if not text or len(text.strip()) < 3:
                return 'unknown'
            
            # Count Arabic vs Latin characters
            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
            latin_chars = len(re.findall(r'[a-zA-Z]', text))
            
            if arabic_chars > latin_chars:
                return 'arabic'
            elif latin_chars > arabic_chars:
                return 'english'
            else:
                return 'mixed'
        except:
            return 'unknown'

    def extract_storage(self, text: str) -> Optional[int]:
        """Extract storage capacity in GB"""
        text = text.lower()
        
        # Check for TB first
        tb_match = re.search(self.storage_patterns['tb'], text, re.IGNORECASE)
        if tb_match:
            return int(tb_match.group(1)) * 1024  # Convert TB to GB
        
        # Check for GB
        gb_match = re.search(self.storage_patterns['gb'], text, re.IGNORECASE)
        if gb_match:
            return int(gb_match.group(1))
        
        return None

    def extract_ram(self, text: str) -> Optional[int]:
        """Extract RAM capacity in GB"""
        text = text.lower()
        match = re.search(self.ram_patterns, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def extract_battery(self, text: str) -> Optional[int]:
        """Extract battery percentage"""
        text = text.lower()
        match = re.search(self.battery_patterns, text, re.IGNORECASE)
        if match:
            battery = int(match.group(1))
            # Validate battery percentage (0-100)
            if 0 <= battery <= 100:
                return battery
        return None

    def clean_price(self, price_str: str) -> Optional[float]:
        """Clean and convert price to numeric"""
        if pd.isna(price_str):
            return None
        
        # Remove EGP and normalize
        price_str = str(price_str).lower()
        price_str = re.sub(r'egp|جنيه?|ج\.م|£', '', price_str)
        
        # Extract numbers and commas
        numbers = re.findall(r'[\d,]+', price_str)
        if numbers:
            # Take the largest number (main price)
            price_str = max(numbers, key=len)
            try:
                return float(price_str.replace(',', ''))
            except:
                return None
        return None

    def extract_condition(self, text: str) -> str:
        """Extract device condition"""
        text = text.lower()
        
        for condition, keywords in self.condition_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return condition
        
        return 'unknown'

    def extract_phone_model(self, title: str, brand: str) -> Optional[str]:
        """Extract specific phone model"""
        text = title.lower()
        brand_lower = brand.lower()
        
        # Map brand to pattern
        if 'iphone' in brand_lower or 'apple' in brand_lower:
            pattern = self.phone_models['iphone']
        elif 'samsung' in brand_lower:
            pattern = self.phone_models['samsung']
        elif 'xiaomi' in brand_lower or 'redmi' in brand_lower:
            pattern = self.phone_models['xiaomi']
        elif 'oppo' in brand_lower:
            pattern = self.phone_models['oppo']
        elif 'realme' in brand_lower:
            pattern = self.phone_models['realme']
        elif 'huawei' in brand_lower:
            pattern = self.phone_models['huawei']
        else:
            return None
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return None

    def create_search_text(self, row: pd.Series) -> str:
        """Create a comprehensive search text combining all relevant fields"""
        parts = []
        
        # Add title and description
        if pd.notna(row['title']):
            parts.append(row['title'])
        
        if pd.notna(row['description_clean']):
            parts.append(row['description_clean'])
        
        # Add extracted features as text
        if pd.notna(row['storage_gb']):
            parts.append(f"{row['storage_gb']}GB")
            parts.append(f"{row['storage_gb']} جيجا")
        
        if pd.notna(row['ram_gb']):
            parts.append(f"{row['ram_gb']}GB RAM")
            parts.append(f"رامات {row['ram_gb']}")
        
        if pd.notna(row['battery_percent']):
            parts.append(f"battery {row['battery_percent']}%")
            parts.append(f"بطارية {row['battery_percent']}%")
        
        if pd.notna(row['phone_model']):
            parts.append(row['phone_model'])
        
        # Add condition
        parts.append(row['condition_extracted'])
        
        # Add brand
        parts.append(row['brand'])
        
        return ' '.join(parts).lower()

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main function to clean the entire dataset"""
        print("Starting data cleaning...")
        
        # Create a copy
        df_clean = df.copy()
        
        # 1. Normalize text fields
        print("1. Normalizing text...")
        df_clean['title_clean'] = df_clean['title'].apply(self.normalize_arabic_text)
        df_clean['description_clean'] = df_clean['description'].apply(lambda x: self.normalize_arabic_text(x).replace('Description', '').replace('&&&', ' '))
        
        # 2. Detect language
        print("2. Detecting languages...")
        df_clean['title_language'] = df_clean['title_clean'].apply(self.detect_language)
        df_clean['desc_language'] = df_clean['description_clean'].apply(self.detect_language)
        
        # 3. Clean price
        print("3. Cleaning prices...")
        df_clean['price_clean'] = df_clean['price'].apply(self.clean_price)
        
        # 4. Extract phone specifications
        print("4. Extracting phone specifications...")
        
        # Combine title and description for extraction
        df_clean['full_text'] = (df_clean['title_clean'] + ' ' + df_clean['description_clean']).fillna('')
        
        df_clean['storage_gb'] = df_clean['full_text'].apply(self.extract_storage)
        df_clean['ram_gb'] = df_clean['full_text'].apply(self.extract_ram)
        df_clean['battery_percent'] = df_clean['full_text'].apply(self.extract_battery)
        df_clean['condition_extracted'] = df_clean['full_text'].apply(self.extract_condition)
        df_clean['phone_model'] = df_clean.apply(lambda row: self.extract_phone_model(row['title_clean'], row['brand']), axis=1)
        
        # 5. Create search text
        print("5. Creating search text...")
        df_clean['search_text'] = df_clean.apply(self.create_search_text, axis=1)
        
        # 6. Clean up location
        df_clean['location_clean'] = df_clean['location'].str.replace('•', '').str.strip()
        
        print(f"Data cleaning completed! Dataset shape: {df_clean.shape}")
        
        return df_clean

if __name__ == "__main__":
    # Test the cleaner
    cleaner = PhoneDataCleaner()
    
    # Load data
    df = pd.read_csv('olx_products_2024-02-12.csv')
    print(f"Original dataset: {df.shape}")
    
    # Clean data
    df_clean = cleaner.clean_dataset(df)
    
    # Save cleaned data
    df_clean.to_csv('olx_products_cleaned.csv', index=False)
    print("Cleaned data saved to 'olx_products_cleaned.csv'")
    
    # Show some statistics
    print("\n=== CLEANING STATISTICS ===")
    print(f"Storage extracted: {df_clean['storage_gb'].notna().sum()}/{len(df_clean)} ({df_clean['storage_gb'].notna().mean()*100:.1f}%)")
    print(f"RAM extracted: {df_clean['ram_gb'].notna().sum()}/{len(df_clean)} ({df_clean['ram_gb'].notna().mean()*100:.1f}%)")
    print(f"Battery extracted: {df_clean['battery_percent'].notna().sum()}/{len(df_clean)} ({df_clean['battery_percent'].notna().mean()*100:.1f}%)")
    print(f"Phone model extracted: {df_clean['phone_model'].notna().sum()}/{len(df_clean)} ({df_clean['phone_model'].notna().mean()*100:.1f}%)")
    
    print("\n=== SAMPLE EXTRACTIONS ===")
    sample = df_clean[df_clean['storage_gb'].notna() & df_clean['battery_percent'].notna()].head(5)
    for _, row in sample.iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Storage: {row['storage_gb']}GB | RAM: {row['ram_gb']}GB | Battery: {row['battery_percent']}% | Model: {row['phone_model']} | Condition: {row['condition_extracted']}")
