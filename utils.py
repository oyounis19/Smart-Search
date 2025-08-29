import pandas as pd
import re
import streamlit as st
from gliner import GLiNER

# Cache the model loading
@st.cache_resource
def load_ner_model():
    """Load the GLiNER Arabic NER model"""
    return GLiNER.from_pretrained("NAMAA-Space/gliner_arabic-v2.1")

def normalize_arabic_numbers(text):
    """Convert Arabic numerals to Western numerals"""
    arabic_to_western = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    return text.translate(arabic_to_western)

def extract_price_limit(text):
    """Extract price limit from text using patterns"""
    text = normalize_arabic_numbers(text.lower())
    
    # Patterns for price limits
    patterns = [
        r'(?:under|less than|below|<|اقل من|تحت|أقل من)\s*(\d+)\s*(?:k|الف|ألف)?',
        r'(\d+)\s*(?:k|الف|ألف)?\s*(?:or less|maximum|max|حد أقصى)',
        r'budget\s*(\d+)\s*(?:k|الف|ألف)?',
        r'up to\s*(\d+)\s*(?:k|الف|ألف)?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            price = int(match.group(1))
            # If 'k' or equivalent, multiply by 1000
            if 'k' in match.group(0) or 'الف' in match.group(0) or 'ألف' in match.group(0):
                price *= 1000
            return price
    return None

def extract_battery_percentage(text):
    """Extract battery percentage from text"""
    text = normalize_arabic_numbers(text)
    
    patterns = [
        r'(?:battery|بطاريه|بطارية)\s*(\d+)%',
        r'(\d+)%\s*(?:battery|بطاريه|بطارية)',
        r'(\d+)\s*%'  # Simple percentage
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            percentage = int(match.group(1))
            if 0 <= percentage <= 100:
                return percentage
    return None

def normalize_entities(entities):
    """Normalize and deduplicate entities by taking the one with highest score for each type"""
    
    # Group entities by label
    entity_groups = {}
    for entity in entities:
        label = entity['label']
        if label not in entity_groups:
            entity_groups[label] = []
        entity_groups[label].append(entity)
    
    # For each label, keep only the entity with highest score
    normalized_entities = []
    for label, entity_list in entity_groups.items():
        if entity_list:
            # Sort by score (descending) and take the highest
            best_entity = max(entity_list, key=lambda x: x['score'])
            normalized_entities.append(best_entity)
    
    return normalized_entities

def parse_query_with_ner(query, model):
    """Parse query using GLiNER NER model and regex patterns"""
    
    # Define entity labels for extraction
    labels = [
        "BRAND", "MODEL", "STORAGE", "RAM", "BATTERY",
        "PRICE", "COLOR", "PERCENTAGE", "CONDITION",
    ]
    query = query.replace(',', '')
    # Extract entities using GLiNER
    raw_entities = model.predict_entities(query, labels, threshold=0.3)

    # Normalize and deduplicate entities
    entities = normalize_entities(raw_entities)
    print("Extracted Entities:", entities)
    # Initialize result
    parsed = {
        'raw_query': query,
        'brand': None,
        'model': None, 
        'storage': None,
        'ram': None,
        'color': None,
        'price_max': None,
        'battery_min': None,
        'condition': None,
        'entities_found': entities
    }
    
    # Process NER entities
    for entity in entities:
        entity_text = entity['text'].lower()
        entity_label = entity['label']
        
        if entity_label == 'BRAND':
            # Brand mapping
            brand_mapping = {
                'iphone': 'Apple', 'ايفون': 'Apple', 'apple': 'Apple',
                'samsung': 'Samsung', 'سامسونج': 'Samsung',
                'huawei': 'Huawei', 'هواوي': 'Huawei',
                'xiaomi': 'Xiaomi', 'شاومي': 'Xiaomi',
                'oppo': 'OPPO', 'اوبو': 'OPPO',
                'oneplus': 'OnePlus', 'ون بلس': 'OnePlus'
            }
            
            for key, brand in brand_mapping.items():
                if key in entity_text:
                    parsed['brand'] = brand
                    break
            
            # If no mapping found, use the extracted text
            if not parsed['brand']:
                parsed['brand'] = entity['text']
        
        elif entity_label == 'MODEL':
            parsed['model'] = entity['text']
        
        elif entity_label == 'STORAGE':
            # Add storage extraction (e.g., "128GB", "256", etc.)
            parsed['storage'] = entity['text']
        
        elif entity_label == 'RAM':
            # Add RAM extraction (e.g., "8GB RAM", "6GB", etc.)
            parsed['ram'] = entity['text']
        
        elif entity_label == 'BATTERY':
            # Extract battery info from NER and try to get percentage
            battery_text = entity['text']
            # Try to extract percentage from NER result
            battery_match = re.search(r'(\d+)%', battery_text)
            if battery_match:
                battery_percent = int(battery_match.group(1))
                if 0 <= battery_percent <= 100:
                    parsed['battery_min'] = battery_percent
            else:
                # Store the raw battery text for filtering
                parsed['battery_raw'] = battery_text
        
        elif entity_label == 'COLOR':
            parsed['color'] = entity['text']
        
        elif entity_label == 'CONDITION':
            parsed['condition'] = entity['text']
    
    # Extract price limit using regex
    parsed['price_max'] = extract_price_limit(query)
    
    # Extract battery percentage using regex (if not already found by NER)
    if not parsed['battery_min']:
        battery_percent = extract_battery_percentage(query)
        if battery_percent:
            parsed['battery_min'] = battery_percent
    
    return parsed

def search_products(df, parsed_query):
    """Filter products based on parsed query"""
    
    filtered_df = df.copy()
    applied_filters = []
    
    # Filter by brand
    if parsed_query['brand']:
        brand_filter = filtered_df['brand'].str.contains(
            parsed_query['brand'], case=False, na=False
        )
        filtered_df = filtered_df[brand_filter]
        applied_filters.append(f"Brand: {parsed_query['brand']}")
    
    # Filter by model/product (search in title)
    if parsed_query['model']:
        model_filter = filtered_df['title'].str.contains(
            parsed_query['model'], case=False, na=False
        )
        filtered_df = filtered_df[model_filter]
        applied_filters.append(f"Model: {parsed_query['model']}")
    
    # Filter by storage (search in title/description)
    if parsed_query['storage']:
        storage_filter = (
            filtered_df['title'].str.contains(parsed_query['storage'], case=False, na=False) |
            filtered_df['description'].str.contains(parsed_query['storage'], case=False, na=False)
        )
        filtered_df = filtered_df[storage_filter]
        applied_filters.append(f"Storage: {parsed_query['storage']}")
    
    # Filter by RAM (search in title/description)
    if parsed_query['ram']:
        ram_filter = (
            filtered_df['title'].str.contains(parsed_query['ram'], case=False, na=False) |
            filtered_df['description'].str.contains(parsed_query['ram'], case=False, na=False)
        )
        filtered_df = filtered_df[ram_filter]
        applied_filters.append(f"RAM: {parsed_query['ram']}")
    
    # Filter by color (search in title/description)
    if parsed_query['color']:
        color_filter = (
            filtered_df['title'].str.contains(parsed_query['color'], case=False, na=False) |
            filtered_df['description'].str.contains(parsed_query['color'], case=False, na=False)
        )
        filtered_df = filtered_df[color_filter]
        applied_filters.append(f"Color: {parsed_query['color']}")
    
    # Filter by price limit
    if parsed_query['price_max']:
        price_filter = filtered_df['price'] <= parsed_query['price_max']
        filtered_df = filtered_df[price_filter]
        applied_filters.append(f"Price ≤ {parsed_query['price_max']:,} EGP")
    
    # Filter by minimum battery percentage (search in description for batteries >= battery_min)
    if parsed_query['battery_min']:
        # Extract all battery percentages from descriptions and filter for >= battery_min
        def check_battery_condition(description):
            if pd.isna(description):
                return False
            # Find all battery percentages in the description
            battery_matches = re.findall(r'(\d+)%', str(description))
            if battery_matches:
                # Check if any battery percentage is >= the minimum required
                battery_percentages = [int(match) for match in battery_matches if match.isdigit()]
                return any(percentage >= parsed_query['battery_min'] for percentage in battery_percentages)
            return False
        
        battery_filter = filtered_df['description'].apply(check_battery_condition)
        filtered_df = filtered_df[battery_filter]
        applied_filters.append(f"Battery ≥ {parsed_query['battery_min']}%")
    
    # Handle raw battery text from NER (if no percentage found)
    elif 'battery_raw' in parsed_query and parsed_query['battery_raw']:
        battery_filter = filtered_df['description'].str.contains(
            parsed_query['battery_raw'], case=False, na=False
        )
        filtered_df = filtered_df[battery_filter]
        applied_filters.append(f"Battery: {parsed_query['battery_raw']}")
    
    # Filter by condition
    if parsed_query['condition']:
        condition_filter = (
            filtered_df['title'].str.contains(parsed_query['condition'], case=False, na=False) |
            filtered_df['description'].str.contains(parsed_query['condition'], case=False, na=False)
        )
        filtered_df = filtered_df[condition_filter]
        applied_filters.append(f"Condition: {parsed_query['condition']}")
    
    # Simple text search in title and description for remaining terms (only if no filters applied)
    if not applied_filters:
        query_lower = parsed_query['raw_query'].lower()
        if query_lower.split():  # Check if query has words
            first_word = query_lower.split()[0]
            text_filter = (
                filtered_df['title'].str.lower().str.contains(first_word, na=False) |
                filtered_df['description'].str.lower().str.contains(first_word, na=False)
            )
            filtered_df = filtered_df[text_filter]
            applied_filters.append(f"Text search: '{first_word}'")
    
    # Sort by price (descending)
    filtered_df = filtered_df.sort_values('price', ascending=False)

    return filtered_df, applied_filters

@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    try:
        df = pd.read_csv('data/olx_products_cleaned.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'data/olx_products_cleaned.csv' exists.")
        return pd.DataFrame()
