# 🔍 Smart Search - OLX Products

A bilingual (Arabic/English) intelligent search application for OLX products using Named Entity Recognition (NER) and natural language processing.

## Live Demo

Access the live demo of the application [here](https://smart-search1.streamlit.app/).

## Usecases

Every search system can benefit from this approach, where a user searches in natural language (English or Arabic) and gets relevant results applying the constraints/filters in the query.

### Examples

- **🍔 Food Delivery & Restaurant Discovery**
**User says**: "I'm looking for a spicy chicken pizza without olives"
**​System understands**:
Dish: Pizza
Protein: Chicken
Profile: Spicy
Exclude: Olives
- **​🛍️ E-commerce**:
**​User says**: "Show me 15-inch laptops with at least 16GB RAM for under 25,000 EGP"
**​System understands**:
Category: Laptop
Screen Size: 15-inch
RAM: >= 16GB
Price: < 25000 EGP
- **​🏠 Real Estate**:
​**User says**: "I need a 3-bedroom apartment for rent in Zamalek with a balcony"
**​System understands**:
Property Type: Apartment
Listing Type: Rent
Location: Zamalek
Bedrooms: 3
Feature: Balcony
- **​✈️ Travel & Flight Booking**:
**​User says**: "Find me a non-stop flight from Cairo to London next Friday for two adults"
**​System understands**:
Type: Flight
Origin: Cairo
Destination: London
Date: Next Friday
Stops: 0
Passengers: 2

## 🌟 Features

- **Bilingual Search**: Supports both Arabic and English queries
- **Smart Entity Extraction**: Automatically extracts brands, models, storage, RAM, colors, prices, and battery information
- **Advanced Filtering**: Filter products by multiple criteria simultaneously
- **Natural Language Processing**: Uses GLiNER Arabic NER model for intelligent query parsing
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience
- **Real Product Data**: Works with scraped OLX Egypt product data

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Smart-Search
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run main.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Dataset

The project uses OLX Egypt product data with the following structure:

- **Products**: 9,292 mobile phone listings
- **Features**: Title, price, location, brand, description, seller info, images
- **Sources**: Data scraped from OLX Egypt marketplace

### Data Files

- `data/olx_products_cleaned.csv` - Cleaned dataset used by the application
- `data/olx_products.csv` - Raw scraped data
- `data/olx_scrapper.py` - Web scraping utility for OLX data collection

## 🔧 Technical Architecture

### Core Components

1. **main.py** - Streamlit web application interface
2. **utils.py** - NER processing and search logic
3. **data/olx_scrapper.py** - Data collection utilities
4. **notebooks/eda.ipynb** - Exploratory data analysis

### NER Model

- **Model**: [NAMAA-Space/gliner_arabic-v2.1](https://huggingface.co/NAMAA-Space/gliner_arabic-v2.1)
- **Capabilities**: Extracts entities from Arabic and English text
- **Extracted Entities**: BRAND, MODEL, STORAGE, RAM, BATTERY, PRICE, COLOR, CONDITION

### Search Algorithm

1. **Query Parsing**: Extract entities using GLiNER NER model
2. **Entity Normalization**: Map Arabic brands to English equivalents
3. **Multi-criteria Filtering**: Apply filters based on extracted entities
4. **Regex Enhancement**: Additional pattern matching for prices and battery percentages
5. **Result Ranking**: Sort by price and relevance

## 📝 Usage Examples

### English Queries

```text
"iPhone 13 Pro 256GB under 70000"
"Samsung A54 less than 15k"
"iPhone 11 128GB white 80% battery"
```

### Arabic Queries

```text
"ايفون 14 برو بطاريه 80%"
"هواوي P50 اقل من 8 الف"
"شاومي ريدمي نوت12"
```

### Supported Search Parameters

- **Brand**: Apple, Samsung, Huawei, Xiaomi, OPPO, OnePlus, etc.
- **Model**: iPhone 13, Galaxy S23, Redmi Note, etc.
- **Storage**: 64GB, 128GB, 256GB, 512GB
- **RAM**: 4GB, 6GB, 8GB, 12GB
- **Price Limits**: "under 50k", "less than 20000", "اقل من"
- **Battery**: Minimum battery percentage
- **Colors**: White, Black, Blue, etc.
- **Condition**: New, Used, Sealed

## 🛠️ Project Structure

```text
Smart Search/
├── main.py                     # Streamlit web application
├── utils.py                    # NER processing and search utilities
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── olx_products_cleaned.csv # Cleaned product dataset
│   ├── olx_products.csv        # Raw scraped data
│   └── olx_scrapper.py         # Web scraping utility
└── notebooks/
    └── eda.ipynb              # Exploratory data analysis
```

## 🔍 How It Works

1. **Input Processing**: User enters search query in Arabic or English
2. **Entity Extraction**: GLiNER model identifies relevant entities (brand, model, price, etc.)
3. **Query Enhancement**: Regex patterns extract additional information like price limits and battery percentages
4. **Filtering**: Multiple filters applied to the dataset based on extracted entities
5. **Results Display**: Matched products displayed with images, prices, and details

## 📈 Performance & Accuracy

- **Search Speed**: Instant results through cached model loading
- **Entity Recognition**: High accuracy for mobile phone-specific entities
- **Multi-language Support**: Seamless Arabic-English processing
- **Scalability**: Handles 9K+ product dataset efficiently

## 🔮 Future Improvements

- **Model Fine-tuning**: Domain-specific training for improved accuracy
- **Expanded Categories**: Support for laptops, tablets, and other electronics
- **Advanced Filters**: Location-based search, seller ratings
- **Real-time Data**: Live OLX integration
- **Voice Search**: Arabic/English voice input support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Dependencies

```text
pandas>=1.5.0          # Data manipulation
streamlit>=1.28.0      # Web interface
regex>=2023.10.0       # Pattern matching
gliner>=0.2.0          # NER model
torch>=2.0.0           # Deep learning framework
transformers>=4.30.0   # Hugging Face transformers
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Resources

- [GLiNER Arabic Model](https://huggingface.co/NAMAA-Space/gliner_arabic-v2.1)
- [Streamlit Documentation](https://docs.streamlit.io)
- [OLX Egypt](https://www.dubizzle.com.eg)

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: For production use, consider fine-tuning the NER model on domain-specific data to improve accuracy for specialized queries.
