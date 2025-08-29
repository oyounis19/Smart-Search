import datetime
import pandas as pd
from bs4 import BeautifulSoup
import requests
import dateparser

class OLXScraper:
    def __init__(self, base_url = None, headers = None):
        self.headers = headers if headers else {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
        self.base_url = base_url if base_url else 'https://www.dubizzle.com.eg'
        self.results = []
        self.products = []
        self.df = pd.DataFrame()
        self.current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    def __get_product_details(self, details):
        brand = None
        ad_type = None
        in_warranty = None
        deliverable = None
        payment_type = None
        condition = None

        if details is None:
            return brand, ad_type, in_warranty, deliverable, payment_type, condition
            
        spans = details.findAll('span')

        for i in range(0, len(spans), 2):
            key = spans[i].text.strip()
            value = spans[i + 1].text.strip()

            if key == 'Brand':
                brand = value
            elif key == 'Ad Type':
                ad_type = value
            elif key == 'Is Deliverable':
                deliverable = 1 if value == 'Yes' else 0
            elif key == 'In Warranty':
                in_warranty = 1 if value == 'Yes' else 0
            elif key == 'Payment Option':
                payment_type = value
            elif key == 'Condition':
                condition = value
        return brand, ad_type, in_warranty, deliverable, payment_type, condition

    def __get_seller_details(self, details):
        seller_name = None
        seller_joined = None
        seller_profile = None

        if details is None:
            return seller_name, seller_joined, seller_profile
        
        spans = details.findAll('span')

        if spans:
            seller_name = spans[0].text.strip()

            if len(spans) > 1:
                seller_joined = spans[1].text.strip()

        profile_link = details.find('a')

        if profile_link:
            seller_profile = self.base_url + profile_link['href']

        return seller_name, seller_joined, seller_profile

    def get_mobile_data(self, pageNo, lang='en', start_page=1):
        for i in range(start_page, pageNo+1):
            r = requests.get(f'{self.base_url}/{lang}/mobile-phones-tablets-accessories-numbers/mobile-phones/?page='+str(i+1), headers=self.headers)
            content = r.content
            soup = BeautifulSoup(content, 'html.parser')
            for d in soup.findAll('li', attrs={'aria-label':'Listing'}):
                print('page:', i, '#ad:', len(self.products) + 1, 'ad_title:', d.find('h2').text.strip())
                if d.find('div', {'aria-label': 'Price'}) is not None:
                    price = d.find('div', {'aria-label': 'Price'}).find('span').text.strip() # Price of the product
                    nego = 1 if len(d.find('div', {'aria-label': 'Price'})) > 1 else 0 # 1 if the price is negotiable, 0 if not
                else:
                    price = None
                    nego = None

                title = d.find('h2').text.strip() # Title of the product
                location = d.find('span', {'aria-label': 'Location'}).text.strip() # Location of the ad
                img = d.find('img')['src'] # Cover image
                date_str = d.find('span', {'aria-label': 'Creation date'}).text.strip() # Relative date
                date = dateparser.parse(date_str) # Convert the relative date to an actual date
                
                product_url = self.base_url + d.find('a')['href'] # URL of the product

                # Get request the product page
                p_r = requests.get(product_url, headers=self.headers)
                p_content = p_r.content
                p_soup = BeautifulSoup(p_content, 'html.parser')

                ad_id_div = p_soup.find('div', {'class': '_171225da'})
                ad_id = ad_id_div.text.replace('Ad id ', '').strip() if ad_id_div else None # Advertisement ID

                img_links = [img.find('img')['src'] for img in p_soup.find_all('div', {'class': 'image-gallery-slide'})] # All images of the product

                p_details = p_soup.find('div', {'aria-label': 'Details'}) # Details of the product
                if p_details is not None:
                    brand, ad_type, in_warranty, deliverable, payment_type, condition = self.__get_product_details(p_details) # Get the details of the product

                description = p_soup.find('div', {'aria-label': 'Description'})
                if description is not None:
                    description = description.text.strip()
                    description = description.replace('\n', '&&&') # Description of the product

                s_details = p_soup.find('div', {'aria-label': 'Seller description'}) # Details of the seller
                seller_name, seller_joined, seller_profile = self.__get_seller_details(s_details) # Get the details of the seller

                self.products.append([ad_id, title, price, nego, location, date, product_url, img, img_links, brand, description, ad_type, in_warranty, deliverable, payment_type, condition, seller_name, seller_joined, seller_profile])

        self.df = pd.DataFrame(self.products,columns=['ad_id', 'title','price','Negotiable', 'location', 'date', 'product_url', 'main_image', 'image_links', 'brand', 'description', 'ad_type', 'in_warranty', 'deliverable', 'payment_type', 'condition', 'seller_name', 'seller_joined', 'seller_profile'])
        self.df.to_csv(f'olx_products_{self.current_date}.csv', index=False, encoding='utf-8')

        return self.df

    def get_cars_data(self, pageNo, lang='en', start_page=1):
        for i in range(start_page, pageNo+1):
            r = requests.get(f'{self.base_url}/{lang}/vehicles/cars-for-sale/?page='+str(i), headers=self.headers)
            content = r.content
            print(r.url)
            soup = BeautifulSoup(content, 'html.parser')
            for d in soup.findAll('li', attrs={'aria-label':'Listing'}):
                print('page:', i, '#ad:', len(self.products) + 1, 'ad_title:', d.find('h2').text.strip())
                
                if d.find('div', {'aria-label': 'Price'}) is not None:
                    price = d.find('div', {'aria-label': 'Price'}).find('span').text.strip() # Price of the product
                    nego = 1 if len(d.find('div', {'aria-label': 'Price'})) > 1 else 0 # 1 if the price is negotiable, 0 if not
                else:
                    price = None
                    nego = None

                title = d.find('h2').text.strip() # Title of the product
                location = d.find('span', {'aria-label': 'Location'}).text.strip() # Location of the ad
                img = d.find('img')['src'] # Cover image
                date_str = d.find('span', {'aria-label': 'Creation date'}).text.strip() # Relative date
                date = dateparser.parse(date_str) # Convert the relative date to an actual date
                
                product_url = self.base_url + d.find('a')['href'] # URL of the product

                # Get request the product page
                p_r = requests.get(product_url, headers=self.headers)
                p_content = p_r.content
                p_soup = BeautifulSoup(p_content, 'html.parser')

                ad_id_div = p_soup.find('div', {'class': '_171225da'})
                ad_id = ad_id_div.text.replace('Ad id ', '').strip() if ad_id_div else None # Advertisement ID

                img_links = [img.find('img')['src'] for img in p_soup.find_all('div', {'class': 'image-gallery-slide'})] # All images of the product

                p_details = p_soup.find('div', {'aria-label': 'Details'}) # Details of the product
                if p_details is not None:
                    brand, ad_type, in_warranty, deliverable, payment_type, condition = self.__get_product_details(p_details) # Get the details of the product
                else:
                    brand, ad_type, in_warranty, deliverable, payment_type, condition = None, None, None, None, None, None

                description = p_soup.find('div', {'aria-label': 'Description'})
                if description is not None:
                    description = description.text.strip()
                    description = description.replace('\n', '&&&') # Description of the product

                s_details = p_soup.find('div', {'aria-label': 'Seller description'}) # Details of the seller
                seller_name, seller_joined, seller_profile = self.__get_seller_details(s_details) # Get the details of the seller

                self.products.append([ad_id, title, price, nego, location, date, product_url, img, img_links, brand, description, ad_type, in_warranty, deliverable, payment_type, condition, seller_name, seller_joined, seller_profile])

        self.df = pd.DataFrame(self.products,columns=['ad_id', 'title','price','Negotiable', 'location', 'date', 'product_url', 'main_image', 'image_links', 'brand', 'description', 'ad_type', 'in_warranty', 'deliverable', 'payment_type', 'condition', 'seller_name', 'seller_joined', 'seller_profile'])
        self.df.to_csv(f'olx_cars_{self.current_date}.csv', index=False, encoding='utf-8')

        return self.df


scraper = OLXScraper()
scraper.get_cars_data(2, 'en', start_page=1)