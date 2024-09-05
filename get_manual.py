import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import re

# Send a GET request to the website
url = 'https://www.maine.gov/sos/cec/rules/10/ch101.htm'
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')
links = soup.find_all('a')

# Ensure the benefits_manual folder exists
os.makedirs('benefits_manual', exist_ok=True)

# Function to sanitize filenames
def sanitize_filename(filename):
    # Remove invalid characters and strip leading/trailing whitespace
    return re.sub(r'[\\/*?:"<>|\r\n]', "", filename).strip()

# Download and save doc and docx files
for link in links:
    href = link.get('href')
    if href and (href.endswith('.doc') or href.endswith('.docx')):
        doc_url = urljoin(url, href)
        doc_response = requests.get(doc_url)
        sanitized_filename = sanitize_filename(link.text) + ('.doc' if href.endswith('.doc') else '.docx')
        file_path = os.path.join('benefits_manual', sanitized_filename)
        with open(file_path, 'wb') as f:
            f.write(doc_response.content)
            print(f"Downloaded {sanitized_filename} to benefits_manual folder")