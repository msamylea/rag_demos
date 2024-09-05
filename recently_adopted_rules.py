import requests
from bs4 import BeautifulSoup

# Send a GET request to the website
url = 'https://www.maine.gov/dhhs/oms/about-us/policies-rules/recently-adopted-rules'
response = requests.get(url)

# Parse the HTML content
page = response.text
soup = BeautifulSoup(page, 'html.parser')

# Find all div elements with class 'summary'
summary_divs = soup.find_all('div', class_='summary')

# Open the file in write mode
with open('recently_adopted_rules/recently_adopted_rules.txt', 'w', encoding='utf-8') as file:
    # Extract and write the text of the preceding span and the text within each div.summary
    for div in summary_divs:
        span = div.find_previous('span', id=True)
        if span:
            title = span.get_text(strip=True)
            file.write(f"Title: {title}\n")
        file.write(div.get_text(separator="\n", strip=True) + "\n\n")