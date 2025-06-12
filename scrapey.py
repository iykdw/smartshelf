import json

import requests
from bs4 import BeautifulSoup
isbns = []
for i in range(1, 108):
    response = requests.get(f"https://www.penguin.co.uk/penguin-classics/classics-list?page={i}")

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a", class_="BookCard_link__xMxil")

    for link in links:
        url = link["href"]
        isbn = url.split("/")[-1]
        isbns.append(isbn)

with open("isbns.json", "w") as f:
    f.write(json.dumps(isbns))
