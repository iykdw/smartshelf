import pyperclip

from main import add_book

base_url = input("Base URL: ").strip()

while True:
    isbn = input()
    return_url = f"{base_url}{add_book(isbn).__dict__['_headers']['location']}"
    pyperclip.copy(return_url)
    print(f"\n\n\nCopied ...{return_url[-3:]}, continue\n\n\n")
