from main import add_book

while True:
    isbn = input()
    print(f"https://books.girlwhoisaro.bot/{add_book(isbn).__dict__['_headers']['location']}")
