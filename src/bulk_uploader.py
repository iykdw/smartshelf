import sys

import serial

from main import add_book

serial_port = sys.argv[1]

base_url = input("Base URL: ").strip()

conn = serial.Serial(serial_port, 9600, timeout=1)
conn.write(b"H")
while conn.in_waiting > 0:
    print(conn.readline().decode("utf-8").strip())

while True:
    while True:
        line = conn.readline().decode("utf-8").strip()
        print(line)
        if "Waiting for a URL..." in line:
            break
    isbn = input("ISBN: ")
    return_url = f"{base_url}{add_book(isbn).__dict__['_headers']['location']}"
    print("Sending URL to writer...")
    conn.write(return_url.encode())
    while True:
        if conn.in_waiting == 0:
            continue

        line = conn.readline().decode("utf-8").strip()
        print(line)
        if "WRITE DONE!" in line:
            break

    print("Written, continue.")
