from http.client import HTTPResponse
from urllib.error import HTTPError

import pytest

from src.main import InvalidISBNError, get_data_from_api, get_data_from_gb
from src.smartshelf_secrets import Secrets

secrets = Secrets()

url_not_found = "https://girlwhoisaro.bot/smartshelf"
good_url = "https://google.com"


def test_get_data_from_api():
    """Test against a known-good URL"""
    response = get_data_from_api(good_url)
    assert isinstance(response, HTTPResponse)


def test_get_data_from_api_url_not_found():
    """Test against a known-bad URL"""
    response = get_data_from_api(url_not_found)
    assert isinstance(response, HTTPError)
    assert response.code == 404


def test_get_data_from_gb():
    book = get_data_from_gb("9780241759011")
    print(book.title)


def test_get_data_from_gb_bad_isbn():
    with pytest.raises(InvalidISBNError):
        get_data_from_gb("978024175901")

