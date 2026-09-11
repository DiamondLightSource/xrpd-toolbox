from pathlib import Path

import httpx

I15_1_PDFCURL_ENDPOINT = "https://i15-1-pdfget.diamond.ac.uk/data2pdf"


def send_post_request(url: str, data: dict) -> httpx.Response:
    """Send a JSON payload to a URL via HTTP POST."""
    response = httpx.post(url, json=data)
    response.raise_for_status()
    return response


def send_xy_to_pdfcurl(xy_filepath: str, composition: str, wavelength: float) -> dict:
    """Uses the default arguments for pdfcurl"""

    name = Path(xy_filepath).stem

    pdfcurl_args = {
        "composition": composition,
        "wavelength": wavelength,
        "name": name,
        "input_filepath": xy_filepath,
    }

    response = send_post_request(url=I15_1_PDFCURL_ENDPOINT, data=pdfcurl_args)

    return response.json()
