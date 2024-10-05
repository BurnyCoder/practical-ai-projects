import requests
import os

url = 'https://clipdrop-api.co/text-to-image/v1'
headers = {
    'x-api-key': os.environ.get("CLIPDROP_API_KEY")
}
files = {
    'prompt': (None, 'shot of vaporwave fashion dog in miami', 'text/plain')
}

response = requests.post(url, headers=headers, files=files)

if response.ok:
    with open('result.png', 'wb') as f:
        f.write(response.content)
    print("Image saved as result.png")
else:
    print("Error:", response.status_code, response.text)
