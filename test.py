import requests, os

key = os.getenv("BAISAN_API")

resp = requests.get(

    "https://api.edgefn.net/v1/models",

    headers={"Authorization": f"Bearer {key}"}

)

print(resp.json())