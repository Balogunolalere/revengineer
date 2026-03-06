import re

with open("instagram/client.py", "r") as f:
    text = f.read()

# Change User-Agent
text = re.sub(
    r'"Mozilla/5\.0.*?Safari/537\.36"\s*\)',
    '"Instagram 269.0.0.18.75 Android (26/8.0.0; 480dpi; 1080x1920; samsung; SM-G930F; hero2lte; samsungexynos8890; en_US; 314665256)")',
    text, flags=re.DOTALL
)

# Change X-IG-App-ID
text = re.sub(r'"X-IG-App-ID": "936619743392459"', '"X-IG-App-ID": "567067343352427"', text)

# We need to add Android UUIDs and Authorization header.
# Let's use a dynamic injection in start or request.
