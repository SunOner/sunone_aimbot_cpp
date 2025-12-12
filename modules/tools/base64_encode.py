import base64

with open("images/body.jpg", "rb") as image_file:
    base64_string = base64.b64encode(image_file.read()).decode('utf-8')

print(base64_string)