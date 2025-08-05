import torch
import torchvision
import torchvision.transforms as transforms
from mimetypes import guess_type
import matplotlib.pyplot as plt
import openai
import base64
from imagenet_classes import IMAGENET100_CLASSES
from collections import defaultdict
import os 
import pdb
import json
import fnmatch

# Give a path of an image and the class name
# IMAGE_PATH, IMAGE_CLASS = "/data/ai22mtech12002/projects/data/imagenet/train/n01537544/n01537544_148.JPEG", "Indigo bunting"

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # image_path = image_path[0][0]
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def get_concepts(image_path, img_cls):
    data_url = local_image_to_data_url(image_path)
    response = openai.chat.completions.create(
            model="gpt4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a multimodal AI trained to provide the concepts associated with the given images."
                },
                {
                    "role": "user",
                    "content": [
                                {
                                    "type": "text",
                                    "text": f"I have an image of a {img_cls}. Can you list the key concepts or attributes present in this image? Please provide a detailed list including physical features, behaviors, and any other relevant characteristics. For example, you might include items like '(whiskers, fur, 4 legs, mammal, small, cute, ...)'. Return comma-separated short 1-4 word concepts, while also being comprehensive. Do not return anything else. No numbering or bulleting the concepts."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                    "url": data_url
                                    }
                                } 
                    ] 
                }
            ]
        )
    return response.choices[0].message.content.split(', ')
    # pdb.set_trace()

openai.api_type = 'azure'
openai.api_version = '2023-05-15'
openai.api_key = 'cecd591f637f4178a3740697677ecd60'
openai.azure_endpoint = 'https://lab1055gpt4vision.openai.azure.com/'

class_img_con_dict = defaultdict(lambda: defaultdict(list))
class_ids = IMAGENET100_CLASSES.keys()
file_pattern = '*.JPEG'

for cls in class_ids:
    print(f"Processing class: {cls}")
    # Replace with your actual data path
    class_path = "<data_path>" + cls
    if os.path.exists(class_path):
        for filename in os.listdir(class_path):
            if fnmatch.fnmatch(filename, file_pattern):
                print(filename)
                file_path = os.path.join(class_path, filename)
                try:
                    gpt4_concepts = get_concepts(file_path, IMAGENET100_CLASSES[cls])
                    class_img_con_dict[cls][filename] = gpt4_concepts
                    class_img_con_dict[cls] = dict(class_img_con_dict[cls])
                except openai.BadRequestError as e:
                    pass
                except Error as e: # type: ignore
                    pass
                except Exception as e:
                    pass
                
                
class_img_con_dict = dict(class_img_con_dict)

# Replace with your desired output file path
with open('./data/imagenet/inet100_instanceconcept.json', 'a') as f:
    json.dump(class_img_con_dict, f)
