import pandas as pd
import urllib.request 
from PIL import Image 
import pytesseract
import requests
import cv2
import re
import os
import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import BlipProcessor, BlipForConditionalGeneration

data_folder = '<path-to-data-folder>'

def compute_performance_metrics(df, model_name):

    ## compute performance metrics for target classification
    df['predicted_target'] = df['predicted_target'].str.lower()
    df['predicted_source'] = df['predicted_source'].str.lower()
    df['TARGET'] = df['TARGET'].str.lower()
    df['SOURCE'] = df['SOURCE'].str.lower()

    acc = accuracy_score(df['TARGET'], df['predicted_target'])
    f1 = f1_score(df['TARGET'], df['predicted_target'], average='macro')
    precision = precision_score(df['TARGET'], df['predicted_target'], average='macro')
    recall = recall_score(df['TARGET'], df['predicted_target'], average='macro')

    class_rep = classification_report(df['TARGET'], df['predicted_target'], digits=4)

    with open(f'results/results-{model_name}.txt', 'w') as f:
        f.write(f'Results for target classification\n\n')
        f.write(f'Accuracy: {acc}\n')
        f.write(f'F1: {f1}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n\n')
        f.write(f'{class_rep}\n\n')

    print('Results for target classification\n\n')
    print(f'Accuracy: {acc}\n')
    print(f'F1: {f1}\n')
    print(f'Precision: {precision}\n')
    print(f'Recall: {recall}\n\n')
    print(f'{class_rep}\n\n')

    ## compute performance metrics for source classification

    acc = accuracy_score(df['SOURCE'], df['predicted_source'])
    f1 = f1_score(df['SOURCE'], df['predicted_source'], average='macro')
    precision = precision_score(df['SOURCE'], df['predicted_source'], average='macro')
    recall = recall_score(df['SOURCE'], df['predicted_source'], average='macro')
    
    class_rep = classification_report(df['SOURCE'], df['predicted_source'], digits=4)

    with open(f'results/results-{model_name}.txt', 'a') as f:
        f.write(f'Results for source classification\n\n')
        f.write(f'Accuracy: {acc}\n')
        f.write(f'F1: {f1}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n\n')
        f.write(f'{class_rep}\n')

    
    print('Results for source classification\n\n')
    print(f'Accuracy: {acc}\n')
    print(f'F1: {f1}\n')
    print(f'Precision: {precision}\n')
    print(f'Recall: {recall}\n\n')
    print(f'{class_rep}\n')
    
        
def download_images():

    #downloading images using the IBERIFIER API
    user = ''
    password = ''

    session = requests.Session()
    auth = session.post('https://repositorio.iberifier.eu/api')
    session.auth = (user, password)

    df = pd.read_csv(f'{data_folder}/annotations.csv')
    df = df[df['STATUS'] == 'Falso']
    df['IMAGE'] = df['IMAGE'].apply(lambda x: x.replace('=IMAGE("', '').replace('", 1)', ''))

    for i, row in df.iterrows():

        try:
            #check if image was already downloaded
            if os.path.isfile(f'{data_folder}/images/{row["INDEX"]}.jpg'):
                continue

            print('Downloading image: ', row['IMAGE'])
            
            data = requests.get(row['IMAGE']).content
            with open(f'{data_folder}/images/{row["INDEX"]}.jpg', 'wb') as handler:
                handler.write(data)

        except:
            print(f'Error with {row["INDEX"]}, {row["IMAGE"]}')

def extract_text_from_images():

    df = pd.read_csv(f'{data_folder}/annotations.csv')
    df['IMAGE'] = df['IMAGE'].apply(lambda x: x.replace('=IMAGE("', '').replace('", 1)', ''))
    
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        ## use tesseract to extract text from images
        try:
            # print('Extracting text from image: ', row['IMAGE'])
            img = cv2.imread(f'{data_folder}/images/{row["INDEX"]}.jpg')
            text = pytesseract.image_to_string(img, lang='spa')
            # print(text)
            df.loc[i, 'image_text'] = re.sub(r'\W+', ' ', text)
        
        except:
            print(f'Error with {row["INDEX"]}, {row["IMAGE"]}')

    df.to_csv(f'{data_folder}/annotations.csv', index=False)
    

def add_captions():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

    df = pd.read_csv(f'{data_folder}/annotations.csv')

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):

        try:
            img_url = f'data/images/{row["INDEX"]}.jpg'
            raw_image = Image.open(img_url)

            text = "a screenshot of"
            inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

            out = model.generate(**inputs)

            df.loc[i, 'blip_caption'] = processor.decode(out[0], skip_special_tokens=True)
        except:
            print(f'Error with {row["INDEX"]}, {row["IMAGE"]}')

    df.to_csv(f'{data_folder}/annotations.csv', index=False)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=False, help='Name of the model to use')
    parser.add_argument('--label_type', type=str, help='Use short or long label names')
    args = parser.parse_args()

    model_name = args.model_name
    df = pd.read_csv(f'results/predictions-{model_name}.csv')
    compute_performance_metrics(df, model_name)