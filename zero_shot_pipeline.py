from transformers import pipeline
from PIL import Image
import pandas as pd
import tqdm
import argparse

from utils import compute_performance_metrics

data_folder = '<path-to-data-folder>'

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
parser.add_argument('--label_type', type=str, help='Use short or long label names')
args = parser.parse_args()

if args.label_type == 'short':
    texts_target = {
        "organization": "Organization",
        "brand": "Brand",
        "other": "Other"
    }

    texts_source = {
        "corporate information": "Corporate",
        "paid advertising": "Advertising",
        "other": "Other"
    }

elif args.label_type == 'long':
    texts_target = {
        "a screenshot of false content targeting an organization (a company or an institution)": "Organization",
        "a screenshot of false content targeting a brand": "Brand",
        "a screenshot of false content targeting something else": "Other"
    }

    texts_source = {
        "a screenshot of false corporate information": "Corporate",
        "a screenshot of false persuasive advertisement": "Advertising",
        "a screenshot of false content from an individual or an unknown source": "Other"
    }


detector = pipeline(model=args.model_name, task="zero-shot-image-classification", use_fast=True, device=1)

df = pd.read_csv(f'{data_folder}/annotations.csv')
df = df[df['SPLIT'] == 'test']

for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    img_url = f'{data_folder}/images/{row["INDEX"]}.jpg'
    raw_image = Image.open(img_url)

    result_target = detector(raw_image, candidate_labels=texts_target.keys())
    result_source = detector(raw_image, candidate_labels=texts_source.keys())

    print(result_target[0])
    print(result_target[0]['label'])
    print(texts_target[result_target[0]['label']])

    df.loc[i, 'predicted_target'] = texts_target[result_target[0]['label']]
    df.loc[i, 'predicted_source'] = texts_source[result_source[0]['label']]

model_name = args.model_name.replace('/', '-') + f'-{args.label_type}-label-names'
df.to_csv(f'results/predictions-{model_name}-zero-shot-{args.label_type}-label-names.csv', index=False)

compute_performance_metrics(df, model_name)