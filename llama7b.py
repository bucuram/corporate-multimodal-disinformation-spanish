from vllm import LLM, SamplingParams
import transformers
from transformers import AutoTokenizer
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import string
import json
from tqdm import tqdm
import argparse

from utils import compute_performance_metrics

data_folder = '<path-to-data-folder>'

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
parser.add_argument('--caption', type=bool, default=False)
parser.add_argument('--description', type=bool, default=False)
parser.add_argument('--examples', type=bool, default=False)
args = parser.parse_args()

print('description', args.description)
print('caption', args.caption)
print('examples', args.examples)

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

prompt_target = [
    "[INST] <<SYS>> You are a helpful assistant and have to respond with only one of the following options: 'Organization', 'Brand', 'Other'.<</SYS>> You are given a screenshot of false content. Given this information, your task is to classify who is affected by the false content. Please respond with one of the following target categories: 'Organization', 'Brand', 'Other'.", 
    "\n[INST] The text from the image is the following: ",
    "\nThe caption of the image is the following: ",
    "\nThe description of the image is the following: ",
    "[/INST]",
    "\nThe target category is: '\n"
]

example_1 = ["\n[INST] The text from the image is the following: Resultados Chueca 2023 MA PP MMMM PSOE",
    "\nThe caption of the image is the following: a screenshot of a pie chart with the results of results",  
    "\nThe description of the image is the following: The image features a political bar graph with red and blue colors, likely representing political parties. The bar graph consists of a large circle divided into segments, each segment corresponding to one of the political parties. The bar graph provides information about the election results, likely showing the preference or support for each political party.",       
    "[/INST]",
    "\nThe target category is: 'Organization'\n"]

example_2 = ["\n[INST] The text from the image is the following: KIA Fans Club 3 de mayo a las 10 07 Y Felicitaciones a Zulay Arteaga por ganar un auto Kia Todavía nos quedan 6 autos para cualquiera que comparta nuestra publicación y tome su número de caja de la suerte ya que solo seis cajas tienen las llaves de nuestro auto Asegúrese de que su nombre esté en nuestra lista de ganadores de autos Regístrate aquí y https cutt ly q52tzpF Y Porque después de que el registro del nombre sea exitoso el premio del automóvil se enviará directamente al ganador PY O 1669 7598 comentarios 3008 veces compartida",
    "\nThe caption of the image is the following: a screenshot of a facebook post with a picture of a car",  
    "\nThe description of the image is the following: In addition to the cars, there are several small ""number"" icons placed around the advertisement in various positions. Four of these numbers are prominently displayed near the right side of the cars, while others can be seen scattered throughout the advertisement. The car on the left side is partially visible but seems to be less prominent in the scene.",       
    "[/INST]",
    "\nThe target category is: 'Brand'\n"]

example_3 = ["\n[INST] The text from the image is the following: Dña Moo IEA Y Quieres un IPhone 13 porsólo 2 euros DA Py Ye NW leralebedeva5522 8 Te han etiquetado en la foto Enhorabuena has ganado un IPhone 13 Mé LATE IEEE y TY ér Oclick_ess Y DARARRLALAL rafa_tudela Amcor 92 Avalentin9perez Osara_delgado96 Ajandrohg Osoledad ochoa 948011 noeliaeheh Ajuak 7 rafarodri23 Hace 25 minutos OI EOS Eo Sa 0 L",
    "\nThe caption of the image is the following: a screenshot of a person holding a cell phone with emoticions",  
    "\nThe description of the image is the following: The image is a cell phone screen displaying a message conversation in Spanish. There are multiple conversations happening simultaneously. On the screen, there are several people, likely the conversation participants, and a few emojis and even a heart-shaped emoji. The messages include Spanish language phrases, making it evident that this is a text conversation in Spanish. The screen seems to be a part of a social media platform or a messaging app, such as WhatsApp or Telegram.",       
    "[/INST]",
    "\nThe target category is: 'Other'\n"]


prompt_source = [
    "[INST] <<SYS>> You are a helpful assistant and have to respond with only one of the following options: 'Corporate', 'Advertising', 'Other'.<</SYS>>  You are given a screenshot of false content. Given this information, your task is to clasify the type of false content. Please respond with one of the three categories of content: 'Corporate', 'Advertising', 'Other'.",
    "\n[INST] The text from the image is the following: ",
    "\nThe caption of the image is the following: ",
    "\nThe description of the image is the following: ",
    "[/INST]",
    "\nThe content is: '\n"
]

example_1_source = ["[INST] The text from the image is the following: Resultados Chueca 2023 MA PP MMMM PSOE",
    "\nThe caption of the image is the following: a screenshot of a pie chart with the results of results",  
    "\nThe description of the image is the following: The image features a political bar graph with red and blue colors, likely representing political parties. The bar graph consists of a large circle divided into segments, each segment corresponding to one of the political parties. The bar graph provides information about the election results, likely showing the preference or support for each political party.",       
    "[/INST]",
    "\nThe content is: 'Other'\n"]

example_2_source = ["[INST] The text from the image is the following: KIA Fans Club 3 de mayo a las 10 07 Y Felicitaciones a Zulay Arteaga por ganar un auto Kia Todavía nos quedan 6 autos para cualquiera que comparta nuestra publicación y tome su número de caja de la suerte ya que solo seis cajas tienen las llaves de nuestro auto Asegúrese de que su nombre esté en nuestra lista de ganadores de autos Regístrate aquí y https cutt ly q52tzpF Y Porque después de que el registro del nombre sea exitoso el premio del automóvil se enviará directamente al ganador PY O 1669 7598 comentarios 3008 veces compartida",
    "\nThe caption of the image is the following: a screenshot of a facebook post with a picture of a car",  
    "\nThe description of the image is the following: In addition to the cars, there are several small ""number"" icons placed around the advertisement in various positions. Four of these numbers are prominently displayed near the right side of the cars, while others can be seen scattered throughout the advertisement. The car on the left side is partially visible but seems to be less prominent in the scene.",       
    "[/INST]",
    "\nThe content is: 'Corporate'\n"]

example_3_source = ["[INST] The text from the image is the following: INQUILAX INMOBILIARIA INICIO SOBRE INQUILAX QUIERES ALQUILAR MORE v",
    "\nThe caption of the image is the following: a screenshot of a website page with a view of the ocean",  
    "\nThe description of the image is the following: There are several chairs visible throughout the scene, some near the pool and others closer to the bench or other amenities. A couch can be seen in the background, offering additional seating options for guests. The property features large windows and an inviting outdoor setting, providing a comfortable and picturesque environment for visitors to enjoy.",       
    "[/INST]",
    "\nThe content is: 'Advertising'\n"]


if args.examples == True:
    if args.description == False:
        example_1.pop(2)
        example_2.pop(2)
        example_3.pop(2)
        example_1_source.pop(2)
        example_2_source.pop(2)
        example_3_source.pop(2)

    if args.caption == False:
        example_1.pop(1)
        example_2.pop(1)
        example_3.pop(1)
        example_1_source.pop(1)
        example_2_source.pop(1)
        example_3_source.pop(1)

df = pd.read_csv(f'{data_folder}/annotations.csv')
df = df[df['SPLIT'] == 'test']

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_text = row["image_text"]
    caption = row["blip_caption"]
    description = row["description"]

    prompts_target = [
        prompt_target[0]
    ]

    prompts_source = [
        prompt_source[0]
    ]

    if args.examples == True:

        prompts_target.extend(example_1)
        prompts_target.extend(example_2)
        prompts_target.extend(example_3)
        prompts_source.extend(example_1_source)
        prompts_source.extend(example_2_source)
        prompts_source.extend(example_3_source)

    prompts_target.append(prompt_target[1] + img_text + "\n")
    prompts_source.append(prompt_source[1] + img_text + "\n")

    if args.caption == True:
        prompts_target.append(prompt_target[2] + caption + "\n")
        prompts_source.append(prompt_source[2] + caption + "\n")

    if args.description == True: 
        prompts_target.append(prompt_target[3] + description + "\n")
        prompts_source.append(prompt_source[3] + description + "\n")

    prompts_target += [
        prompt_target[4],
        prompt_target[5],
    ]

    prompts_source += [
        prompt_source[4],
        prompt_source[5],
    ]
    
    sequences = pipeline(
        ''.join(prompts_target),
        do_sample=True,
        temperature=0.7,
        top_p=1,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=15,
    )   

    response = sequences[0]['generated_text'].split("The target category is: '")[-1].strip()
    target_output = response.strip().lower().translate(str.maketrans('', '', string.punctuation))
    target_output = target_output.replace('\n',' ').strip().split(' ')[0]
    print('entire message 🤖--->', response)
    if target_output not in ['organization', 'brand', 'other']:
        target_output = 'other'
    df.loc[i, 'predicted_target'] = target_output
    print('target 🤖--->', target_output)
    print('target 👩‍💻--->', row['TARGET'])
    print('-------------------')
    
    sequences = pipeline(
        ''.join(prompts_source),
        do_sample=True,
        temperature=0.7,
        top_p=1,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=15,
    )
    response = sequences[0]['generated_text'].split("The content is: ")[-1].strip()
    source_output = response.strip().lower().translate(str.maketrans('', '', string.punctuation))
    source_output = source_output.replace('\n',' ').strip().split(' ')[0]
    if source_output not in ['corporate', 'advertising', 'other']:
        source_output = 'other'
    df.loc[i, 'predicted_source'] = source_output
    print('entire message 🤖--->', response)
    print('source 🤖--->', source_output)
    print('source 👩‍💻--->', row['SOURCE'])
    print('-------------------')

model_name = model_name.replace('/', '-') + f'-caption-{args.caption}-description-{args.description}--examples-{args.examples}'
df.to_csv(f'results/predictions-{model_name}.csv', index=False)

compute_performance_metrics(df, model_name)