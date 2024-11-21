import pandas as pd
import tqdm
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re
import argparse
import torch
import string
from pprint import pprint

from llava.eval.run_llava import load_image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from utils import compute_performance_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--image_text', type=bool, default=False)
args_cmd = parser.parse_args()

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
    )


args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "conv_mode": None,
    "sep": ",",
    "temperature": 0.7,
    "top_p": 1.0,
    "num_beams": 1,
    "max_new_tokens": 5,
})()

prompt_target_template = ["Instruction: You are given a screenshot of false content.",
                 ".\nWho is affected by the false content in the image? Choose only one of the following target categories: 'Organization', 'Brand', 'Other'.",
                 "\nAnswer:"]

image_text_prompt = "The text from the image is the following: "

prompt_source_template = ["Instruction: You are given a screenshot of false content.",
                ".\nWhat kind of content is present in the image? Choose only one of the three categories of content: 'Corporate', 'Advertising', 'Other'.",
                "\nAnswer:"]
 
prompt_description = "Write a detailed description of the contents of the image."

data_folder = '<path-to-data-folder>'
df = pd.read_csv(f'{data_folder}/annotations.csv')
df = df[df['SPLIT'] == 'test']

disable_torch_init()

for i, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

    img_url = f'/media/ssd/Ana/Maldita/data/images/{row["INDEX"]}.jpg'
    prompt_target = prompt_target_template.copy()
    prompt_source = prompt_source_template.copy()

    if args_cmd.image_text == True:
        prompt_target.insert(1, image_text_prompt + row['image_text'].strip())
        prompt_source.insert(1, image_text_prompt + row['image_text'].strip())
    
    qs = ''.join(prompt_target)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(img_url)
    images_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    
    target_output = outputs.strip().lower().translate(str.maketrans('', '', string.punctuation))
    target_output = target_output.replace('\n',' ').strip().split(' ')[0]
    print('target 🤖--->', target_output)
    print('target 👩‍💻--->', row['TARGET'])
    print()

    if target_output not in ['organization', 'brand', 'other']:
        target_output = 'other'

    df.loc[i, 'predicted_target'] = target_output

    qs = ''.join(prompt_source)
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(img_url)
    images_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    source_output = outputs.strip().lower().translate(str.maketrans('', '', string.punctuation))
    source_output = source_output.replace('\n',' ').strip().split(' ')[0]
    print('source 🤖--->', source_output)
    print('source 👩‍💻--->', row['SOURCE'])
    print()

    if source_output not in ['corporate', 'advertising', 'other']:
        source_output = 'other'
    df.loc[i, 'predicted_source'] = source_output

model_name = model_path.replace('/', '-') + f'-image-text-{args_cmd.image_text}'

df.to_csv(f'../results/predictions-{model_name}.csv', index=False)

compute_performance_metrics(df, model_name)
