import pandas as pd

import base64
from PIL import Image
from io import BytesIO

import hashlib

import openai
import requests

# For caption generation
from transformers import BlipProcessor, BlipForConditionalGeneration

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Utility to clean user input by removing extra spaces
def clean_user_input(user_query):
    cleaned_query = ' '.join(user_query.split())
    return cleaned_query

# Convert base64 string to a PIL Image object
def base64_to_pil_image(base64_string):

    # Decode the base64 string
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object from the decoded data
    image_bytes = BytesIO(image_data)

    # Open the image using PIL
    image = Image.open(image_bytes)

    return image


# This hash function is to create a unique id for the prompt
def generate_hash(input_string, algorithm='sha256'):
    """
    Generate a hash for a given string using the specified algorithm.
    
    Parameters:
        input_string (str): The input string to hash.
        algorithm (str): The hashing algorithm to use. Options include 'md5', 'sha1', 'sha256', etc.
        
    Returns:
        str: The hexadecimal representation of the hash.
    """
    # Get the hashing function from hashlib based on the specified algorithm
    hash_function = getattr(hashlib, algorithm)
    
    # Create a hash object and update it with the encoded string
    hash_object = hash_function()
    hash_object.update(input_string.encode('utf-8'))
    
    # Return the hexadecimal digest of the hash
    return hash_object.hexdigest()

# Call Stability AI API and generate an image
def generate_stability_ai_image(STABILITY_AI_API_KEY, prompt):
    host = 'https://api.stability.ai/v2beta/stable-image/generate/sd3'
    params = {
        "prompt" : prompt,
        "negative_prompt" : '',
        "width": 512,
        "height": 512,
        "aspect_ratio" : '1:1',
        "seed" : 0,
        "output_format" : 'jpeg',
        "model" : "sd3",
        "mode" : "text-to-image"
    }
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_AI_API_KEY}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return base64.b64encode(response.content)
    # To test the function: response = generate_stability_ai_image("cute shiba inu")

# Call Dall-E Open AI API and generate an image
def call_dalle_api(OPENAI_API_KEY, prompt):
    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI()
    response = client.images.generate(
    model="dall-e-2",
    prompt=prompt,
    size="512x512",
    quality="standard",
    n=1,
    response_format="b64_json"
    )

    return response.data[0].b64_json
    # To test the function: response = call_dalle_api("A realistic image of a shiba inu with a birthday hat on the street")

# Produce output
def update_image_record(prompt, user_preference):
    record_id = generate_hash(prompt, algorithm='md5')        

    # Load dataframe
    df = pd.read_csv('image_generation_results.csv')

    # Get record for prompt
    record = df[df["Unique Id"] == record_id]

    # Update fields
    record["User Preference"] = user_preference
    df[df["Unique Id"] == record_id] = record

    # Save back to file
    df.to_csv('image_generation_results.csv', index=False)

# Function for caption generation
def generate_caption(processor, model, image):
    # Process the image and generate a caption
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to calculate Rouge score
def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

# This function calculates scores for the models and saves back to the results file
def process_image_record(record_id):
    # Load the pre-trained model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load prompt for images
    prompt_file = open(f'generated_images/{record_id}_prompt.txt')
    prompt = prompt_file.read().split(',')[1]

    # Generate caption for Stability AI image
    img = Image.open(f"generated_images/{record_id}_stability_ai.jpg")
    stability_ai_caption = generate_caption(processor, model, img)

    # Generate caption for Dall-E image
    img = Image.open(f"generated_images/{record_id}_dalle.jpg")
    dalle_caption = generate_caption(processor, model, img)

    # Generate Stability AI BLEU score
    # Prompted captions and generated captions
    reference_captions = [prompt]
    generated_captions = [stability_ai_caption]

    # Calculate BLEU scores for each pair of reference and generated captions
    bleu_scores = [sentence_bleu([ref.split()], gen.split()) for ref, gen in zip(reference_captions, generated_captions)]

    # Calculate the average BLEU score
    stability_ai_bleu = sum(bleu_scores) / len(bleu_scores)

    # Print the average BLEU score
    print(f"Average BLEU Score: {stability_ai_bleu:.4f}")

    # Generate Dall-E BLEU score
    # Prompted captions and generated captions
    reference_captions = [prompt]
    generated_captions = [dalle_caption]

    # Calculate BLEU scores for each pair of reference and generated captions
    bleu_scores = [sentence_bleu([ref.split()], gen.split()) for ref, gen in zip(reference_captions, generated_captions)]

    # Calculate the average BLEU score
    dalle_bleu = sum(bleu_scores) / len(bleu_scores)

    # Print the average BLEU score
    print(f"Average BLEU Score: {dalle_bleu:.4f}")

    # Generate Stability AI ROUGE Score
    reference_captions = [prompt]
    generated_captions = [stability_ai_caption]
    for ref, gen in zip(reference_captions, generated_captions):
        stability_ai_rouge_score = calculate_rouge(ref, gen)
        print(f"ROUGE scores for reference: '{ref}' and generated: '{gen}': {stability_ai_rouge_score}")

    # Generate Dall-E ROUGE Score
    reference_captions = [prompt]
    generated_captions = [dalle_caption]
    for ref, gen in zip(reference_captions, generated_captions):
        dalle_rouge_score = calculate_rouge(ref, gen)
        print(f"ROUGE scores for reference: '{ref}' and generated: '{gen}': {dalle_rouge_score}")


    # Load dataframe
    df = pd.read_csv('image_generation_results.csv')

    # Get record for prompt
    record = df[df["Unique Id"] == record_id]

    if len(record) == 1:
        # Update fields
        record["Stability AI Caption"] = stability_ai_caption
        record["Dall-E Caption"] = dalle_caption
        record["Stability AI BLEU"] = stability_ai_bleu
        record["Dall-E BLEU"] = dalle_bleu
        record["Stability AI Rouge"] = str(stability_ai_rouge_score)
        record["Dall-E Rouge"] = str(dalle_rouge_score)
        df[df["Unique Id"] == record_id] = record

        # Save back to file
        df.to_csv('image_generation_results.csv', index=False)        
    else:
        # Write record
        file_output = open('image_generation_results.csv', 'a')
        file_output.write(f'"{record_id}","{prompt}","{stability_ai_caption}","{dalle_caption}","{stability_ai_bleu}","{dalle_bleu}","{stability_ai_rouge_score}","{dalle_rouge_score}",""\r\n')
        file_output.close()

    # return higher scoring model
    if stability_ai_bleu >= dalle_bleu:
        return 'Stability AI'
    else:
        return 'Dall-E'