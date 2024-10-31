import base64
from PIL import Image
from io import BytesIO
import hashlib

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