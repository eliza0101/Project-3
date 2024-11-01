# Project-3 

## Project: 
Text-to-Image Generation and Analysis

## Presentation: 
https://shorturl.at/cDq5y  

## Project Overview
This project implements a user-centric workflow for AI image generation, comparing outputs from Stable Diffusion and DALL-E. It aims to enhance the image generation process by incorporating user feedback and analyzing the performance of different AI models.

## Files
* project 3_main_merged.ipynb
* common_functions.py
* image_generation_results.csv
* project 3_works.ipynb (alternative file if "merged" doesn't run for you)

## Table of Contents
* Features
* Installation
* Usage
* Workflow
* Components
* Data Analysis
* Future Enhancements

## Features
* Dual image generation using Stability AI (Stable Diffusion) and Open AI (Dall-E) APIs
* User-friendly interface for prompt input and image comparison
* Automated caption generation for AI-produced images
* Calculation of BLEU and ROUGE scores for quality assessment
* User preference collection and storage
* Data analysis using K-means clustering and Random Forest classification

## Installation

* Repository:
https://github.com/eliza0101/Project-3.git 

* Packages
    - pip install datasets
    - pip install pandas datasets torch transformers diffusers openai
    - pip install accelerate
    - pip install opencv-python

* Set up API keys
  - Create a .env file in the root directory
  - Add your API keys:
    - STABILITY_AI_API_KEY=<your_stability_ai_key>
    - OPENAI_API_KEY=<your_openai_key>

## Dependencies 
```
import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import hashlib
import openai
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import os
from dotenv import load_dotenv
import pandas as pd
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import common_functions as functions
```

## Models Used
* Stable Diffusion 3
* DALL-E 2
* BlipForConditionalGeneration

## Usage
1. Run the Gradio interface:
2. Enter an image prompt in the provided text box.
3. Click "Generate Images" to create images using both Stability AI and DALL-E.
4. Select your preferred image by clicking the corresponding button.
5. View the analysis results in the image_generation_results.csv file.

## Workflow
*(Team member's main responsibilities)
1. User Input: Collect image prompts via Gradio interface *(Eliza)
2. Image Generation: Use Stability AI and Open AI APIs *(Eliza)
3. Caption Generation: Apply BLIP model to describe generated images *(Tunji)
4. Evaluation: Calculate BLEU and ROUGE scores *(Tunji)
5. User Feedback: Collect user preferences between generated images *(Eliza)
6. Data Storage: Save information in a structured CSV file *(Eliza)
7. Analysis: (Tunji)
* Perform K-means clustering to visualize AI performance patterns
* Train Random Forest classifier to predict user preferences

## Components
* common_functions.py: Contains utility functions for image generation, caption creation, and data processing

## Data Analysis
The project uses two main techniques for data analysis:
1. K-means Clustering: Visualizes patterns in AI performance across different prompts and models
2. Random Forest Classifier: Predicts user preferences based on historical data

## Future Enhancements
* Expand the dataset with more diverse prompts and preferences
* Implement a dynamic feedback loop for more effective prompt guidance
* Explore additional image quality assessment metrics
* Develop visualizations to clarify AI prompt interpretation
* Integrate the workflow with broader creative platforms

## Resources: 
- https://realpython.com/generate-images-with-dalle-openai-api/ (Save image data to file)
- https://stackoverflow.com/questions/6375942/how-do-you-base-64-encode-a-png-image-for-use-in-a-data-uri-in-a-css-file (base-64 encode PNG for data-uri in a CSS file)
- https://stackoverflow.com/questions/3715493/encoding-an-image-file-with-base64  (convert base64 string to PIL image)
- https://github.com/gradio-app/gradio/issues/2283 (Accept base 64 strings as value for 
gr.Image and gr.Gallery)
- https://stackoverflow.com/questions/2323128/convert-string-in-base64-to-image-and-save-on-filesystem (Decode base64 string)
- https://www.gradio.app/docs/gradio/image (Gradio)
- https://www.gradio.app/main/docs/gradio/image (Create buttons on Gradio)
