from flask import Flask, request, jsonify
from supabase import create_client, Client
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)

# Load Supabase credentials
from config import SUPABASE_URL, SUPABASE_API_KEY

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading or processing image: {e}")
        return None

    # Process the image and get embeddings
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Convert to CPU and detach from the computation graph
    embedding = outputs[0].cpu().numpy().tolist()
    return embedding

@app.route('/search-image', methods=['POST'])
def search_image():
    data = request.get_json()
    input_image_url = data.get('image_url')
    
    # Convert threshold to float, with a default value of 0.8
    threshold = float(data.get('threshold', 0.8))

    input_embedding = get_image_embedding(input_image_url)
    if input_embedding is None:
        return jsonify({"error": "Failed to generate embedding for the input image."}), 400

    # Fetch all images and their embeddings from Supabase
    response = supabase.table("images").select("image_url, embedding, user_id").execute()

    if not response.data:
        return jsonify({"error": "No images found in the database."}), 404

    images_data = response.data
    stored_embeddings = []
    user_ids = []

    for image in images_data:
        embedding_str = image['embedding']
        try:
            embedding = ast.literal_eval(embedding_str)
            stored_embeddings.append(embedding)
            user_ids.append(image['user_id'])
        except Exception as e:
            continue  # Skip this image if embedding parsing fails

    if not stored_embeddings:
        return jsonify({"error": "No valid embeddings found in the database."}), 404

    stored_embeddings = np.array(stored_embeddings)
    input_embedding_np = np.array(input_embedding).reshape(1, -1)

    # Calculate similarities
    similarities = cosine_similarity(input_embedding_np, stored_embeddings)[0]
    
    # Ensure we are comparing with a float threshold
    matching_indices = np.where(similarities >= threshold)[0]

    if len(matching_indices) == 0:
        return jsonify({"message": "Image not found."}), 404

    matching_images = [(images_data[i]['image_url'], user_ids[i]) for i in matching_indices]

    return jsonify({"matching_images": matching_images}), 200
