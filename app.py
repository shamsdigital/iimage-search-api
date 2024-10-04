# app.py
from flask import Flask, request, jsonify
from supabase import create_client, Client
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast

# Create Flask app
app = Flask(__name__)


# Access environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_embedding(image_url):
    """Downloads an image from the given URL and returns its embedding vector."""
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
    """Searches for images in the 'images' table that are similar to the input image."""
    data = request.json
    input_image_url = data.get('image_url')
    threshold = data.get('threshold', 0.8)  # default threshold

    # Generate embedding for the input image
    input_embedding = get_image_embedding(input_image_url)
    if input_embedding is None:
        return jsonify({"error": "Failed to generate embedding for the input image."}), 400

    # Fetch all images and their embeddings from Supabase
    try:
        response = supabase.table("images").select("image_url, embedding, user_id").execute()
        images_data = response.data if response.data else []
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return jsonify({"error": "Failed to fetch images from the database."}), 500

    if not images_data:
        return jsonify({"message": "No images found in the database."}), 404

    # Prepare embeddings for similarity computation
    stored_embeddings = []
    user_ids = []
    for image in images_data:
        embedding_str = image['embedding']
        try:
            embedding = ast.literal_eval(embedding_str)
            stored_embeddings.append(embedding)
            user_ids.append(image['user_id'])
        except Exception as e:
            print(f"Error parsing embedding for image {image['image_url']}: {e}")
            continue

    if not stored_embeddings:
        return jsonify({"message": "No valid embeddings found in the database."}), 404

    stored_embeddings = np.array(stored_embeddings)
    input_embedding_np = np.array(input_embedding).reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(input_embedding_np, stored_embeddings)[0]

    # Find indices where similarity exceeds the threshold
    matching_indices = np.where(similarities >= threshold)[0]

    if len(matching_indices) == 0:
        return jsonify({"message": "No matching images found."}), 404

    # Retrieve matching image URLs and corresponding user_ids
    matching_images = [{"image_url": images_data[i]['image_url'], "user_id": user_ids[i]} for i in matching_indices]

    return jsonify({"matching_images": matching_images}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
