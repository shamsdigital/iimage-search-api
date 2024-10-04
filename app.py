# app.py

from flask import Flask, request, jsonify
from supabase import create_client, Client
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Initialize Flask app
app = Flask(__name__)

# Initialize Supabase client using environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Move model to appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_image_embedding(image_url):
    """
    Downloads an image from the given URL and returns its embedding vector.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading or processing image: {e}")
        return None

    # Process the image and get embeddings
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to device

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Normalize the embedding
    embedding = outputs[0].cpu().numpy().tolist()
    return embedding

def search_image(input_image_url, threshold=0.8, top_n=5):
    """
    Searches for images in the 'images' table that are similar to the input image.

    Parameters:
        input_image_url (str): The URL of the image to search.
        threshold (float): The cosine similarity threshold to consider a match.
        top_n (int): Number of top similar images to return.

    Returns:
        list: A list of tuples containing matching image URLs and their corresponding user_ids.
    """
    # Generate embedding for the input image
    input_embedding = get_image_embedding(input_image_url)
    if input_embedding is None:
        print("Failed to generate embedding for the input image.")
        return []

    # Fetch all images and their embeddings from Supabase
    try:
        response = supabase.table("images").select("image_url, embedding, user_id").execute()
        images_data = response.data
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return []

    if not images_data:
        print("No images found in the database.")
        return []

    similarities = []

    for image in images_data:
        embedding_str = image.get('embedding')
        user_id = image.get('user_id')
        image_url = image.get('image_url')

        if embedding_str is None:
            print(f"Skipping image {image_url} due to None embedding.")
            continue

        try:
            # Safely evaluate the embedding string to a list
            embedding = ast.literal_eval(embedding_str)
            embedding = np.array(embedding, dtype=float)

            # Ensure the embedding has the correct dimension
            if embedding.shape[0] != 1024 or len(input_embedding) != 1024:
                print(f"Unexpected embedding shape for image {image_url}: {embedding.shape}")
                continue

            # Calculate cosine similarity
            similarity = cosine_similarity([input_embedding], [embedding])[0][0]
            if similarity >= threshold:
                similarities.append({
                    "image_url": image_url,
                    "user_id": user_id,
                    "similarity_score": similarity
                })

        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            continue

    # Sort the results by similarity score in descending order
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)

    # Return top_n results
    return similarities[:top_n]

@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    Endpoint to search for similar images.
    Expects a JSON body with 'image_url' and optional 'threshold' and 'top_n'.
    """
    data = request.get_json()

    # Validate input
    if not data or 'image_url' not in data:
        return jsonify({"error": "Missing 'image_url' in request body."}), 400

    image_url = data['image_url']
    threshold = data.get('threshold', 0.8)
    top_n = data.get('top_n', 5)

    # Perform the search
    matching_images = search_image(image_url, threshold, top_n)

    if matching_images:
        return jsonify({"matching_images": matching_images}), 200
    else:
        return jsonify({"message": "No similar images found."}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
