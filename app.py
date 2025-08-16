import os
import uuid
import pathlib
import io
import base64
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = genai.Client(api_key="AIzaSyCPw79xvCt4ZNOXJh4ORZ0OBZ4S7bZka7U")
MODEL_ID = "gemini-2.0-flash-exp"
# Image storage
IMAGE_DIR = "tryon_results"
os.makedirs(IMAGE_DIR, exist_ok=True)

def log_image_info(image_data, prefix=""):
    """Helper function to log image information"""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        logger.info(f"{prefix} Image received - Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        logger.error(f"{prefix} Failed to process image info: {str(e)}")

def combine_images(shirt_img_data, user_img_data):
    """Combine shirt and user images side by side"""
    logger.info("Combining images...")
    
    try:
        shirt_img = Image.open(io.BytesIO(base64.b64decode(shirt_img_data)))
        user_img = Image.open(io.BytesIO(base64.b64decode(user_img_data)))
        
        # Resize to match height while maintaining aspect ratio
        target_height = max(shirt_img.height, user_img.height)
        shirt_img = shirt_img.resize((int(target_height * (shirt_img.width/shirt_img.height)), target_height))
        user_img = user_img.resize((int(target_height * (user_img.width/user_img.height)), target_height))
        
        # Create combined image
        combined = Image.new('RGB', (shirt_img.width + user_img.width, target_height), (255, 255, 255))
        combined.paste(shirt_img, (0, 0))
        combined.paste(user_img, (shirt_img.width, 0))
        
        logger.info(f"Combined image created - Size: {combined.size}")
        return combined
        
    except Exception as e:
        logger.error(f"Error combining images: {str(e)}")
        raise

def save_image(image, prefix="result"):
    """Save image to disk and return filename"""
    try:
        filename = f"{prefix}_{uuid.uuid4()}.png"
        filepath = os.path.join(IMAGE_DIR, filename)
        image.save(filepath, "PNG")
        logger.info(f"Saved image: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise



# Gemini setup


# Helper function to decode base64 image and save it temporarily
def decode_and_save_base64_image(base64_data, filename):
    try:
        image_data = base64.b64decode(base64_data)
        filepath = os.path.join(IMAGE_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        return filepath
    except Exception as e:
        logger.error(f"Failed to decode and save image: {str(e)}")
        raise

@app.route("/try-on", methods=["POST"])
def virtual_tryon():
    logger.info("\n" + "=" * 50)
    logger.info("Received new try-on request")

    try:
        if not request.is_json:
            logger.error("Request must be JSON")
            return jsonify({"success": False, "error": "Request must be JSON"}), 400

        data = request.get_json()
        shirt_image_b64 = data.get("shirtImage")
        user_image_b64 = data.get("userImage")

        if not shirt_image_b64 or not user_image_b64:
            logger.error("Missing image data")
            return jsonify({"success": False, "error": "Both shirt and user images are required"}), 400

        # Save shirt image
        shirt_img_path = os.path.join(IMAGE_DIR, f"shirt_{uuid.uuid4()}.png")
        with open(shirt_img_path, "wb") as f:
            f.write(base64.b64decode(shirt_image_b64))
        logger.info(f"Shirt image saved: {shirt_img_path}")

        # Save user image
        user_img_path = os.path.join(IMAGE_DIR, f"user_{uuid.uuid4()}.png")
        with open(user_img_path, "wb") as f:
            f.write(base64.b64decode(user_image_b64))
        logger.info(f"User image saved: {user_img_path}")

        # Open images using PIL
        shirt_img = Image.open(shirt_img_path)
        user_img = Image.open(user_img_path)

        # Prompt
        prompt = """
     {
  "task": "virtual clothing try-on",
  "input": {
    "person_image": "Image of the target person",
    "shirt_image": "Image of the shirt to be applied"
  },
  "requirements": {
    "preserve_subject": {
      "pose": "Keep identical to the person in the input image",
      "body_proportions": "Do not alter height, width, or shape of the body",
      "facial_features": "Preserve all original facial details exactly",
      "skin_tone": "No changes to color or texture"
    },
    "apply_shirt": {
      "fit": "Natural and realistic, aligned with body contours",
      "integration": "Match lighting, shading, and perspective of the person image",
      "details": "Include folds, wrinkles, and texture from the shirt image"
    },
    "prohibitions": [
      "Do not modify the person's face or hair",
      "Do not alter background",
      "Do not change the shirt's original color or design"
    ]
  },
  "output": {
    "image": "Realistic image of the same person wearing the provided shirt"
  }
}
"""

        # Gemini multi-modal generation
        logger.info("Calling Gemini with multi-image input...")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, user_img, shirt_img],
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )

        # Save result
        filenames = []
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                filename = f"result_{uuid.uuid4()}.png"
                filepath = os.path.join(IMAGE_DIR, filename)
                pathlib.Path(filepath).write_bytes(part.inline_data.data)
                logger.info(f"Generated AI image saved: {filepath}")
                filenames.append(filename)

        if not filenames:
            return jsonify({"success": False, "error": "No image generated"}), 500

        return jsonify({
            "success": True,
            "imageUrl": f"/results/{filenames[0]}",
            "message": "AI virtual try-on complete"
        })

    except Exception as e:
        logger.error(f"Try-on failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/text-to-image", methods=["POST"])
def text_to_image():
    logger.info("\n" + "=" * 50)
    logger.info("Received new text-to-image request")

    try:
        if not request.is_json:
            logger.error("Request must be JSON")
            return jsonify({"success": False, "error": "Request must be JSON"}), 400

        data = request.get_json()
        prompt_text = data.get("prompt")

        if not prompt_text:
            logger.error("Missing prompt text")
            return jsonify({"success": False, "error": "Prompt text is required"}), 400

        logger.info(f"Generating image for prompt: {prompt_text[:100]}...")

        # Create a detailed prompt for better image generation
        enhanced_prompt = f"""
        Create a high-quality, detailed image based on this description: {prompt_text}
        
        Requirements:
        - High resolution and realistic details
        - Professional quality suitable for commercial use
        - Rich colors and proper lighting
        - Clear and focused subject matter
        - Artistic but realistic representation
        """

        # Call Gemini for text-to-image generation
        logger.info("Calling Gemini for text-to-image generation...")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[enhanced_prompt],
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )

        # Save generated image
        filenames = []
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                filename = f"text2img_{uuid.uuid4()}.png"
                filepath = os.path.join(IMAGE_DIR, filename)
                pathlib.Path(filepath).write_bytes(part.inline_data.data)
                logger.info(f"Generated text-to-image saved: {filepath}")
                filenames.append(filename)

        if not filenames:
            return jsonify({"success": False, "error": "No image generated"}), 500

        return jsonify({
            "success": True,
            "imageUrl": f"/results/{filenames[0]}",
            "message": "Text-to-image generation complete",
            "prompt": prompt_text
        })

    except Exception as e:
        logger.error(f"Text-to-image generation failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/results/<filename>")
def get_result(filename):
    try:
        filepath = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filename}")
            return jsonify({"error": "File not found"}), 404
            
        logger.info(f"Serving image: {filename}")
        return send_file(filepath, mimetype="image/png")
        
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify service is running"""
    return jsonify({
        "status": "healthy",
        "service": "Gemini AI Image Generation Service",
        "endpoints": [
            "/try-on - Virtual clothing try-on",
            "/text-to-image - Text to image generation",
            "/results/<filename> - Get generated images"
        ]
    })

if __name__ == "__main__":
    logger.info("Starting Gemini AI Image Generation server...")
    app.run(host="0.0.0.0", port=5000, debug=True)



