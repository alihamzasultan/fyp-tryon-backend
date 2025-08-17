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
import time
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variable (Railway will set this)
import os
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCPw79xvCt4ZNOXJh4ORZ0OBZ4S7bZka7U")
if not api_key:
    logger.error("GOOGLE_API_KEY environment variable not set!")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

client = genai.Client(api_key=api_key)
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
        time.sleep(60)

        # Prompt
        prompt = """
        {
          "task": "virtual try-on",
          "instructions": [
            "Use the provided person image as the base. Do not redraw or regenerate the person.",
            "Overlay and integrate the provided garment image onto the person.",
            "Do not change the person’s pose, body proportions, face, or hair.",
            "Do not change the background or lighting.",
            "Do not generate a new background (no white or artificial background).",
            "Maintain original resolution, camera angle, and style."
          ],
          "validation":{"if the image of the person is not in front pose or it is unclear or blur, than do not generate the output"}
          "output": {
            "image": "Realistic final image of the same person wearing the garment, with background unchanged."
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
                filename = f"generated_{uuid.uuid4()}.png"
                filepath = os.path.join(IMAGE_DIR, filename)
                pathlib.Path(filepath).write_bytes(part.inline_data.data)
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

# Add this helper function in your backend file
def generate_image(prompt):
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )

        filenames = []
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                filename = f"generated_{uuid.uuid4()}.png"
                filepath = os.path.join(IMAGE_DIR, filename)
                pathlib.Path(filepath).write_bytes(part.inline_data.data)
                filenames.append(filename)

        return filenames
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        return []
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_prompt = data.get("prompt", "A futuristic city")  # Note: Typo in "prompt" fixed below
    
    # Hardcoded base prompt
    base_prompt = """Generate a high-quality, photorealistic image of a Requested garment in a white clear background
- clear and front view of the garment
- Excellent fabric texture and details
- Professional product photography quality
- Clean background
- Well-lit with studio lighting
Specific requirements: """
    
    # Combine with user prompt
    full_prompt = base_prompt + user_prompt
    
    try:
        logger.info(f"Generating image with prompt: {full_prompt}")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                response_modalities=["Text", "Image"]
            )
        )

        # Process only the first image part found
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                filename = f"generated_{uuid.uuid4()}.png"
                filepath = os.path.join(IMAGE_DIR, filename)
                pathlib.Path(filepath).write_bytes(part.inline_data.data)
                logger.info(f"Generated image saved as {filename}")
                return jsonify({
                    "success": True,
                    "message": "Image generated successfully!",
                    "imageUrl": f"/results/{filename}"
                })

        return jsonify({"success": False, "error": "No image generated"}), 500

    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
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
    try:
        # Check if Gemini client is working
        api_key_status = "configured" if api_key else "missing"
        
        return jsonify({
            "status": "healthy",
            "service": "Gemini AI Image Generation Service",
            "api_key": api_key_status,
            "model": MODEL_ID,
            "endpoints": [
                "/try-on - Virtual clothing try-on",
                "/generate - Text to image generation",
                "/results/<filename> - Get generated images"
            ]
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "service": "Gemini AI Image Generation Service"
        }), 500

@app.route("/test", methods=["GET"])
def test_endpoint():
    """Simple test endpoint to verify basic functionality"""
    try:
        return jsonify({
            "message": "Backend is working!",
            "timestamp": str(datetime.datetime.now()),
            "python_version": "3.x"
        })
    except Exception as e:
        logger.error(f"Test endpoint failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Gemini AI Image Generation server...")
    # Railway sets PORT environment variable
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Server will run on port {port}")
    
    try:
        app.run(host="0.0.0.0", port=port, debug=False)  # Set debug=False for production
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise











