from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import easyocr
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ------------------------------
# MODEL AND TOKENIZER LOADING
# ------------------------------
MODEL_NAME = "abclexd/memesensex"

try:
    print(f"🔄 Loading model from Hugging Face: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    tokenizer, model = None, None

# Initialize EasyOCR reader (Tagalog + English)
reader = easyocr.Reader(['en', 'tl'], gpu=False)


# ------------------------------
# INFERENCE FUNCTION
# ------------------------------
def run_inference(image_bytes):
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Extract text using OCR
        result_text = reader.readtext(image)
        extracted_text = " ".join([res[1] for res in result_text])

        # Prepare input for model
        inputs = tokenizer(extracted_text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=1).item()

        label = "Sexual" if prediction == 1 else "Non-Sexual"

        return {
            "text": extracted_text,
            "prediction": label,
            "confidence": round(probs[0][prediction].item(), 3)
        }

    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}


# ------------------------------
# ROUTES
# ------------------------------
@app.route("/process_predict", methods=["POST"])
def process_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    image_bytes = image.read()

    result = run_inference(image_bytes)
    if "error" in result:
        return jsonify(result), 500

    return jsonify({
        "status": "success",
        "message": "Image processed successfully",
        "data": result
    }), 201


# ------------------------------
# MAIN ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)