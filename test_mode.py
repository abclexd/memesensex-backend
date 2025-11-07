import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
import numpy as np
import re
import os
import io
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_multimodal_v3.pth")

# =========================
# 1. Text Preprocessing
# =========================
def preprocess_text(text):
    emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002B00-\U00002BFF"  # arrows, etc.
        "\U0001FA70-\U0001FAFF"  # extended symbols
        "]+",
        flags=re.UNICODE
    )
    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    # Lowercase and strip
    text = text.lower().strip()
    # Keep letters (including accented), and spaces
    text = re.sub(r'[^a-zñáéíóúü\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text

# =========================
# 2. OCR Extraction
# =========================
def ocr_extract_text(image_path, confidence_threshold=0.6):
    reader = easyocr.Reader(['en', 'tl'], gpu=torch.cuda.is_available())
    # # preprocess image for ocr
    # image = cv2.cvtColor(image_path, cv2.COLOR_RGB2GRAY)
    # # image = cv2.GaussianBlur(image,(5,5),0)
    
    # result = reader.readtext(image, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7)

    # # Extract text and confidence scores
    # texts = []
    # confidences = []
    
    # for detection in result:
    #     bbox, text, confidence = detection
    #     texts.append(text)
    #     confidences.append(confidence)
    # final_text = " ".join(texts)
    # preprocess_txt = preprocess_text(final_text)
    # avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    # return final_text, preprocess_txt, avg_confidence

    # Convert to grayscale
    gray = cv2.cvtColor(image_path, cv2.COLOR_RGB2GRAY)

    # First pass: OCR on raw grayscale
    result = reader.readtext(gray, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7)
    texts, confidences = [], []

    for detection in result:
        if len(detection) == 3:
            _, text, conf = detection
        else:
            text, conf = detection

        if isinstance(text, list):
            text = " ".join([str(t) for t in text if isinstance(t, str)])

        texts.append(text)
        try:
            confidences.append(float(conf))
        except (ValueError, TypeError):
            confidences.append(0.0)

    final_text = " ".join(texts)
    avg_conf = sum(confidences)/len(confidences) if confidences else 0.0

    # If confidence is low, retry with Gaussian blur
    if avg_conf < confidence_threshold:
        texts, confidences = [], []
        gauss_img = cv2.GaussianBlur(gray, (5,5), 0)
        result = reader.readtext(gauss_img, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7)

        for detection in result:
            if len(detection) == 3:
                _, text, conf = detection
            else:
                text, conf = detection

            if isinstance(text, list):
                text = " ".join([str(t) for t in text if isinstance(t, str)])

            texts.append(text)
            try:
                confidences.append(float(conf))
            except (ValueError, TypeError):
                confidences.append(0.0)

        final_text_gauss = " ".join(texts)
        avg_conf_gauss = sum(confidences)/len(confidences) if confidences else 0.0

        # Keep the version with higher confidence
        if avg_conf_gauss > avg_conf:
            final_text, avg_conf = final_text_gauss, avg_conf_gauss
    
    if not final_text:
        return "", "", 0.0

    preprocess_txt = preprocess_text(final_text)
    return final_text, preprocess_txt, avg_conf


# =========================
# 3. Image Preprocessing
# =========================
def resize_normalize_image(image_path, target_size=(224, 224)):

    preprocess_image = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = preprocess_image(image_path).unsqueeze(0)  # Add batch dimension
    return image_tensor


# =========================
# 4. Model Definitions
# =========================
class CrossAttentionModule(nn.Module):
    def __init__(self, query_dim, key_value_dim, hidden_dim=256, num_heads=8, dropout=0.1):
        super(CrossAttentionModule, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)  # √dk

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query projection for H (image features)
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        # Key and Value projections for G (text features)
        self.key_proj = nn.Linear(key_value_dim, hidden_dim)
        self.value_proj = nn.Linear(key_value_dim, hidden_dim)

        # Output projection WO
        self.out_proj = nn.Linear(hidden_dim, query_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # MLP for final transformation
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, H, G):
        """
        Args:
            H: Query features [batch_size, seq_len_h, query_dim] (e.g., image patches)
            G: Key/Value features [batch_size, seq_len_g, key_value_dim] (e.g., text tokens)
        """
        batch_size, seq_len_h, _ = H.shape
        seq_len_g = G.shape[1]

        # Step 1: Project to Q, K, V
        Q = self.query_proj(H)  # WiQ H
        K = self.key_proj(G)    # WiK G
        V = self.value_proj(G)  # WiV G

        # Step 2: Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_h, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_g, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_g, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Compute attention ATTi(H,G) = softmax((WiQ H)T(WiK G)/√dk)(WiV G)T
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)

        # Step 4: Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_h, self.hidden_dim
        )

        # MATT(H,G) = [ATT1...ATTh]WO
        matt_output = self.out_proj(attention_output)

        # Step 5: Z = LN(H + MATT(H,G))
        Z = self.norm1(H + matt_output)

        # Step 6: TIM(H,G) = LN(Z + MLP(Z))
        mlp_output = self.mlp(Z)
        tim_output = self.norm2(Z + mlp_output)

        return tim_output

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name='jcblaise/roberta-tagalog-base'):
        super(MultimodalClassifier, self).__init__()

        # Image encoder (ResNet-18, keep spatial features)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # keep until last conv (before avgpool)
        self.image_encoder = nn.Sequential(*modules)  # output: (B, 512, 7, 7)

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(model_name)

        # Cross-attention using paper formula
        # Image attends to text
        self.img_to_text_attention = CrossAttentionModule(
            query_dim=512,
            key_value_dim=self.text_encoder.config.hidden_size,
            hidden_dim=256,
            num_heads=8
        )

        # Text attends to image
        self.text_to_img_attention = CrossAttentionModule(
            query_dim=self.text_encoder.config.hidden_size,
            key_value_dim=512,
            hidden_dim=256,
            num_heads=8
        )

        # Fusion & classifier
        self.fusion = nn.Sequential(
            nn.Linear(512 + self.text_encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # Extract image features
        batch_size = images.size(0)
        img_feats = self.image_encoder(images)           # (B, 512, 7, 7)
        img_feats = img_feats.flatten(2).permute(0, 2, 1)  # (B, 49, 512)

        # Extract text features
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = text_outputs.last_hidden_state  # (B, seq_len, H)

        # Cross-attention following paper formula
        # TIM(img_feats, txt_feats) and TIM(txt_feats, img_feats)
        attended_img = self.img_to_text_attention(img_feats, txt_feats)
        attended_txt = self.text_to_img_attention(txt_feats, img_feats)

        # Pool attended outputs
        img_repr = attended_img.mean(dim=1)  # (B, 512)
        txt_repr = attended_txt[:, 0, :]     # CLS token (B, hidden_size)

        # Fusion
        fused = torch.cat([img_repr, txt_repr], dim=1)
        return self.fusion(fused)

# =========================
# 5. Load Model & Tokenizer
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultimodalClassifier(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("jcblaise/roberta-tagalog-base")

# =========================
# 6. Inference Function
# =========================
def run_inference(image_path):
    # Convert bytes → PIL image
    if isinstance(image_path, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_path)).convert("RGB")
    elif isinstance(image_path, str):
        pil_img = Image.open(image_path).convert("RGB")
    elif isinstance(image_path, Image.Image):
        pil_img = image_path.convert("RGB")
    else:
        raise TypeError(f"Unsupported input type: {type(image_path)}")

    # OCR
    np_image= np.array(pil_img)
    raw_text, clean_text, confidence= ocr_extract_text(np_image)

    if clean_text == "":
        return {
            "error": "This is not a meme. Upload a valid meme image with text.",
        }
    
    elif len(clean_text.split()) < 3:
        return {
            "error": "Insufficient text detected in the meme. Please upload a meme with more text. Minimum is 3 words.",
            "clean_text": clean_text,
            "raw_text": raw_text,
            "confidence": confidence
        }

    # Image
    img_tensor = resize_normalize_image(pil_img).to(device)

    # Tokenize text
    encoding = tokenizer(
        clean_text, return_tensors='pt',
        padding=True, truncation=True, max_length=128
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    

    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_class = 'sexual' if pred_class == 1 else 'non-sexual'

    return {
        'original_size': pil_img.size,
        'prediction': pred_class,
        'probabilities': probs.cpu().numpy().tolist(),
        'raw_text': raw_text,
        'clean_text': clean_text,
        'confidence': confidence
    }


# =========================
# 7. Run as main
# =========================
# if __name__ == "__main__":
#     # Example: load image from path
#     IMAGE_PATH = "backend/OIP (1).jfif"

#     # test_dimension_sensitivity(IMAGE_PATH)

#     result = run_inference(IMAGE_PATH)

#     # Print results
#     print("Original Image Size:", result['original_size'])
#     print("Prediction:", result['prediction'])
#     print("Probabilities:", result['probabilities'])
#     print("Raw OCR Text:", result['raw_text'])
#     print("Clean OCR Text:", result['clean_text'])
#     print("OCR Confidence:", result['confidence'])


#     # Preprocess image
#     pil_img = Image.open(IMAGE_PATH).convert("RGB")
#     img_tensor = resize_normalize_image(pil_img).to(device)

#     # -----------------------------
#     # Generate ResNet heatmap
#     # -----------------------------
#     features = {}
#     def hook_fn(module, input, output):
#         features['value'] = output.detach()

#     last_conv = model.image_encoder[-1]
#     hook_handle = last_conv.register_forward_hook(hook_fn)

#     with torch.no_grad():
#         _ = model(img_tensor, 
#                 input_ids=torch.zeros(1,1, dtype=torch.long, device=device), 
#                 attention_mask=torch.ones(1,1, dtype=torch.long, device=device))

#     hook_handle.remove()

#     feat_tensor = features['value']
#     heatmap_grid = feat_tensor[0].mean(dim=0).cpu().numpy()
#     heatmap_grid = (heatmap_grid - heatmap_grid.min()) / (heatmap_grid.max() - heatmap_grid.min())
#     heatmap_resized = np.uint8(255 * heatmap_grid)
#     heatmap_resized = Image.fromarray(heatmap_resized).resize(pil_img.size, Image.NEAREST)
#     heatmap_resized = np.array(heatmap_resized)

#     probs = result['probabilities'][0]
#     prob_text = f"non-sexual: {probs[0]:.2f}, sexual: {probs[1]:.2f}"

#     # -----------------------------
#     # Plot everything in one figure
#     # -----------------------------
#     fig, ax = plt.subplots(figsize=(6,6))

#     ax.imshow(pil_img)  # original image
#     ax.imshow(heatmap_resized, cmap='jet', alpha=0.4, interpolation='nearest')  # overlay heatmap
#     ax.axis('off')
#     ax.set_title(f"{result['prediction']} ({prob_text})", fontsize=14, color='blue')

#     # Add colorbar
#     sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label('Feature Intensity')

#     plt.show()