import easyocr
from PIL import Image
import re
from torchvision import transforms
import matplotlib.pyplot as plt

# data={"image_path":"", "text":"", "preprocess_image":""}

# to preprocess the text extracted from the meme
def preprocess_text(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002B00-\U00002BFF"  # arrows, etc.
        "\U0001FA70-\U0001FAFF"  # extended symbols
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# to extract and preprocess text from image using OCR
def ocr_extract_text(image_path):
    reader =  easyocr.Reader(['en', 'tl'], gpu=True)
    result = reader.readtext(image_path, detail=0)
    final_text = " ".join(result)
    preprocess_txt= preprocess_text(final_text)
    return final_text, preprocess_txt

# to resize and normalize image for model input
def resize_normalize_image(image, target_size= (224, 224)):
    preprocess_image= transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image).convert('RGB')

    image = preprocess_image(image)
    image = image.unsqueeze(0)
    return image

# if __name__ == "__main__":
#     input_image= "backend/test_image.jpg"
#     data["image_path"]= input_image
#     data["text"]= ocr_extract_text(input_image)
#     data["preprocess_image"]= resize_normalize_image(input_image)
#     print(data)



