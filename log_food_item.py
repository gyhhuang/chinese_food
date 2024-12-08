import argparse
import csv
import json
import os
from datetime import datetime
import torch
from PIL import Image
from gtts import gTTS
from util_scripts.resnet import load_resnet_model, get_transforms

FOOD_MENU_DIR = "food_menu"
os.makedirs(FOOD_MENU_DIR, exist_ok=True)


def load_food_data(csv_file):
    food_data = {}
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            food_data[row["dish_id"]] = {
                "chinese": row["dish_hanzi"],
                "english": row["dish_english"],
                "pinyin": row["dish_pinyin"]
            }
    return food_data


def generate_audio(text, lang, filepath):
    """
    Generates an MP3 file using gTTS for the given text and language.
    """
    tts = gTTS(text=text, lang=lang)
    tts.save(filepath)
    return filepath


def create_food_log(prediction, food_data):
    """
    Creates a JSON object and corresponding files for the predicted food item.
    """
    if prediction not in food_data:
        raise ValueError(f"Prediction '{prediction}' not found in food database.")

    food_info = food_data[prediction]
    chinese = food_info["chinese"]
    pinyin_text = food_info["pinyin"]
    english = food_info["english"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sanitized_english = english.replace(' ', '_')

    # Create a subdirectory for this food item
    food_dir = os.path.join(FOOD_MENU_DIR, f"{prediction}_{sanitized_english}")
    os.makedirs(food_dir, exist_ok=True)

    # Generate audio files
    chinese_audio_path = os.path.join(food_dir, f"{sanitized_english}_chinese.mp3")
    english_audio_path = os.path.join(food_dir, f"{sanitized_english}_english.mp3")
    generate_audio(chinese, lang="zh", filepath=chinese_audio_path)
    generate_audio(english, lang="en", filepath=english_audio_path)

    # Create the JSON object
    food_item = {
        "dish_id": prediction,
        "chinese_characters": chinese,
        "pinyin": pinyin_text,
        "english_translation": english,
        "date_logged": timestamp,
        "audio_files": {
            "chinese": chinese_audio_path,
            "english": english_audio_path
        }
    }

    # Save the JSON object to a file
    json_path = os.path.join(food_dir, f"{prediction}_{sanitized_english}_data.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(food_item, json_file, indent=4, ensure_ascii=False)

    return food_item


def predict_food(model, image_path, transform, food_data_csv):
    """
    Predicts the food item from the image and returns its log.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    # Create the food log using the predicted class
    predicted_class_str = str(predicted_class)
    if predicted_class_str in food_data_csv:
        return create_food_log(predicted_class_str, food_data_csv)
    else:
        raise ValueError(f"Predicted class {predicted_class} not found in the food dictionary.")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the food item from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    csv_file_path = "data_scripts/csv/food_dict_final.csv"
    model_path = "data_scripts/rw_test_set/best_model.pth"

    food_data_csv = load_food_data(csv_file_path)
    model = load_resnet_model(model_path, num_classes=208)
    transform = get_transforms()

    try:
        food_log = predict_food(model, args.image_path, transform, food_data_csv)
        print(f"Food log created successfully: {json.dumps(food_log, indent=4, ensure_ascii=False)}")
    except ValueError as e:
        print(e)
