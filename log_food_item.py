import os
import json
from gtts import gTTS
from datetime import datetime
import csv

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

    # Create a subdirectory for this food item
    food_dir = os.path.join(FOOD_MENU_DIR, f"{prediction}_{english}")
    os.makedirs(food_dir, exist_ok=True)

    # Generate audio files
    chinese_audio_path = os.path.join(food_dir, f"{english}_chinese.mp3")
    english_audio_path = os.path.join(food_dir, f"{english}_english.mp3")
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
    json_path = os.path.join(food_dir, f"{prediction}_{english}_data.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(food_item, json_file, indent=4, ensure_ascii=False)

    return food_item


# Example usage
if __name__ == "__main__":
    csv_file_path = "food_list.csv"
    food_data_csv = load_food_data(csv_file_path)

    TEST_INPUT = "4"
    try:
        food_log = create_food_log(TEST_INPUT, food_data_csv)
        print(f"Food log created successfully: {json.dumps(food_log, indent=4, ensure_ascii=False)}")
    except ValueError as e:
        print(e)
