import pandas as pd
from pypinyin import pinyin, Style

data = pd.read_csv('../data_scripts/csv/food_dict.csv')


def convert_to_pinyin(text):
    pinyin_list = pinyin(text, style=Style.TONE3, v_to_u=True)  # TONE3 includes tones as numbers
    return ' '.join([item[0] for item in pinyin_list])


data['dish_pinyin'] = data['dish_hanzi'].apply(convert_to_pinyin)

data.to_csv('food_dict_final.csv', index=False)

print(data)
