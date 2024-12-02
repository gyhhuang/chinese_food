import pandas as pd
from sklearn.model_selection import train_test_split


###### PARAMETERS ######
threshold = 200
test_size = 0.1
data_path = 'haochi_ai/parsed_data/full_dataset.csv'
id_to_food_path = 'haochi_ai/food_dict_final.csv'
########################


data = pd.read_csv(data_path)
id_to_food = pd.read_csv(id_to_food_path)

dish_count_dict = dict(sorted(data['dish_id'].value_counts().to_dict().items()))

dish_count_df = pd.DataFrame(dish_count_dict.items(), columns=['dish_id', 'count'])
dish_count_df.to_csv('haochi_ai/parsed_data/dish_id_count.csv', index=False)

ineligible_foods = [key for key in dish_count_dict.keys() if dish_count_dict[key] < threshold]
eligible_foods = [key for key in dish_count_dict.keys() if dish_count_dict[key] >= threshold]

print('Ineligible foods:', ineligible_foods)

filtered_data = data[~data['dish_id'].isin(ineligible_foods)]

train_data, test_data = train_test_split(
    filtered_data,
    test_size=test_size,
    stratify=filtered_data['dish_id'], 
    random_state=42  
)

train_data.to_csv('haochi_ai/parsed_data/train_data.csv', index=False)
test_data.to_csv('haochi_ai/parsed_data/test_data.csv', index=False)
