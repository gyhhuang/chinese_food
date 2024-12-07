import numpy as np
import pandas as pd

###### PARAMETERS ######
data_path = 'haochi_ai/data_scripts/test_data.csv'
id_to_food_path = 'haochi_ai/food_dict_final.csv'
########################

data = pd.read_csv(data_path)
id_to_food = pd.read_csv(id_to_food_path)

dish_count_dict = dict(sorted(data['dish_id'].value_counts().to_dict().items()))

dish_count_df = pd.DataFrame(dish_count_dict.items(), columns=['dish_id', 'count'])

counts = dish_count_df['count'].to_numpy()

max_count = np.max(counts)
min_count = np.min(counts)
range_count = max_count - min_count

print(max_count)
print(min_count)
print(range_count)
