import pandas as pd

train_data = pd.read_csv('haochi_ai/train_list.csv')

train_dict = dict(sorted(train_data['dish_id'].value_counts().to_dict().items()))

id_to_food = pd.read_csv('haochi_ai/food_dict_final.csv')

food_dict = {}

for key in train_dict.keys():

    corresponding_row = id_to_food[id_to_food['dish_id'] == key]
    corresponding_food = corresponding_row['dish_english'].to_string(index=False)

    food_dict[corresponding_food] = train_dict[key]

threshold = 200

above_threshold = 0
below_threshold = 0

for key in food_dict.keys():
    if food_dict[key] > threshold:
        above_threshold += 1
    else:
        below_threshold += 1
        print(key, food_dict[key])

print('Above threshold:', above_threshold)
print('Below threshold:', below_threshold)




# train_data = pd.read_csv('haochi_ai/train_list.csv')
# val_data = pd.read_csv('haochi_ai/val_list.csv')
# test_data = pd.read_csv('haochi_ai/test_truth_list.csv')

# train_dict = dict(sorted(train_data['dish_id'].value_counts().to_dict().items()))
# val_dict = dict(sorted(val_data['dish_id'].value_counts().to_dict().items()))
# test_dict = dict(sorted(test_data['dish_id'].value_counts().to_dict().items()))

# total_counts = {}

# for key in train_dict.keys():
#     total_counts[key] = train_dict[key]

# for key in val_dict.keys():
#     if key in total_counts:
#         total_counts[key] += val_dict[key]
#     else:
#         total_counts[key] = val_dict[key]

# for key in test_dict.keys():
#     if key in total_counts:
#         total_counts[key] += test_dict[key]
#     else:
#         total_counts[key] = test_dict[key]

# print(total_counts)


