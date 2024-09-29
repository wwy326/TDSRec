import pandas as pd

# 读取CSV文件
data = pd.read_csv('../cd/cd.csv', skiprows=1, names=['user_id', 'item_id', 'time'])
data['time'] = data['time'].astype(int)
train_data = ['user_id'+'\t'+'item_seq'+'\t'+'last_item'+'\t'+'time_item_seq'+'\t'+'time_last_item']
data_cl = []
# 按用户id进行分组，并按时间戳排序
grouped_data = data.sort_values(by='time').groupby('user_id')

# 处理每个用户的数据
test_data = ['user_id'+'\t'+'item_seq'+'\t'+'last_item'+'\t'+'time_item_seq'+'\t'+'time_last_item']
for user_id, group in grouped_data:
    if len(group) <= 5:
        pass
    else:
        item_ids = group['item_id'].tolist()
        sorted_item_ids = sorted(item_ids)
        last_item_id = sorted_item_ids[-1]
        time_list = group['time'].tolist()
        time_list_except_last = time_list[:-1]
        time_list_last = time_list[-1]
        data_cl.append( f"{' '.join(map(str, sorted_item_ids[:-1]))}")
        test_data.append(
            f"{user_id}\t{' '.join(map(str, sorted_item_ids[:-1]))}\t{last_item_id}\t{' '.join(map(str, time_list_except_last))}\t{time_list_last}")
        # train_data.append(f"{user_id}:{','.join(map(str, group['time'].tolist()))}")
        train_data.append(
            f"{user_id}\t{' '.join(map(str, sorted_item_ids[:-2]))}\t{sorted_item_ids[-2]}\t{' '.join(map(str, time_list[:-2]))}\t{time_list[-2]}")


# 将 test_data 写入文件
# with open('test_cd.csv', 'w') as file:
#     for data in test_data:
#         file.write(data + '\n')
#
# 将 train_data 写入文件
# with open('train_cd.csv', 'w') as file:
#     for data in train_data:
#         file.write(data + '\n')


with open('cd_item.csv', 'w') as file:
    for data in data_cl:
        file.write(data + '\n')