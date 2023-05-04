
def getExactCardinality(data):
    data_set = {val for val in data}
    return len(data_set)



# with open('shakespeare_x_2_new_card.csv', 'r') as file:
#     data = file.read().split(',')
#     print(len(data))
#     data_set = {val for val in data}
#     print(len(data_set))