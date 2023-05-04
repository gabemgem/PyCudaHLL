import string

def remove_punc_and_newlines():
    with open('shakespeare.txt', 'r') as file:
        data = file.read().replace('\n', ' ').replace('.', ' ')
        data = data.translate(str.maketrans('','',string.punctuation))
        with open('temp.txt', 'w') as output:
            output.write(data)

def make_into_csv():
    with open('temp.txt', 'r') as file:
        data = file.read()
        data = ','.join(' '.join(data.split()).split(' '))
        with open('shakespeare.csv', 'w') as output:
            output.write(data)

def make_larger_same_card(multiply:int):
    with open('shakespeare.csv') as file:
        data = file.read()
        with open(f'shakespeare_x_{multiply}.csv', 'w') as output:
            output.write(data)
            for i in range(1, multiply):
                output.write(',')
                output.write(data)

def make_larger_new_card(multiply:int):
    with open('shakespeare.csv') as file:
        data = file.read()
        with open(f'shakespeare_x_{multiply}_new_card.csv', 'w') as output:
            output.write(data)
            split_data = data.split(',')
            for i in range(1, multiply):
                temp = ','.join([v+str(i) for v in split_data])
                output.write(',')
                output.write(temp)

make_larger_same_card(5)
make_larger_new_card(5)

make_larger_same_card(10)
make_larger_new_card(10)

make_larger_same_card(50)
make_larger_new_card(50)

make_larger_same_card(100)
make_larger_new_card(100)