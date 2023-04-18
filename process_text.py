import string

# with open('shakespeare.txt', 'r') as file:
#     data = file.read().replace('\n', ' ').replace('.', ' ')
#     data = data.translate(str.maketrans('','',string.punctuation))
#     with open('temp.txt', 'w') as output:
#         output.write(data)

with open('temp.txt', 'r') as file:
    data = file.read()
    data = ','.join(' '.join(data.split()).split(' '))
    with open('shakespeare.csv', 'w') as output:
        output.write(data)
