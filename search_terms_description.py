from search_parser import *

search_string = 'this is test input'
dictionary = make_dictionary(glove_file)
results = find_ten(search_string.split(),dictionary)

for word in search_string.split():
    print(word+':'+str(results[word]))