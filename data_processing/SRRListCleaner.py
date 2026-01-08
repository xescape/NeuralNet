'''
Created on Oct 25, 2018

@author: Javi
'''
import re 
import os

def loadTable(input_path):
    '''
    loads the SRARunTable and filters out some of the useless columns. useful ones are defined in key _columns.
    '''
    key_columns = [0, 11]
    
    with open(input_path, 'r') as input:
        data = input.read()
    
    data = re.split('\n', data.strip())
    data = [re.split('\t', x.strip()) for x in data[1:]]
    
    results = []
    transposed_table = [list(x) for x in zip(*data)]
    for column in key_columns:
        results.append(transposed_table[column]) 
    
    return results

def loadList(list_path):
    '''
    loads the full sample list, to be filtered
    '''
    with open(list_path, 'r') as input:
        data = input.read()
    
    data = re.split('\n', data.strip())
    
    return data

def filterList(table, list):
    '''
    filters the list according to some stuff on the table
    right now it's according to unique biosample
    '''
    
    print(table)

    biosamples = set()
    results = []
    
    for samp, acc in zip(*table):
        if samp not in biosamples and acc in list:
            biosamples.add(samp)
            results.append(acc)
    
    return results

def output(results, output_path):
    
    with open(output_path, 'w') as output:
        output.write('\n'.join(results))
    
# def removefiles(results, files_path, files_out_path):
#     
#     os.chdir(files_path)    
    
if __name__ == '__main__':

    dir = '/d/data/plasmo'
    input_path = '/'.join([dir, 'SraRunTable.txt'])
    list_path = '/'.join([dir, 'SRR_Acc_List.txt'])
    output_path = '/'.join([dir, 'new_accs.txt'])
    files_path = '/'.join([''])
    files_out_path = files_path + '/out'
    
    table = loadTable(input_path)
    list = loadList(list_path)
    new_list = filterList(table, list)
    output(new_list, output_path)
    
    print('script finished')