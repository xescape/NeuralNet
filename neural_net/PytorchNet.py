import sys, os, csv, logging, argparse
import pandas as pd
import sklearn as skl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV, LassoLars
from pathlib import Path
from sklearn.model_selection import KFold

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

def run(df, meta_df, out_path, params):

    print('Input shape is {0}'.format(str(df.shape)))
    logger = logging.getLogger('main')
    logger.info('New run with params layers: {0}, dropout {1}'.format(str(params['layers']), params['dropout']))

    data = df.values
    meta = meta_df.values

    #SMALL SET
    data = data
    meta = meta

    #scale data to [0, 1]
    mms = MinMaxScaler()
    data = mms.fit_transform(data)
    # meta = normalize(meta, norm='max', axis = 0) #TODO: RUN THIS
        
    #prefilter
    #try to reuse prefiltering results, but will redo if it didn't work
    prefilter_path = out_path / 'prefilter.txt'
    try:
        with open(prefilter_path, 'r') as f:
            reader = csv.reader(f, dialect = 'excel-tab', quoting=csv.QUOTE_NONNUMERIC)
            data_idx = list(map(int, next(reader)))
            # print(data_idx)
            filtered_data = data[:, data_idx]
        print('existing features used')
    except:
        print('no existing features, prefiltering')
        data_idx, filtered_data = prefiltering(data, meta)
        with open(prefilter_path, 'w', newline = '') as f:
            writer = csv.writer(f, dialect = 'excel-tab')
            writer.writerow(data_idx)

    logger.info('filtered data shape is ' + str(filtered_data.shape))

    n_folds = 5
    kf = KFold(n_splits=n_folds)
    out_path = in_path / 'results'
    out_path.mkdir(parents=True, exist_ok=True)
    
    for train_index, test_index in kf.split(data):

        f_data_train, f_data_test = filtered_data[train_index], filtered_data[test_index]
        meta_train, meta_test = meta[train_index], meta[test_index]

        model, avg_error, avg_expected_error = trainNet(f_data_train, f_data_test, meta_train, meta_test, params)
        # write_result(meta_test, prefilter_results, out_path / 'dense_filtered.tsv')
    
    print('end of run')


def prefiltering(data, meta):
    #make a linear model and get rid of the bottom stuff
    # model = LassoLarsCV(n_jobs=-1)
    model = LassoLars(alpha=0.001)
    model.fit(data, meta.reshape((meta.shape[0],)))
    coefs = np.abs(model.coef_)

    good_idx = np.where(coefs > 0)[0]

    res = data[:,good_idx]

    print('{0} features selected through prefiltering'.format(len(good_idx)))
    return good_idx, res

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

class Net(nn.Module):
    def __init__(self, input_dims, neurons, dropout):
        super(Net, self).__init__()
        #layers

        self.layers = [nn.Linear(input_dims[1],neurons[0])]
        for i in range(0, len(neurons) - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i+1]))
        
        self.predict = nn.Linear(neurons[-1], 1)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())
        self.training = True
    
    def forward(self, x):
        for l in self.layers:
            x = F.relu(l(x.float()))
            x = F.dropout(x, training = self.training)
        
        x = self.predict(x)
        return x

def trainNet(data_train, data_test, meta_train, meta_test, params):

    n_epochs = 1000
    net = Net(data_train.shape, params['layers'], params['dropout'])
    
    criterion = net.criterion
    optimizer = net.optimizer
    parallel_net = nn.DataParallel(net)


    
    train_loader = torch.utils.data.DataLoader([(d, m) for d, m in zip(data_train, meta_train)], batch_size = 24, shuffle = True)
    test_loader = torch.utils.data.DataLoader([(d, m) for d, m in zip(data_test, meta_test)], batch_size = 24, shuffle = True)
    running_loss = 0.0 

    for epoch in range(n_epochs):
        net.training = True
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = parallel_net(inputs)
            loss = criterion(outputs.double(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 200 == 199:
            net.training = False
            running_error = 0
            running_expected_error = 0
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data 
                outputs = net(inputs)
                error = criterion(outputs, labels) 
                expected_error = criterion(torch.full_like(labels, np.average(labels)), labels)
                running_error += error.item()
                running_expected_error += expected_error.item()
            avg_error = running_error / data_test.shape[0]
            avg_expected_error = running_expected_error / data_test.shape[0]
            print('epoch: {0}, average loss: {1}, expected loss: {2}, ratio: {3}'.format(epoch, avg_error, avg_expected_error, avg_error / avg_expected_error))

    return net, avg_error, avg_expected_error

def importData(paintings_path, meta_path):
    '''
    imports the chromsome painting and meta data
    '''
    #for the paintings
    # if alt:
    #     df = read
    df = pd.read_csv(paintings_path, sep='\t', header=0, index_col=0)

    #for the meta
    meta_df = pd.read_csv(meta_path, sep='\t', header=0, index_col=0)
    
    df, meta_df = df.align(meta_df, axis=0, join='inner')
    
    return df, meta_df

def write_result(meta_test, result, out_path):

    if out_path.is_file():
        header = False
    else:
        header = True

    if not out_path.parent.is_dir():
        out_path.parent.mkdir(parents=True)

    with open(out_path, 'a') as output:
        if header:
            output.write('real\tpredict\n')
        else:
            output.write('\n')

        main_text = ['{0}\t{1}'.format(pair[0], pair[1]) for pair in zip(list(meta_test.reshape((-1,))), list(result.reshape((-1,))))]
        output.write('\n'.join(main_text))

def configLogger(_path):
    
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s:\n%(message)s\n')
    
    fh = logging.FileHandler(_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

if __name__ == "__main__":

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    #default paths
    # default_local_input_path = Path('D:\\Documents\\data\\plasmo\\nn_scalar')
    default_local_input_path = Path('/d/data/plasmo/nn_scalar2')
    default_local_output_path = default_local_input_path / 'output'
    default_scinet_input_path = Path('/home/xescape/scratch/nn_scalar')
    default_scinet_output_path = default_scinet_input_path / 'output'
    default_layers = [32, 16]
    default_dropout = 0.5

    #parse args
    parser = argparse.ArgumentParser(description = 'Runs the NN model using Keras')
    parser.add_argument('--local', action = 'store_true', default = False, dest = 'local')
    parser.add_argument('-i', '--input', action = 'store', dest = 'input')
    parser.add_argument('-o', '--output', action = 'store', dest = 'output')
    parser.add_argument('-l', '--layers', action = 'store', nargs = "*", default = default_layers, type = int, dest = 'layers')
    parser.add_argument('-d', '--dropout', action = 'store', default = default_dropout, type = float, dest = 'dropout')
    #for the new transformed data
    #defaults to running on scinet beluge in case I forget

    args = vars(parser.parse_args())

    if args['input']:
        in_path = Path(args['input'])
    elif args['local'] == True :
        in_path = default_local_input_path 
    else:
        in_path = default_scinet_input_path
    
    if args['output']:
        out_path = Path(args['output'])
    elif args['local'] == True :
        out_path = default_local_output_path 
    else:
        out_path = default_scinet_output_path
    
    params = {
        'layers': args['layers'],
        'dropout': args['dropout']
    }

    data_path = in_path / 'combined_scores.tsv'
    meta_path = in_path / 'meta.tsv'
    

    #make the log folder if it's not there
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / 'logs.txt'
    configLogger(log_path)

    df, meta_df = importData(data_path, meta_path)
    run(df, meta_df, out_path, params)

    print('end!')

    
    

    




