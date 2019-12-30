import sys, os, csv, logging, argparse
import pandas as pd
import sklearn as skl
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, utils, regularizers, backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV
from pathlib import Path
from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def run(df, meta_df, out_path, params):

    print('Input shape is {0}'.format(str(df.shape)))
    logger = logging.getLogger('main')
    logger.info('New run with params layers: {0}, dropout {1}'.format(str(params['layers']), params['dropout']))

    data = df.to_numpy()
    meta = meta_df.to_numpy()

    #scale data to [0, 1]
    mms = MinMaxScaler()
    data = mms.fit_transform(data)
        
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
        n_f = 256
        data_idx, filtered_data = prefiltering(data, meta, n_f)
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
        # data_train, data_test = data[train_index], data[test_index]
        meta_train, meta_test = meta[train_index], meta[test_index]

        # print('lasso filtered')
        # lasso_filtered_results, sorted_idx = trainLasso(f_data_train, f_data_test, meta_train, meta_test)
        # write_result(meta_test, lasso_filtered_results, out_path / 'lasso_filtered.tsv')
        
        # print('lasso not filtered')
        # lasso_results, sorted_idx = trainLasso(data_train, data_test, meta_train, meta_test)
        # write_result(meta_test, lasso_results, out_path / 'lasso_nofilter.tsv')
        
        # print('dense filtered')
        prefilter_results = trainPrefilterModel(f_data_train, f_data_test, meta_train, meta_test, params)
        write_result(meta_test, prefilter_results, out_path / 'dense_filtered.tsv')

        # print('dense not filtered')
        # nofilter_results = trainPrefilterModel(data_train, data_test, meta_train, meta_test, log_path, model_path)
        # write_result(meta_test, nofilter_results, out_path / 'dense_nofilter.tsv')
    
    print('end of run')

def trainLasso(data_train, data_test, meta_train, meta_test):
    model = LassoLarsCV(cv = 5, n_jobs = -1)
    
    model.fit(data_train, meta_train.reshape((meta_train.shape[0])))
    print('lasso model r2 = {0}'.format(model.score(data_test, meta_test.reshape((meta_test.shape[0])))))

    coefs = np.abs(model.coef_)
    sorted_idx = sorted(range(coefs.shape[0]), key = lambda x: coefs[x], reverse = True)

    return model.predict(data_test), sorted_idx

def prefiltering(data, meta, n):
    #make a linear model and get rid of the bottom stuff
    model = LassoLarsCV(n_jobs=-1)
    model.fit(data, meta.reshape((meta.shape[0],)))
    coefs = np.abs(model.coef_)
    # coefs = coefs.reshape((coefs.shape[1],))
    # cutoff = max(sorted(coefs)[n], 0.00000000000000000000001) #minimum coefficient

    good_idx = np.where(coefs > 0)[0]

    res = data[:,good_idx]

    print('{0} features selected through prefiltering'.format(len(good_idx)))
    return good_idx, res

def makePrefilterModel(input_dim, neurons , dropout):
    '''
    this is the main function to make the model. start small
    '''
    model = tf.keras.Sequential()
    model.add(layers.Dense(input_dim, input_dim=input_dim, activation='relu', kernel_initializer='normal'))

    for l in neurons:
        model.add(layers.Dense(l, activation='relu', kernel_initializer='normal'))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, kernel_initializer='normal'))

    model.compile(optimizer=optimizers.Nadam(),
                loss = 'mse',
                metrics=['mae'])

    return model

def trainPrefilterModel(data_train, data_test, meta_train, meta_test, params):
    input_dim = data_train.shape[1]
    model = makePrefilterModel(input_dim, params['layers'], params['dropout'])

    model.fit(data_train, meta_train, epochs=1500, batch_size=input_dim, shuffle=True, validation_split=0.25, verbose=0) #no tensorboard

    res = model.evaluate(data_test, meta_test)
    
    logger = logging.getLogger('main')
    logger.info('Loss of {2} in data mean = {0} and std = {1}'.format(np.mean(meta_test), np.std(meta_test), str(res)))
    return model.predict(data_test)

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
    default_local_input_path = Path('D:\\Documents\\data\\plasmo\\nn_scalar')
    default_local_output_path = default_local_input_path / 'output'
    default_scinet_input_path = Path('/home/xescape/scratch/nn_scalar')
    default_scinet_output_path = default_scinet_input_path / 'output'
    default_layers = [128, 256, 64, 32, 16]
    default_dropout = 0.3

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

    
    

    




