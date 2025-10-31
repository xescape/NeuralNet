import sys, os, csv, logging, argparse
import pandas as pd
import sklearn as skl
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers, utils, regularizers, backend as K
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV, LassoLars
from pathlib import Path
from sklearn.model_selection import KFold

from tensorflow.keras.losses import MAE as mae
from kerastuner.tuners import Hyperband
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def run(df, meta_df, out_path, params):

    print('Input shape is {0}'.format(str(df.shape)))
    logger = logging.getLogger('main')
    logger.info('New run with params layers: {0}, dropout {1}'.format(str(params['layers']), params['dropout']))

    data = df.values.astype(np.float64)
    meta = meta_df.values.astype(np.float64)

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
        data_idx, filtered_data = prefiltering(data, meta)
        with open(prefilter_path, 'w', newline = '') as f:
            writer = csv.writer(f, dialect = 'excel-tab')
            writer.writerow(data_idx)

    logger.info('filtered data shape is ' + str(filtered_data.shape))

    out_path = in_path / 'results'
    out_path.mkdir(parents=True, exist_ok=True)
    os.chdir(out_path)

    # n_folds = 5
    # kf = KFold(n_splits=n_folds)
    
    # for train_index, test_index in kf.split(data):

    #     f_data_train, f_data_test = filtered_data[train_index], filtered_data[test_index]
    #     meta_train, meta_test = meta[train_index], meta[test_index]
        

        # print('dense filtered')
    prefilter_results = trainPrefilterModel(filtered_data, meta, params, out_path)
    write_result(meta, prefilter_results, out_path / 'dense_filtered.tsv')
    
    print('end of run')

def trainLasso(data_train, data_test, meta_train, meta_test):
    model = LassoLarsCV(cv = 5, n_jobs = -1)
    
    model.fit(data_train, meta_train.reshape((meta_train.shape[0])))
    print('lasso model r2 = {0}'.format(model.score(data_test, meta_test.reshape((meta_test.shape[0])))))

    coefs = np.abs(model.coef_)
    sorted_idx = sorted(range(coefs.shape[0]), key = lambda x: coefs[x], reverse = True)

    return model.predict(data_test), sorted_idx

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



def trainPrefilterModel(data, meta, params, out_path):

    def build_model(hp):
        model = tf.keras.Sequential()
        l1_nodes = hp.Int('l1_nodes', min_value=8, max_value=96, step=8)
        l2_nodes = hp.Int('l2_nodes', min_value=8, max_value=96, step=8)
        l3_nodes = hp.Int('l3_nodes', min_value=8, max_value=96, step=8)
        dropout = hp.Choice('dropout', [0.3, 0.4, 0.5, 0.6])
        
        model.add(layers.Dense(l1_nodes, input_shape=(data.shape[1],), activation='relu', kernel_initializer='normal'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(l2_nodes, activation='relu', kernel_initializer='normal'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(l3_nodes, activation='relu', kernel_initializer='normal'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1, kernel_initializer='normal'))

        model.compile(optimizer='nadam',
                    loss = 'mse',
                    metrics=['mae'])

        return model

    input_dim = data.shape[1]
    logger = logging.getLogger('main')

    tuner = Hyperband(build_model, objective = 'val_loss', max_epochs = 500, project_name='v1a', directory=out_path)
    tuner.search(data, meta, epochs = 3, validation_split = 0.2, verbose=0)

    params = tuner.get_best_hyperparameters(num_trials=1)[0]
    try:
        logger.info('best params are: n_layers:{0}, n_nodes:{1}, scheme: {2}, dropout: {3}'.format(params.get('l1_nodes'), params.get('l2_nodes'), params.get('l3_nodes'), params.get('dropout')))
    except:
        logger.info('couldnt print best params')

    model = build_model(params)
    model.fit(data, meta, epochs=500, batch_size=10, shuffle=True, validation_split=0.25, verbose=0) #no tensorboard

    res = model.predict(data)
    error = np.sum(mae(meta, res), axis=None)
    expected_error = np.sum(mae(meta, np.full_like(meta, np.average(meta, axis=None))), axis=None)

    
    logger.info('Error of {0} in data with expected loss {1}. Ratio = {2}'.format(error, expected_error, error / expected_error))
    return model.predict(data)

# def mae(x, y):
#     return np.sum(np.abs(y - x))

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

    
    

    




