'''
inspired by saliency
'''
import numpy as np
import tensorflow as tf
from pathlib import Path
import statistics as stats

'''
assuming we already loaded the data I guess
'''
def delta(data, idx, model, out_path):
    res = []
    abs_res = []
    for x in range(data.shape[0]):
        sample = data[x, :].reshape((1, -1))
        base = model.predict(sample)[0,0]
        diffs = np.zeros((sample.shape[1],))
        abs_diffs = np.zeros_like(diffs)
        for y in range(sample.shape[1]):
            tmp = sample.copy()
            if tmp[0, y] == 0:
                tmp[0, y] = 1
                factor = 1
            else:
                tmp[0, y] = 0
                factor = -1
            alt = model.predict(tmp)[0,0]
            diff = (base - alt) * factor
            diffs[y] = diff
            abs_diffs[y] = np.abs(diff)
        res.append(np.mean(diffs))
        abs_res.append(np.mean(abs_diffs))

    with open(out_path, 'w') as output:
        output.write('block,delta,abs_delta\n')
        for i, r, ar in zip(list(idx), res, abs_res):
            output.write("{0},{1},{2}\n".format(i, r, ar))
    
if __name__ == "__main__":
    main_path = Path("/d/data/plasmo/training_data/nn_logs")
    data_path = main_path / "curr_data5.npz"
    model_path = main_path / "curr_model5.h5"
    out_path = main_path / "delta_result5.csv"

    model = tf.keras.models.load_model(model_path)
    data_store = np.load(data_path)
    idx = data_store['idx']
    data = data_store['data']

    delta(data, idx, model, out_path)
    print('done')