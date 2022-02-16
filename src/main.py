import numpy as np
import os
import scipy.sparse as ss
import gc
from sklearn.preprocessing import normalize
DATA_PATH = './Dataset/'
MUSIC_DATA = 'DoubanMusic.txt'


def data_loader():
    # Load data from dataset and return the raw matrix
    usr_list = []
    music_list = []
    score_list = []
    mod_score_list = []

    with open(DATA_PATH+MUSIC_DATA, 'r', encoding='UTF-8') as f:
        line_list = f.readlines()
    for line in line_list:
        rec = line.split('\t')
        for i in range(1, len(rec)):
            usr_list.append(int(rec[0]))
            music_list.append(int(rec[i].split(',')[0]))
            score_list.append(int(rec[i].split(',')[1]))
            if int(rec[i].split(',')[1]) == -1:
                mod_score_list.append(0.5)
            else:
                mod_score_list.append(int(rec[i].split(',')[1]))

    rows = np.array(usr_list)
    cols = np.array(music_list)
    v = np.array(score_list)
    mod_v = np.array(mod_score_list)

    sparse_m = ss.coo_matrix((v, (rows, cols)))
    mod_sparse_m = ss.coo_matrix((mod_v, (rows, cols)))
    ss.save_npz('raw', sparse_m)
    ss.save_npz('mod_raw', mod_sparse_m)


def compute_avrg():
    raw_sparse_matrix = ss.load_npz('raw.npz')
    raw_matrix = np.array(raw_sparse_matrix.todense())
    sum_score = 0
    score_list = []
    item_avrg = []

    for i in range(0, raw_matrix.shape[1]):
        for j in range(0, len(raw_matrix[..., i])):
            s = raw_matrix[j][i]
            if s > 0:
                score_list.append((j, s))
                sum_score += s
        if sum_score != 0:
            avrg = sum_score/len(score_list)
            item_avrg.append(avrg)
        else:
            item_avrg.append(0)
        print(str(i)+':'+str(avrg))
        sum_score = 0
        score_list = []

    print(len(score_list))
    os.system('pause')
    item_avrg = np.array(item_avrg)
    np.save('item_avrg', item_avrg)
    print('item_avrg saved')


def normalized(raw_matrix):
    score_list = []  # for normal scores
    rows = []
    cols = []
    v = []
    sum_score = 0

    for i in range(0, raw_matrix.shape[1]):
        print(i)
        for j in range(0, len(raw_matrix[..., i])):
            s = raw_matrix[j][i]
            if s > 0:
                score_list.append((j, s))
                sum_score += s
        if sum_score != 0:
            avrg = sum_score/len(score_list)

        for (j, s) in score_list:
            rows.append(j)
            cols.append(i)
            v.append(s-avrg)

        score_list = []
        sum_score = 0

    print('count:' + str(len(v)))
    rows = np.array(rows)
    cols = np.array(cols)
    v = np.array(v)
    sparse_m = ss.coo_matrix((v, (rows, cols)))
    full_m = sparse_m.todense()
    print(full_m.shape)
    ss.save_npz('normalized', sparse_m)
    return np.array(full_m)


def compute_item_sim(normalized_matrix):
    print('start here:')
    normalized_one_m = normalize(normalized_matrix, axis=0)
    print('normalized!')

    sim_m = normalized_one_m.T.dot(normalized_one_m)
    print('complete!')
    print(sim_m.shape)

    np.save('item_sim', sim_m)
    print('sim_matrix saved!')
    return sim_m


def compute_pred():
    sim_m = np.load('item_sim.npy')
    norm_sim_m = normalize(sim_m, axis=0, norm='l1')
    print('normed')
    del sim_m
    gc.collect()

    raw_sparse_matrix = ss.load_npz('mod_raw.npz')
    raw_m = np.array(raw_sparse_matrix.todense())
    pred_m = raw_m.dot(norm_sim_m)
    np.save('pred_m', pred_m)

    pred_m = np.load('pred_m.npy')
    print('SHAPE:')
    print(pred_m.shape)
    os.system('pause')

    print(pred_m)
    count = 0
    for i, j in zip(raw_sparse_matrix.row, raw_sparse_matrix.col):
        print((i, j, count, pred_m[i][j]))
        pred_m[i][j] = 0
        count += 1

    np.save('pred_m', pred_m)
    print('done!')
    return pred_m


def predict():
    pred_m = np.load('pred_m.npy')
    print('entered')
    sorted_index = np.argsort(-pred_m, axis=1)
    del pred_m
    gc.collect()
    print('sorted')

    out = sorted_index[..., 0:100]
    with open('answer2.txt', 'w', encoding='UTF-8') as f:
        i = 0
        for line in out:
            print(i)
            f.write(str(i)+'\t')
            for j in range(0, len(line)-1):
                f.write(str(line[j])+',')

            f.write(str(line[len(line)-1])+'\n')
            i += 1


def main():
    k = int(input('input:'))

    if k == 1:  # read the input data and normalize the data matrix
        data_loader()
        os.system('pause')
        normalized()

    elif k == 2:  # calculate the avrg rating for each user and item
        compute_avrg()

    elif k == 3:  # calculate the sim matrix
        normalized_sparse_matrix = ss.load_npz('normalized.npz')
        normalized_matrix = normalized_sparse_matrix.todense()
        compute_item_sim(np.array(normalized_matrix))

    elif k == 4:  # calculate the pred matrix
        compute_pred()

    elif k == 5:
        predict()  # give the answer


if __name__ == '__main__':
    main()
