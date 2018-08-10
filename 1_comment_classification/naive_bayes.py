from sklearn import metrics
#import nltk

# ~0.87 test AUC (80%:20% split)

''' Voc size = 18844
Num of predictions: 871
0.8589674230128674 '''

pos_file = './data/pos.txt'
neg_file = './data/neg.txt'

def read_train_data(pos_file, neg_file, likel):
    n_pos = 0; n_neg = 0
    lid = 0
    prior_pos = 0; prior_neg = 0
    with open(pos_file) as fp:
        for line in fp:
            if lid%5 != 0:
                ls = line.split(' ')
                for w in ls:
                    w = w.strip()
                    if len(w) < 3: continue
                    likel['pos'][w] = 0
                    likel['neg'][w] = 0
                    n_pos += 1
            lid += 1
    prior_pos = lid
    with open(neg_file) as fp:
        for line in fp:
            if lid%5 != 0:
                ls = line.split(' ')
                for w in ls:
                    w = w.strip()
                    if len(w) < 3: continue
                    likel['pos'][w] = 0
                    likel['neg'][w] = 0
                    n_neg += 1
            lid += 1
    prior_neg = lid - prior_pos

    n = len(likel['pos'])
    #print 'Voc size = {}'.format(n)
    alpha = 2.0
    print 'Alpha = {}'.format(alpha)

    lid = 0
    with open(pos_file) as fp:
        for line in fp:
            if lid%5 != 0:
                ls = line.split(' ')
                for w in ls:
                    w = w.strip()
                    if len(w) < 3: continue
                    likel['pos'][w] += 1
            lid += 1
    for w in likel['pos']:
        likel['pos'][w] = (likel['pos'][w] + alpha) / (n_pos + n*alpha)
    with open(neg_file) as fp:
        for line in fp:
            if lid%5 != 0:
                ls = line.split(' ')
                for w in ls:
                    w = w.strip()
                    if len(w) < 3: continue
                    likel['neg'][w] += 1
            lid += 1
    for w in likel['neg']:
        likel['neg'][w] = (likel['neg'][w] + alpha) / (n_neg + n*alpha)

    return prior_pos, prior_neg


def read_test_data(pos_file, neg_file, likel, prior_pos, prior_neg):
    lid = 0
    gts = []; preds = []; cor = 0.0
    with open(pos_file) as fp:
        for line in fp:
            if lid%5 == 0:
                p = prior_pos / prior_neg
                ls = line.split(' ')
                try:
                    for w in ls:
                        w = w.strip()
                        if len(w) < 3: continue
                        p *= likel['pos'][w] / likel['neg'][w]
                except KeyError:
                    pass
                else:
                    gts.append(1)
                    preds.append(p)
                    if p > 1: cor += 1
            lid += 1
    with open(neg_file) as fp:
        for line in fp:
            if lid%5 == 0:
                p = prior_pos / prior_neg
                ls = line.split(' ')
                try:
                    for w in ls:
                        w = w.strip()
                        if len(w) < 3: continue
                        p *= likel['pos'][w] / likel['neg'][w]
                except KeyError:
                    pass
                else:
                    gts.append(-1)
                    preds.append(p)
                    if p < 1: cor += 1
            lid += 1
    fpr, tpr, thresholds = metrics.roc_curve(gts, preds)
    #print 'Num of predictions: {}'.format(len(gts))
    auc = metrics.auc(fpr, tpr)
    print 'accuracy = {}'.format(cor/len(gts))
    print 'auc = {}'.format(auc)


def naive_bayes():
    likel = {'pos': {}, 'neg': {}}
    pri_pos, pri_neg = read_train_data(pos_file, neg_file, likel)
    read_test_data(pos_file, neg_file, likel, pri_pos, pri_neg)

naive_bayes()

