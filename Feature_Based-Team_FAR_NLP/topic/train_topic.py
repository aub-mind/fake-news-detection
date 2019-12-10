# -*- coding: utf-8 -*-
"""
@author: Fady Baly 
"""

from topic.model import ABLSTM
from topic.utils.fasttext import FastVector
from topic.utils.utils_ import read_articles
from topic.utils.model_helper import *
import tensorflow as tf
from sklearn.metrics import f1_score
import gensim
import os


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def confusion_matrix_(true_labels, predicted_labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for true_label, prediction in zip(true_labels, predicted_labels):
        true_labels_indices = [i for i, x in enumerate(true_label) if x == 1]
        predicted_labels_indices = [i for i, x in enumerate(prediction) if x == 1]
        indices_to_check = list(set(true_labels_indices + predicted_labels_indices))
        for i in indices_to_check:
            if true_label[i] == prediction[i] == 1:
                tp += 1
            if prediction[i] == 1 and true_label[i] != prediction[i]:
                fp += 1
            if true_label[i] == prediction[i] == 0:
                tn += 1
            if prediction[i] == 0 and true_label[i] != prediction[i]:
                fn += 1
    return tp, fp, tn, fn


def f1_score_(true_labels, predicted_labels):
    tp, fp, _, fn = confusion_matrix_(true_labels, predicted_labels)
    # get precision and recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # get f1 score for all classes
    f1_all_classes = np.array(2*(precision*recall)/(precision+recall))

    # get macrof1
    macro_f1 = f1_all_classes*100
    return macro_f1


def f1_score_per_class(true_labels, predicted_labels, unique_labels):
    scores = list()
    true_labels = np.vstack(true_labels)
    predicted_labels = np.vstack(predicted_labels)
    for index, label in zip(range(len(true_labels[0])), unique_labels.keys()):
        scores.append((label, f1_score(true_labels[:, index].astype(np.int32), predicted_labels[:, index].astype(np.int32), average='macro')))
    return scores


def main():
    langs = ['eng']
    sequence_length = 100
    for lang in langs:
        for loop in range(10):
            tf.reset_default_graph()
            print('\n\033[1m' + 'reading articles' + '\033[0m')
            # file_path = 'fra_topics.tsv'
            training_file_path = 'topic/training.tsv'
            test_file_path = 'topic/test.tsv'
            x_train, y_train, unique_labels = read_articles(training_file_path, sequence_length=sequence_length, lang=lang)
            x_test, y_test, unique_labels = read_articles(test_file_path, sequence_length=sequence_length, lang=lang)
            x_train_dist = np.load('train_entities_distribution.npy', allow_pickle=True)
            x_test_dist = np.load('test_entities_distribution.npy', allow_pickle=True)
            print('\033[1m' + 'Done!' + '\033[0m\n')

            config = {'max_len': sequence_length, 'hidden_size': 128, 'bilstm_layers': 1, 'embedding_size': 1024, 'n_class': len(unique_labels),
                      'learning_rate': 1e-3, 'batch_size': 64, 'train_epoch': 70, 'hold_prob': 0.8}
            print('number of classes:', len(unique_labels))

            classifier = ABLSTM(config)
            classifier.build_graph()

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())
            if os.path.exists('topic_models/tracker_' + lang + '_' + str(config['bilstm_layers']) + '_' + str(config['hidden_size']) + '.txt'):
                with open('topic_models/tracker_' + lang + '_' + str(config['bilstm_layers']) + '_' + str(config['hidden_size']) + '.txt', 'r') as reader:
                    for i in reader:
                        best_model = float(i)
            else:
                best_model = 0
            print('current best model:', best_model)
            for epoch in range(config["train_epoch"]):
                true_labels = list()
                predicted_labels = list()
                print("Epoch %d start !" % (epoch + 1))
                for x_batch, y_batch, dist_batch in fill_feed_dict(x_train, y_train, x_train_dist, config["batch_size"]):
                    loss, prediction_sigmoid_logits, logits = run_train_step(classifier, sess, (x_batch, y_batch, dist_batch), keep_prob=config['hold_prob'])
                    true_labels = true_labels + [list(i) for i in y_batch]
                    predicted_labels = predicted_labels + [list(np.round(j)) for j in prediction_sigmoid_logits]
                score_per_class = f1_score_per_class(true_labels, predicted_labels, unique_labels)
                print('train results:')
                f1_all_classes = list()
                for class_, score_ in score_per_class:
                    f1_all_classes.append(score_)
                    print('\t\t' + class_ + ' %0.2f' % (score_*100))
                print('\tloss: %0.2f' % loss, 'f1 score: %0.2f' % (np.mean(f1_all_classes)*100))

                true_labels = list()
                predicted_labels = list()
                for x_batch, y_batch, dist_batch in fill_feed_dict(x_test, y_test, x_test_dist, config["batch_size"]):
                    prediction_sigmoid_logits, loss = run_eval_step(classifier, sess, (x_batch, y_batch, dist_batch), keep_prob=config['hold_prob'])
                    true_labels = true_labels + [list(i) for i in y_batch]
                    predicted_labels = predicted_labels + [list(np.round(j)) for j in prediction_sigmoid_logits]
                score_per_class = f1_score_per_class(true_labels, predicted_labels, unique_labels)
                print('test results:')
                f1_all_classes = list()
                for class_, score_ in score_per_class:
                    f1_all_classes.append(score_)
                    print('\t\t' + class_ + ' %0.2f' % (score_*100))
                print('\tloss: %0.2f' % loss, 'f1 score: %0.2f' % (np.mean(f1_all_classes)*100))
                print('cross validation #' + str(loop+1))

                if np.mean(f1_all_classes) > best_model:
                    print('saving best model!')
                    saver.save(sess, 'topic_models/' + lang + '_' + str(config['bilstm_layers']) + '_' + str(config['hidden_size']) + '_model/model.ckpt')
                    best_model = np.mean(f1_all_classes)
                    with open('topic_models/tracker_' + lang + '_' + str(config['bilstm_layers']) + '_' + str(config['hidden_size']) + '.txt', 'w') as writer:
                        writer.write(str(best_model))


if __name__ == '__main__':
    main()
