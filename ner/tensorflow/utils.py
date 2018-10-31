# coding=utf-8
import re
import numpy as np
import sys
import time


def calculate(x, y, id2word, id2tag, res=[]):
    entity = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] == 0 or y[i][j] == 0:
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = [id2word[x[i][j]] + '/' + id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
                entity.append(str(i))
                entity.append(str(j))
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def train(model, sess, saver, epochs, batch_size, data_train, data_test, id2word, id2tag):
    batch_num = int(data_train.y.shape[0] / batch_size)
    batch_num_test = int(data_test.y.shape[0] / batch_size)
    for epoch in range(epochs):
        start_time = time.time()
        for batch in range(batch_num):
            x_batch, y_batch = data_train.next_batch(batch_size)
            feed_dict = {model.input_data: x_batch, model.labels: y_batch}
            pre, _ = sess.run([model.viterbi_sequence, model.train_op], feed_dict)
            acc = 0
            if batch % 100 == 0:
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        if y_batch[i][j] == pre[i][j]:
                            acc += 1
                # Print accuracy every 200 batches
                print('Accuracy: {}'.format(float(acc)/(len(y_batch)*len(y_batch[0]))))
        path_name = './model/model{}.ckpt'.format(str(epoch))

        if epoch > 0 and epoch % 3 == 0:
            saver.save(sess, path_name)
            print(path_name + ' model saved.')
            entityres = []
            entityall = []
            for batch_index in range(batch_num):
                x_batch, y_batch = data_train.next_batch(batch_size)
                feed_dict = {model.input_data: x_batch, model.labels: y_batch}
                pre = sess.run([model.viterbi_sequence], feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch, pre, id2word, id2tag, entityres)
                entityall = calculate(x_batch, y_batch, id2word, id2tag, entityall)
                percentage = int(batch_index / batch_num * 100)
                sys.stdout.write("\r%d%%" % percentage)
                sys.stdout.flush()

            intersection = [i for i in entityres if i in entityall]
            if len(intersection) != 0:
                precision = float(len(intersection)) / len(entityres)
                recall = float(len(intersection)) / len(entityall)
                print('Training set precision: {}, recall rate: {}'.format(precision, recall))
                print('F rate: {}'.format((2 * recall * precision) / (precision + recall)))
            else:
                print('Intersection is 0. ')

            entityres = []
            entityall = []
            for batch_index in range(batch_num_test):
                x_batch, y_batch = data_test.next_batch(batch_size)
                feed_dict = {model.input_data: x_batch, model.labels: y_batch}
                pre = sess.run([model.viterbi_sequence], feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch, pre, id2word, id2tag, entityres)
                entityall = calculate(x_batch, y_batch, id2word, id2tag, entityall)
                percentage = int(batch_index / batch_num * 100)
                sys.stdout.write("\r%d%%" % percentage)
                sys.stdout.flush()

            intersection = [i for i in entityres if i in entityall]
            if len(intersection) != 0:
                precision = float(len(intersection)) / len(entityres)
                recall = float(len(intersection)) / len(entityall)
                print('Test set precision: {}, recall rate: {}'.format(precision, recall))
                print('F rate: {}'.format((2 * recall * precision) / (precision + recall)))
            else:
                print('Intersection is 0. ')

        print('====Epoch {} takes {} seconds'.format(epoch, time.time() - start_time))
