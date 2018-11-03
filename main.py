import argparse
import os
import pickle as pkl
import numpy as np
import sys
import model
import tensorflow as tf

EOS = "<EOS>"
UNK = "<UNK>"
EOS_ID = 0
UNK_ID = 1

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imdb", help="dataset")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--vocab_size", type=int, default=5000, help="size of predefined vocabulary")
parser.add_argument("--max_seqlen", type=int, default=400, help="maximum sequence length")
parser.add_argument("--num_units", type=int, default=300, help="num of units")
parser.add_argument("--num_hidden", type=int, default=500, help="hidden units of inference network")
parser.add_argument("--dim_emb", type=int, default=300, help="dimension of embedding")
parser.add_argument("--num_topics", type=int, default=200, help="number of topics")
parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--init_from", type=str, default=None, help="init_from")
parser.add_argument("--save_dir", type=str, default="results", help="dir for saving the model")

def load_dataset(params):
  dirnames = {"imdb": "data/imdb/imdb20000/",
      #"agnews": "data/agnews/ag32000",
      "google-extraction": "data/google-extraction/proc/0.1/",
      "casetype": "data/case/proc"
      }

  dirname = dirnames[params.dataset]
  with open(os.path.join(dirname, "vocab.pkl"), "rb") as f:
    vocab = pkl.load(f)
    vocab[EOS] = EOS_ID
    vocab[UNK] = UNK_ID
  
  with open("stop_words.txt", "r") as f:
    stop_words = [line.strip() for line in f.readlines() if line.strip()]
    stop_words.append(UNK)
    stop_words.append(EOS)

  def get_xy(filename):
    with open(os.path.join(dirname, filename), "r") as f:
      lines = [line.strip() for line in f.readlines()]
      samples = [[line.split('\t')[0], line.split('\t')[1].split()] for line in lines]
      y, x = zip(*samples)
      y = list(map(int, y))
      x = [list(map(int, _)) for _ in x]
    return x, y

  labeled_x, labeled_y = get_xy("labeled.data.idx")
  unlabeled_x, unlabeled_y = get_xy("unlabeled.data.idx")
  valid_x, valid_y = get_xy("valid.data.idx")
  test_x, test_y = get_xy("test.data.idx")
  train_x = labeled_x + unlabeled_x
  train_y = labeled_y + [-1] * len(unlabeled_y)

  vocab = {k: vocab[k] for k in vocab if vocab[k] < params.vocab_size}
  stop_words_ids = set([vocab[k] for k in stop_words if k in vocab])
  
  train_x = [[x if x < params.vocab_size else UNK_ID for x in line] for line in train_x]
  valid_x = [[x if x < params.vocab_size else UNK_ID for x in line] for line in valid_x]
  test_x = [[x if x < params.vocab_size else UNK_ID for x in line] for line in test_x]
  
  """ For debugging
  train = train_x[:100], train_y[-100:]
  valid = valid_x[:100], valid_y[:100]
  test = test_x[:100], test_y[:100]
  """
  train = train_x, train_y
  valid = valid_x, valid_y
  test = test_x, test_y

  return train, valid, test, vocab, stop_words_ids

def iterator(data, stop_words_ids, params):
  def batchify():
    x, y = data
    batch_size = params.batch_size
    max_seqlen = params.max_seqlen
    shuffle_idx = np.random.permutation(len(x))
    for i in range(len(x) // batch_size):
      samples = [x[shuffle_idx[j]] for j in range(i*batch_size, i*batch_size + batch_size)]
      labels = [y[shuffle_idx[j]] for j in range(i*batch_size, i*batch_size + batch_size)]
      samples = [sample[:max_seqlen - 1] for sample in samples]
      length = [l + 1 for l in list(map(len, samples))]
      width = max(length)
      indicators = [[1 if token in stop_words_ids else 0 for token in sample] for sample in samples]

      tokens = [[0] + sample + [0] * (width - 1 - len(sample)) for sample in samples]
      targets = [sample + [0] * (width - len(sample)) for sample in samples]
      indicators = [indicator + [0] * (width - len(indicator)) for indicator in indicators]
      
      feature = np.zeros([batch_size, params.vocab_size], dtype='float32')
      for i in range(batch_size):
        for token in samples[i]:
          if token not in stop_words_ids:
            feature[i, token] += 1
      feature = feature / (0.1 + np.sum(feature, axis=1, keepdims=True))
      
      """
      for i in range(params.vocab_size):
        if feature[0, i] != 0:
          print(i, feature[0, i])
      """
      
      output = {"tokens": np.asarray(tokens, dtype='int32'),
          "targets": np.asarray(targets, dtype='int32'),
          "indicators": np.asarray(indicators, dtype='int32'),
          "length": np.asarray(length, dtype='int32'),
          "frequency": feature,
          "label": np.asarray(labels, dtype='int32'),
          }
      """
      for v in output.values():
        print(v.shape)
      """
      yield output
      
  return batchify

def main():
  params = parser.parse_args()
  data_train, data_valid, data_test, vocab, stop_words_ids = load_dataset(params)
  data_train = iterator(data_train, stop_words_ids, params)
  data_valid = iterator(data_valid, stop_words_ids, params)
  data_test = iterator(data_test, stop_words_ids, params)
  params.stop_words = np.asarray([1 if i in stop_words_ids else 0 for i in range(params.vocab_size)])

  os.system("cp -r *.py " + params.save_dir)

  #for x in data_train():
    #print(x)
  
  configproto = tf.ConfigProto()
  configproto.gpu_options.allow_growth = True
  configproto.allow_soft_placement = True
  with tf.Session(config=configproto) as sess:
    train = model.Train(vars(params))
    train.build_graph()

    if params.init_from:
      train.saver.restore(sess, params.init_from)
      print('Model restored from {0}'.format(params.init_from))
    else:
      tf.global_variables_initializer().run()

    train.run(sess, (data_train, data_valid, data_test))

if __name__ == "__main__":
  main()
