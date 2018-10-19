import tensorflow as tf
import os
import numpy as np
import pickle as pkl

class TopicRNN(object):
  def __init__(self, num_units, dim_emb, vocab_size, num_topics, num_hidden, num_layers):
    self.num_units = num_units
    self.dim_emb = dim_emb
    self.num_topics = num_topics
    self.num_hidden = num_hidden
    self.num_layers = num_layers
    self.vocab_size = vocab_size

  def forward(self, inputs, mode="Train"):
    # build inference network
    infer_logits = tf.layers.dense(inputs["frequency"], units=self.num_hidden, activation=tf.nn.relu)
    infer_mean = tf.layers.dense(infer_logits, units=self.num_topics)
    infer_logvar = tf.layers.dense(infer_logits, units=self.num_topics)
    pst_dist = tf.distributions.Normal(loc=infer_mean, scale=infer_logvar * infer_logvar)
    pri_dist = tf.distributions.Normal(loc=tf.zeros_like(infer_mean), scale=tf.ones_like(infer_logvar))
    theta = pst_dist.sample()
    
    # build generative model
    self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.dim_emb], dtype=tf.float32)
    emb = tf.nn.embedding_lookup(self.embedding, inputs["tokens"])
    cells = [tf.nn.rnn_cell.LSTMCell(self.num_units) for _ in range(self.num_layers)]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    rnn_outputs, final_output = tf.nn.dynamic_rnn(cell, inputs=emb, sequence_length=inputs["length"], dtype=tf.float32)

    token_logits = tf.layers.dense(rnn_outputs, units=self.vocab_size) + \
        tf.layers.dense(tf.expand_dims(theta, 1), units=self.vocab_size) * \
        tf.to_float(tf.expand_dims(1 - inputs["indicators"], 2))

    token_loss = tf.contrib.seq2seq.sequence_loss(logits=token_logits,
        targets=inputs["targets"],
        weights=tf.to_float(tf.sequence_mask(inputs["length"])),
        average_across_timesteps=False,
        average_across_batch=False,
        name="token_loss")
    token_loss = token_loss * tf.to_float(tf.sequence_mask(inputs["length"]))
    token_loss = tf.reduce_mean(tf.reduce_sum(token_loss, axis=1))
    
    indicator_logits = tf.squeeze(tf.layers.dense(rnn_outputs,  units=1), axis=2)
    indicator_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(inputs["indicators"]),
        logits=indicator_logits,
        name="indicator_loss")
    indicator_loss = indicator_loss * tf.to_float(tf.sequence_mask(inputs["length"]))
    indicator_loss = tf.reduce_mean(tf.reduce_sum(indicator_loss, axis=1))

    kl_loss = tf.contrib.distributions.kl_divergence(pst_dist, pri_dist)
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1), axis=0)

    total_loss = token_loss + indicator_loss + kl_loss

    tf.summary.scalar(tensor=token_loss, name="token_loss")
    tf.summary.scalar(tensor=indicator_loss, name="indicator_loss")
    tf.summary.scalar(tensor=kl_loss, name="kl_loss")
    tf.summary.scalar(tensor=total_loss, name="total_loss")

    outputs = {
        "token_loss": token_loss,
        "indicator_loss": indicator_loss,
        "kl_loss": kl_loss,
        "loss": total_loss,
        "theta": theta,
        "repre": final_output[-1][1],
        }


    return outputs

class Train(object):
  def __init__(self, params):
    self.params = params
  
  def _create_placeholder(self):
    self.inputs = {
        "tokens": tf.placeholder(tf.int32, shape=[None, None], name="tokens"),
        "indicators": tf.placeholder(tf.int32, shape=[None, None], name="indicators"),
        "length": tf.placeholder(tf.int32, shape=[None], name="length"),
        "frequency": tf.placeholder(tf.float32, shape=[None, self.params["vocab_size"]], name="frequency"),
        "targets": tf.placeholder(tf.int32, shape=[None, None], name="targets"),
        }

  def build_graph(self):
    self._create_placeholder()
    with tf.device('/cpu:0'):
      self.global_step = tf.get_variable('global_step', [],
          initializer=tf.constant_initializer(0), trainable=False)

    model = TopicRNN(num_units = self.params["num_units"],
        dim_emb = self.params["dim_emb"],
        vocab_size = self.params["vocab_size"],
        num_topics = self.params["num_topics"],
        num_layers = self.params["num_layers"],
        num_hidden = self.params["num_hidden"],
        )

    # train output
    with tf.variable_scope('topicrnn'):
      self.outputs_train = model.forward(self.inputs, mode="Train")

    with tf.variable_scope('topicrnn', reuse=True):
      self.outputs_test  = model.forward(self.inputs, mode="Test")

    self.summary = tf.summary.merge_all()
    self.train_op  = tf.train.AdamOptimizer(learning_rate=self.params["learning_rate"]).minimize(self.outputs_train["loss"], global_step=self.global_step)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

  def batch_train(self, sess, inputs):
    keys = list(self.outputs_train.keys())
    outputs = [self.outputs_train[key] for key in keys]
    outputs = sess.run([self.train_op, self.summary] + outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys()})
    ret = {keys[i]: outputs[i+2] for i in range(len(keys))}
    ret["summary"] = outputs[1]
    return ret

  def batch_test(self, sess, inputs):
    keys = list(self.outputs_test.keys())
    outputs = [self.outputs_test[key] for key in keys]
    outputs = sess.run(outputs, feed_dict={self.inputs[k]: inputs[k] for k in self.inputs.keys()})
    return {keys[i]: outputs[i] for i in range(len(keys))}

  def run_epoch(self, sess, datasets):
    train_loss, valid_loss, test_loss = [], [], []
    train_theta, valid_theta, test_theta = [], [], []
    train_repre, valid_repre, test_repre = [], [], []
    train_label, valid_label, test_label = [], [], []

    dataset_train, dataset_dev, dataset_test = datasets
    for batch in dataset_train():
      train_outputs = self.batch_train(sess, batch)
      train_loss.append(train_outputs["loss"])
      train_theta.append(train_outputs["theta"])
      train_repre.append(train_outputs["repre"])
      train_label.append(batch["label"])
      self.writer.add_summary(train_outputs["summary"])
      #print(train_outputs)
      
    for batch in dataset_dev():
      valid_outputs = self.batch_test(sess, batch)
      valid_loss.append(valid_outputs["loss"])
      valid_theta.append(valid_outputs["theta"])
      valid_repre.append(valid_outputs["repre"])
      valid_label.append(batch["label"])
      #print(valid_outputs)

    for batch in dataset_test():
      test_outputs = self.batch_test(sess, batch)
      test_loss.append(test_outputs["loss"])
      test_theta.append(test_outputs["theta"])
      test_repre.append(test_outputs["repre"])
      test_label.append(batch["label"])
      #print(test_outputs)

    train_loss = np.mean(train_loss)
    valid_loss = np.mean(valid_loss)
    test_loss = np.mean(test_loss)

    train_theta, valid_theta, test_theta = np.vstack(train_theta), np.vstack(valid_theta), np.vstack(test_theta)
    train_repre, valid_repre, test_repre = np.vstack(train_repre), np.vstack(valid_repre), np.vstack(test_repre)
    train_label, valid_label, test_label = np.vstack(train_label), np.vstack(valid_label), np.vstack(test_label)

    train_res = [train_loss, train_theta, train_repre, train_label]
    valid_res = [valid_loss, valid_theta, valid_repre, valid_label]
    test_res = [test_loss, test_theta, test_repre, test_label]

    print("train_loss: {:.4f}, valid_loss: {:.4f}, test_loss: {:.4f}".format(train_loss, valid_loss, test_loss))

    return train_res, valid_res, test_res

  def run(self, sess, datasets):
    best_valid_loss = 1e10
    self.writer = tf.summary.FileWriter(os.path.join(self.params["save_dir"], "train"), sess.graph)
    for i in range(self.params["num_epochs"]):
      train_res, valid_res, test_res = self.run_epoch(sess, datasets)
      if best_valid_loss > valid_res[0]:
        print("Best model found at epoch {}".format(i))
        best_valid_loss = valid_res[0]
        with open(os.path.join(self.params["save_dir"], "results.pkl"), "wb") as f:
          pkl.dump([train_res, valid_res, test_res], f)
        self.saver.save(sess, os.path.join(self.params["save_dir"], "model"), global_step = i)
