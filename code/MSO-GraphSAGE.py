import pandas as pd
import numpy as np
import os
from collections import Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定可见的GPU设备编号
import tensorflow as tf
import stellargraph
import random
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.layer.graphsage import GraphSAGEAggregator,MeanAggregator,MaxPoolingAggregator,MeanPoolingAggregator,AttentionalAggregator
from tensorflow import optimizers, losses, metrics
from tensorflow.keras import layers,Model
from sklearn import preprocessing, feature_extraction, model_selection
import matplotlib.pyplot as plt
from tensorflow import keras

'''随机种子设定'''
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
tf.compat.v1.keras.backend.set_session(sess)
tf.keras.backend.clear_session()
tf.keras.backend.set_image_data_format("channels_last")
tf.keras.backend.set_learning_phase(1)

train_edge = pd.read_csv(r'E:\Bigpaper\Landcover\Image2edge15.csv', sep=",", header=None, names=["source", "target"])
train_point = pd.read_csv(r'E:\Bigpaper\Landcover\Image2光谱归一化.csv', sep=",",header=0)
trainpoint_feature_class = train_point.set_index("FID")  # 节点编号索引
trainclassfile = trainpoint_feature_class["gridcode"]  # 构建类别信息表
point_feature_no_class = trainpoint_feature_class.drop(columns = "gridcode")#特征值中删除类别一列
traindataset = stellargraph.StellarGraph({"point": point_feature_no_class}, {"edge": train_edge})  # 构建stellergraph
print(set(trainclassfile))
print(traindataset.info())

train_subjects, test_subjects = model_selection.train_test_split(trainclassfile,train_size=200, random_state=42,stratify=trainclassfile)
val_subjects, test_subjects = model_selection.train_test_split(test_subjects, train_size=200, random_state=42,stratify=test_subjects)
# train_subjects= pd.read_csv(r'E:\Bigpaper\Landcover\data1train.csv', index_col='FID')
# val_subjects = pd.read_csv(r'E:\Bigpaper\Landcover\data1val.csv', index_col='FID')
# test_subjects = pd.read_csv(r'E:\Bigpaper\Landcover\data1test.csv', index_col='FID')
print(Counter(train_subjects))
# print(Counter(val_subjects))
# train_subjects.to_csv(r"E:\Bigpaper\Landcover\data2train.csv",index=True)
# val_subjects.to_csv(r"E:\Bigpaper\Landcover\data2val.csv",index=True)
# test_subjects.to_csv(r"F:\DGL\origin data\data2test.csv",index=True)
target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)
batch_size = 64
num_samples = [15,10]

generator = GraphSAGENodeGenerator(traindataset, batch_size, num_samples,seed=42,weighted=False)
train_gen = generator.flow(train_subjects.index, train_targets,shuffle=False,seed=42)
graphsage_model = GraphSAGE(layer_sizes=[16,16], generator=generator, aggregator=MeanAggregator,bias=True, dropout=0, normalize="l2",activations=['relu','relu'])
x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1],activation="softmax")(x_out)

# class LossAccHistory(Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#         self.acc = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.acc.append(logs.get('acc'))
#         with open(r'F:\xiaolunwen\exp\autograph\data1\loss_acc.csv', mode='a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, logs.get('loss'),logs.get('acc')])
class F1Score(metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.fp = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.fn = tf.Variable(0, dtype=tf.int32, trainable=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        tp = tf.cast(tf.math.count_nonzero(y_true * y_pred), dtype=tf.int32)
        fp = tf.cast(tf.math.count_nonzero(y_pred - y_true), dtype=tf.int32)
        fn = tf.cast(tf.math.count_nonzero(y_true - y_pred), dtype=tf.int32)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = tf.cast(self.tp, tf.float32) / tf.cast(self.tp + self.fp, tf.float32)
        recall = tf.cast(self.tp, tf.float32) / tf.cast(self.tp + self.fn, tf.float32)
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)



model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(lr=0.005,),
    loss=losses.categorical_crossentropy,metrics=["acc",tf.keras.metrics.Precision(),F1Score()],
)

val_gen = generator.flow(val_subjects.index, val_targets,shuffle=False,seed=42)
# history2 = LossAccHistory()
test_gen = generator.flow(test_subjects.index, test_targets,shuffle=False,seed=42)
history = model.fit(train_gen, epochs=250, validation_data=val_gen, verbose=2,shuffle=False)
supervised_weights = model.get_weights()
a = stellargraph.utils.plot_history(history)
plt.show()
# model.summary()
# plot_model(model, to_file='F:\DGI\jiandu-graphsage_model.png', show_shapes=True)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
test_nodes = test_subjects.index
test_mapper = generator.flow(test_nodes)
test_predictions = model.predict(test_mapper)
test_nodes_predictions =  target_encoding.inverse_transform(test_predictions.squeeze())
df_test = pd.DataFrame({"Predicted": test_nodes_predictions, "ID": test_subjects.index})
outputpath=r'E:\Bigpaper\Landcover\消融实验\data2-GraphSAGE.csv'
df_test.to_csv(outputpath,sep=',',index=True,header=True)

# all_nodes = trainclassfile.index
# labels_all_nodes = trainclassfile[all_nodes]
# hold_out_targets = target_encoding.transform(labels_all_nodes)
# all_mapper = generator.flow(all_nodes,hold_out_targets)
# all_predictions = model.predict(all_mapper)
# node_predictions = target_encoding.inverse_transform(all_predictions)
# results = pd.Series(node_predictions, index=all_nodes)
# df = pd.DataFrame({"Predicted": results, "True": labels_all_nodes})
# hold_out_metrics = model.evaluate(all_mapper)
# print("\nall_nodes Metrics:")
# for name, val in zip(model.metrics_names, hold_out_metrics):
#     print("\t{}: {:0.4f}".format(name, val))
# outputpath=r'F:\TGRS\data\DATA2\allresult\data2SAGE-jiami15+6-all.csv'
# df.to_csv(outputpath,sep=',',index=True,header=True)

# suembedding_model = keras.Model(inputs=x_inp , outputs=x_out)
# node_ids = trainclassfile.index
# node_gen = GraphSAGENodeGenerator(traindataset, batch_size, num_samples,seed=42,weighted=False).flow(node_ids)
# node_embeddings = suembedding_model.predict(node_gen,verbose=1)
# print(node_embeddings)
# embeddings_df = pd.DataFrame(node_embeddings, index=trainclassfile.index)

# # 保存为CSV文件
# embeddings_df.to_csv(r'F:\TAZ\沈阳市21年POI数据\GraphSAGEemb-64-linear.csv')