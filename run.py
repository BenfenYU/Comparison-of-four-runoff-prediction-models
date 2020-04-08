# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os,pickle,datetime,copy
from sklearn import metrics,svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,Activation,LSTM,GRU,RNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR,'data')
MODELDIR = os.path.join(BASEDIR,'models')
LSTMDIR_SINGLE = os.path.join(MODELDIR,'LSTM_single')
LSTMDIR_MULTI = os.path.join(MODELDIR,'LSTM_multi')
GRUDIR_SINGLE = os.path.join(MODELDIR,'GRU_single')
GRUDIR_MULTI = os.path.join(MODELDIR,'GRU_multi')
MLPDIR = os.path.join(MODELDIR,'MLP')
SVRDIR = os.path.join(MODELDIR,'SVR')
MLPDIR_MULTI = os.path.join(MODELDIR,'MLP_multi')
SVRDIR_MULTI = os.path.join(MODELDIR,'SVR_multi')

TRAIN_SPLIT = 0.6
TEST_SPLIT = 0.2
BUFFER_SIZE = 1000
BATCHSIZE = 128
EPOCHS =  50
STEP = 1
EVALUATION_INTERVAL = 200
FUTURE_TARGET = 6
RATE = 0.2
RUNOFF = 0
RAINFALL = 1

######################### 画画区 ################################

def plot_train_history(history, title,path):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  #plt.show()
  plt.savefig(fname = path,quality = 95, format='jpg')

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 0]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

def plot_origin_data():
  
  dataset = loadData()

  sha = dataset.shape[-1]
  leng = dataset.shape[0]
  x1 = np.linspace(0, leng,num=leng)
  x2 = np.linspace(0, leng,num = leng)

  y1 = dataset[:,0]
  y2 = dataset[:,1]

  plt.subplot(2, 1, 1)
  plt.plot(x1, y1, 'o-')
  plt.title('Stream and rain')
  plt.ylabel('stream')

  plt.subplot(2, 1, 2)
  plt.plot(x2, y2, '.-')
  plt.xlabel('time (days)')
  plt.ylabel('rain')

  plt.show()

  return

def plot_single_pre(model_dir,past_history = 15,net = 'LSTM',c = 'r'):

  dataset = read_data()
  number = int(dataset.shape[0] * TRAIN_SPLIT)
  n2 = int(dataset.shape[0] * (TRAIN_SPLIT + TEST_SPLIT))
  _,_,_,_,x_test, y_test = trans_data(past_history,future_target = 1)

  model_dir = os.path.join(model_dir,'step = {}'.format(str(past_history))) 
  # 这里要手动compile网络，否则，tf找不到自定义的nse，即使用custom_opjects参数也不成
  model = tf.keras.models.load_model(model_dir,compile = False)  
  metric = ['mae','mape',nse]
  model.compile(optimizer=RMSprop(), loss='mse',\
      metrics=metric)
  y_pre = model.predict(x_test)

  x_l = np.array(y_test).reshape(-1,1)
  y_l = np.array(y_pre).reshape(-1,1)
  linreg = LinearRegression()
  linreg.fit(x_l, y_l)
  y_l_pre = linreg.predict(x_l)

  plt.scatter(y_pre, y_test,s = 2,c =c)
  plt.plot(x_l,y_l_pre,label = net,c = c)

  plt.xlim((0,1))
  plt.ylim((0,1))
  plt.xlabel('Forecasted Daily Runoff')
  plt.ylabel('Oberved Daily Runoff')
  #plt.title("Prediction And Really Data(step = {})".format(str(past_history)))
  plt.legend()

  #plt.show()
  #plt.savefig(os.path.join(BASEDIR,'step_{}'.format(str(past_history))))

def plot_multi_pre(model_dir,past_history = 15,net = 'LSTM',c = 'k'):

  dataset = read_data()
  number = int(dataset.shape[0] * TRAIN_SPLIT)
  n2 = int(dataset.shape[0] * (TRAIN_SPLIT + TEST_SPLIT))
  _,_,_,_,x_test, y_test = trans_data(past_history,future_target = FUTURE_TARGET)

  model_dir = os.path.join(model_dir,"step = {},pre = {}".format(str(past_history),str(FUTURE_TARGET))) 
  # 这里要手动compile网络，否则，tf找不到自定义的nse，即使用custom_opjects参数也不成
  model = tf.keras.models.load_model(model_dir,compile = False)  
  metric = ['mae','mape',nse]
  model.compile(optimizer=RMSprop(), loss='mse',\
      metrics=metric)
  y_pre = model.predict(x_test)

  length = y_test.shape[0]
  l_pre = y_test.shape[1]
  x = []
  y = []
  for i in range(length):
    for j in range(l_pre):
      x.append(y_test[i][j])
      y.append(y_pre[i][j])

  x_l = np.array(x).reshape(-1,1)
  y_l = np.array(y).reshape(-1,1)
  linreg = LinearRegression()
  linreg.fit(x_l, y_l)
  y_l_pre = linreg.predict(x_l)

  plt.scatter(x, y,s = 0.5,c = c)
  plt.plot(x,y_l_pre,label = net,c =c )

  plt.xlim((0,1))
  plt.ylim((0,1))
  plt.xlabel('Forecasted Daily Runoff')
  plt.ylabel('Oberved Daily Runoff')

  #plt.title("Prediction And Really Data(step = {})".format(str(past_history)))
  plt.legend()

  #plt.show()
  #plt.savefig(os.path.join(BASEDIR,'step_{}'.format(str(past_history))))

def plot_one_method(y_test,y_pre,label):  

  maxv = max(np.max(y_pre), np.max(y_test))  
  x = np.array([0,maxv]).reshape(-1,1)

  linreg = LinearRegression()
  linreg.fit(y_pre, y_test)
  y_l_pre = linreg.predict(x)

  plt.scatter(y_pre, y_test,s = 3,alpha=1,label = label)
  plt.plot(x,y_l_pre,label = label)
  plt.plot(x,x,c = 'k')

  plt.xlim((0,maxv))
  plt.ylim((0,maxv))
  plt.xlabel('Forecasted Daily Runoff')
  plt.ylabel('Oberved Daily Runoff')
  #plt.show()

def plot_methods():
  net = Net()
  net.pre_mlp()
  net.pre_svr()
  net.pre_lstm()
  net.pre_gru()
  plt.legend()
  plt.plot([0,1],[0,1],c = 'k',label = 'Diagonal')
  title = "Station A"
  plt.title(title)
  plt.show()

######################### 数据区 ################################

class DataProcess():
  def __init__(self,attri = True):
    self.min_max_scaler = MinMaxScaler()
    self.attri = attri
    # 数据第一列是径流，第二列是降雨
    self.dataset = self.read_data()

  def multivariate_data(self,dataset, target, start_index, end_index, history_size,
                        target_size, step = 1, single_step=False):
    '''
    dataset是numpy数组
    target是以哪一列的数据作为预测的对象
    start_index是从数组什么位置开始
    end_index是到数组什么位置为止
    history_size是用过去多少个数据来预测后面
    target_size是指用过去的数据来预测后面的多少位数据，如：1,2,3,4,5...
              当target_size是2,即用1,2,3来预测4,5
    step是指每隔STEP个数据取一个过去的数据，用来预测，如：1,2,3,4,5,...
        当STEP取2,则是用1,3,5...来做预测
    single_step是指是否仅仅预测单独一次的值

    总之，此函数应当说是 univariate_data(...)的超集
    '''
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
      end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
      indices = range(i-history_size, i, step)
      data.append(dataset[indices])

      if single_step:
        labels.append(target[i+target_size])
      else:
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

  # 生成网络时间序列数据的
  def univariate_data(self,dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
      end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
      indices = range(i-history_size, i)
      # Reshape data from (history_size,) to (history_size, 1)
      data.append(np.reshape(dataset[indices], (history_size, 1)))
      labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

  def read_data(self):
      pkl_file = open(os.path.join(BASEDIR,'data.pickle'), 'rb')
      data = pickle.load(pkl_file)
      dataset = np.array(data)
      pkl_file.close()
      
      dataset[:,0] = self.min_max_scaler.fit_transform\
                      (dataset[:,0].reshape(-1,1)).reshape(dataset.shape[0])
      dataset[:,1] = MinMaxScaler().fit_transform\
                      (dataset[:,1].reshape(-1,1)).reshape(dataset.shape[0])

      return dataset

  def trans_data(self,past_history,future_target,time_dim = True,all_data = False):
    dataset = self.dataset
    if not self.attri:
      dataset = dataset[:,0]

    length = dataset.shape[0]
    number = int(length * TRAIN_SPLIT)
    n2 = int(length * (1- TEST_SPLIT))

    x_all,y_all = self.multivariate_data(dataset, dataset[:, 0], 0,
                                    None, past_history,
                                    target_size = future_target)
    
    if(not time_dim):
      length = x_all.shape[0]
      sketch_num = past_history * x_all.shape[-1]
      x_all.resize((length,sketch_num),refcheck=False)

    if (not all_data):
      # 训练集
      x_train_single, y_train_single = x_all[0:number],y_all[0:number]
      # 验证集
      #x_val_single, y_val_single = x_all[number:n2],y_all[number:n2]
      # 测试集
      x_test, y_test = x_all[number:][0:1000],y_all[number:][0:1000]

      return (x_train_single, y_train_single,x_test, y_test)
    else :
      return (x_all,y_all)

  def inverse_trans(self,data):
    return self.min_max_scaler.inverse_transform(data)

######################### 网络区 ################################

#def nse(y_true,y_pred):
# return 1-K.sum(K.square(y_pred-y_true))/K.sum(K.square(y_true-K.mean(y_true)))

def nse(y_true,y_pred):
  return 1-np.sum(np.square(y_pred-y_true))/np.sum(np.square(y_true-np.mean(y_true)))


def rmse(y_true,y_pred):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  n = y_pred.shape[0]

  rmse = np.sqrt(np.sum(np.square( y_pred-y_true))/n)

  return rmse

# mean absolute relative error
def mare(y_true,y_pred):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  n = y_pred.shape[0]

  rmse = np.sum(np.abs((y_true - y_pred)/y_true)) * 100 /n

  return rmse

class Net():

  def __init__(self,past_history = 15,future_target = FUTURE_TARGET):
    self.data_p = DataProcess()
    # 确定了（优化器，损失函数，metrics）
    self.metric = ['mae','mape']#,nse]
    self.past_history = past_history
    self.future_target = future_target

  def eval(self,y_true,y_pre):

    rms_e = rmse(y_true,y_pre)
    mar_e = mare(y_true,y_pre)
    ns_e = nse(y_true,y_pre)

    return (rms_e,mar_e,ns_e)

  def pre_gru_multi(self,p):
    _,_,x_test, y_test = self.data_p.trans_data(
                                            self.past_history,
                                            future_target = FUTURE_TARGET,
                                            time_dim=True,
                                            all_data=False)
    model = self.load_model_multi(GRUDIR_MULTI,FUTURE_TARGET)
    y_pre = model.predict(x_test)

    
    y_pre = y_pre[:,p-1].reshape(-1,1)
    y_test = y_test[:,p-1].reshape(-1,1)

    y_test = self.data_p.inverse_trans(y_test)
    y_pre = self.data_p.inverse_trans(y_pre)

    print(self.eval(y_test,y_pre))
    plot_one_method(y_test,y_pre,'GRU')

  def pre_lstm_multi(self,p):
    _,_,x_test, y_test = self.data_p.trans_data(
                                            self.past_history,
                                            future_target = FUTURE_TARGET,
                                            time_dim=True,
                                            all_data=False)
    model = self.load_model_multi(LSTMDIR_MULTI,FUTURE_TARGET)
    y_pre = model.predict(x_test)

    y_pre = y_pre[:,p-1].reshape(-1,1)
    y_test = y_test[:,p-1].reshape(-1,1)

    y_test = self.data_p.inverse_trans(y_test)
    y_pre = self.data_p.inverse_trans(y_pre)

    print(self.eval(y_test,y_pre))
    plot_one_method(y_test,y_pre,'LSTM')

  def pre_mlp_multi(self,p):
    _,_,x_test, y_test = self.data_p.trans_data(
                                            self.past_history,
                                            future_target = FUTURE_TARGET,
                                            time_dim=False,
                                            all_data=False)
    model = self.load_model_multi(MLPDIR,FUTURE_TARGET)
    y_pre = model.predict(x_test)

    y_pre = y_pre[:,p-1].reshape(-1,1)
    y_test = y_test[:,p-1].reshape(-1,1)

    y_test = self.data_p.inverse_trans(y_test)
    y_pre = self.data_p.inverse_trans(y_pre)

    print(self.eval(y_test,y_pre))
    plot_one_method(y_test,y_pre,'MLP')

  def pre_svr_multi(self,p):    
    if(p != 1):
      fdir =  'step = {},pre = {}'.format(str(self.past_history),str(p))
    else:
      fdir =  'step = {}'.format(str(self.past_history))

    with open(os.path.join(SVRDIR,fdir),'rb') as f:
      clf = pickle.load(f)

      _,_,x_test,y_test = self.data_p.trans_data(
                                            self.past_history,
                                            future_target = p,
                                            all_data= False,
                                            time_dim=False)
    y_pre = clf.predict(x_test)

    y_pre = y_pre.reshape(-1,1)
    y_test = y_test[:,p-1].reshape(-1,1)

    y_test = self.data_p.inverse_trans(y_test)
    y_pre = self.data_p.inverse_trans(y_pre)

    print(self.eval(y_test,y_pre))
    plot_one_method(y_test,y_pre,'SVR')

  def pre_use_pre(self,target):

    if (target > self.past_history):
      print("预测天数超过历史天数！")

    _,_,_,_,x_test, y_test = self.data_p.trans_data(
                                                  self.past_history,
                                                  future_target = 1)

    model_dir = os.path.join(GRUDIR_SINGLE,'step = {}'.format(str(self.past_history))) 
    # 这里要手动compile网络，否则，tf找不到自定义的nse，即使用custom_opjects参数也不成
    model = tf.keras.models.load_model(model_dir,compile = False)  
    metric = ['mae','mape',nse]
    model.compile(optimizer=RMSprop(), loss='mse',\
        metrics=metric)

    # 降雨是从每一组的最后一天天开始的，即索引中的-1
    rain = x_test[1:,-1,RAINFALL].copy()
    # x要舍去最后的target个值，因为做预测要用到后面的来验证效果
    x_test = x_test[:x_test.shape[0]-target]
    target_pre = []
    for i in range(target):
      y_pre = model.predict(x_test)
      target_pre.append(y_pre)
      for j in range(y_pre.shape[0]):
        # 要加入的降雨为下一天的新值
        new_rainfall = rain[j+i]
        # 要加入的径流是刚预测出的径流数据
        new_runoff = y_pre[j].tolist()[0]
        in_arr = [[new_rainfall,new_runoff]]
        change_arr = x_test[j][1:]
        x_test[j] = np.append(change_arr,in_arr,axis = 0)
    
      x_l = target_pre[i]
      y_l = y_test[i:y_test.shape[0]-target + i]
      linreg = LinearRegression()
      linreg.fit(x_l, y_l)
      y_l_pre = linreg.predict(x_l)

      plt.scatter( x_l,y_l,s = 2,c ='r')
      plt.plot(x_l,y_l_pre,label = 'lstm',c = 'r')
      plt.plot([0,1],[0,1],c = 'k',label = 'y = x')

      plt.xlim((0,1))
      plt.ylim((0,1))
      plt.xlabel('Forecasted Daily Runoff')
      plt.ylabel('Oberved Daily Runoff')
      #plt.title("Prediction And Really Data(step = {})".format(str(past_history)))
      plt.show()

    return

  def pre_lstm(self):
    x_test, y_test = self.data_p.trans_data(
                                            self.past_history,
                                            future_target = 1,
                                            time_dim=True,
                                            all_data=True)
    model = self.load_model(LSTMDIR_SINGLE)
    y_pre = model.predict(x_test)
    y_test = self.data_p.inverse_trans(y_test)
    y_pre = self.data_p.inverse_trans(y_pre)
    
    print(self.eval(y_test,y_pre))
    
    plot_one_method(y_test,y_pre,label = 'LSTM')

  def pre_gru(self):
    x_test, y_test = self.data_p.trans_data(
                                            self.past_history,
                                            future_target = 1,
                                            time_dim=True,
                                            all_data=True)
    model = self.load_model(GRUDIR_SINGLE)
    y_pre = model.predict(x_test)
    y_test = self.data_p.inverse_trans(y_test)
    y_pre = self.data_p.inverse_trans(y_pre)

    print(self.eval(y_test,y_pre))
    
    #plot_one_method(y_test,y_pre,label = 'GRU')

  def pre_mlp(self):
    x_test, y_test = self.data_p.trans_data(
                                            self.past_history,
                                            future_target = 1,
                                            time_dim=False,
                                            all_data=True)
    model = self.load_model(MLPDIR)
    y_pre = model.predict(x_test)
    y_test = self.data_p.inverse_trans(y_test)
    y_pre = self.data_p.inverse_trans(y_pre)

    print(self.eval(y_test,y_pre))
    
    #plot_one_method(y_test,y_pre,label = 'MLP')

  def pre_svr(self):
    with open(os.path.join(SVRDIR,
        'step = {}'.format(str(self.past_history))),'rb') as f:
      clf = pickle.load(f)

      x_test,y_test = self.data_p.trans_data(
                                            self.past_history,
                                            future_target = 1,
                                            all_data= True,
                                            time_dim=False)

    y_pre = clf.predict(x_test)
    y_test = self.data_p.inverse_trans(y_test.reshape(-1,1))
    y_pre = self.data_p.inverse_trans(y_pre.reshape(-1,1))

    print(self.eval(y_test,y_pre))

    #plot_one_method(y_test,y_pre,label = 'SVR')

  def load_model(self,path):
    model_dir = os.path.join(path,'step = {}'.format(str(self.past_history))) 
    # 这里要手动compile网络，否则，tf找不到自定义的nse，即使用custom_opjects参数也不成
    model = tf.keras.models.load_model(model_dir,compile = False) 
    model.compile(optimizer=RMSprop(), loss='mse',\
        metrics=self.metric)
      
    return model

  def load_model_multi(self,path,p):
    model_dir = os.path.join(path,
    'step = {},pre = {}'.format(str(self.past_history),str(p))) 
    # 这里要手动compile网络，否则，tf找不到自定义的nse，即使用custom_opjects参数也不成
    model = tf.keras.models.load_model(model_dir,compile = False) 
    model.compile(optimizer=RMSprop(), loss='mse',\
        metrics=self.metric)
      
    return model

  # lstm多步
  def train_lstm_multi(self):
    # 训练集
    x_train, y_train ,x_test, y_test = \
                      self.data_p.trans_data(self.past_history,
                                              future_target = FUTURE_TARGET)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCHSIZE).repeat()

    #val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    #val_data = val_data.batch(BATCHSIZE).repeat()


    model = Sequential()
    model.add(LSTM(32,input_shape=x_train.shape[-2:],
                                  return_sequences=True))
    model.add(Dropout(RATE))
    model.add(LSTM(32, return_sequences=True)) 
    model.add(Dropout(RATE))
    model.add(LSTM(32))  
    model.add(Dropout(RATE))
    model.add(Dense(FUTURE_TARGET))
    model.add(Activation('tanh'))

    model.compile(optimizer=RMSprop(), loss='mse',\
      metrics=self.metric)
    
    # 训练
    single_step_history = model.fit(train_data, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL # 一个epoch过完一遍数据
                                                #validation_data=val_data,
                                                #validation_steps=50
                                                )

    # 评估模型                                       
    #results = model.evaluate(x_test, y_test)#, BATCHSIZE=BATCHSIZE)

    # 存储模型
    keras_model_path = os.path.join(LSTMDIR_MULTI,"step = {},pre = {}"\
      .format(str(self.past_history),str(FUTURE_TARGET)))
    model.save(keras_model_path)

    '''
    # 把评估结果写入磁盘
    model_results_path = os.path.join(BASEDIR,'models','results.txt')
    with open(model_results_path, 'a') as f:
      f.write('\n'+str(past_history))
      [f.write(','+str(k)) for k in results]
      title = 'Training and validation loss(step:{})'.format(str(past_history))
      plot_train_history(single_step_history,'Training and validation loss',
                        os.path.join(BASEDIR,'models',title+'.jpg'))
    '''

  # gru多步
  def train_gru_multi(self):
    # 训练集
    x_train, y_train ,x_test, y_test = self.data_p.trans_data(self.past_history,
                                                              future_target = FUTURE_TARGET)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCHSIZE).repeat()

    #val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    #val_data = val_data.batch(BATCHSIZE).repeat()


    model = Sequential()
    model.add(LSTM(32,input_shape=x_train.shape[-2:],
                                  return_sequences=True))
    model.add(Dropout(RATE))
    model.add(GRU(32, return_sequences=True)) 
    model.add(Dropout(RATE))
    model.add(GRU(32))  
    model.add(Dropout(RATE))
    model.add(Dense(FUTURE_TARGET))
    model.add(Activation('tanh'))

    model.compile(optimizer=RMSprop(), loss='mse',\
      metrics=self.metric)
    
    # 训练
    single_step_history = model.fit(train_data, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL
                                                #validation_data=val_data,
                                                #validation_steps=50
                                                )

    # 评估模型                                       
    # results = model.evaluate(x_test, y_test)#, BATCHSIZE=BATCHSIZE)

    # 存储模型
    keras_model_path = os.path.join(GRUDIR_MULTI,"step = {},pre = {}"\
      .format(str(self.past_history),str(FUTURE_TARGET)))
    model.save(keras_model_path)

  # lstm单步
  def train_lstm_single(self):

    # 确定了（优化器，损失函数，metrics）
    metric = ['mae','mape',nse]
    # past_history是不同的timestep,做实验用
    for past_history in lag_list:
      # 训练集
      x_train_single, y_train_single ,x_val_single, y_val_single,x_test, y_test = \
                                                  self.data_p.trans_data(
                                                    past_history,
                                                    future_target = 1,
                                                    )

      train_data = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
      train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCHSIZE).repeat()

      val_data = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
      val_data = val_data.batch(BATCHSIZE).repeat()

      model = Sequential()
      model.add(LSTM(32,input_shape=x_train_single.shape[-2:],
                                    return_sequences=True))
      model.add(Dropout(RATE))
      model.add(LSTM(32, return_sequences=True)) 
      model.add(Dropout(RATE))
      model.add(LSTM(32))  
      model.add(Dropout(RATE))
      model.add(Dense(1))
      model.add(Activation('tanh'))

      model.compile(optimizer=RMSprop(), loss='mse',\
        metrics=metric)
      
      # 训练
      single_step_history = model.fit(train_data, epochs=EPOCHS,
                                                  steps_per_epoch=EVALUATION_INTERVAL, # 一个epoch过完一遍数据
                                                  validation_data=val_data,
                                                  validation_steps=50)
      # 评估模型                                       
      #results = model.evaluate(x_test, y_test)

      # 存储模型
      keras_model_path = os.path.join(LSTMDIR_SINGLE,"step = {}".format(str(past_history)))
      model.save(keras_model_path)

      '''
      # 把评估结果写入磁盘
      nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
      model_results_path = os.path.join(BASEDIR,'GRU_models','results.txt')
      with open(model_results_path, 'a') as f:
        f.write('\n'+str(past_history))
        [f.write(','+str(k)) for k in results]
        title = 'Training and validation loss(step:{})'.format(str(past_history))
        plot_train_history(single_step_history,'Training and validation loss',
                          os.path.join(BASEDIR,'models',title+'.jpg'))
      '''

  # gru单步
  def train_gru_single(self):

    # 训练集
    x_train_single, y_train_single ,x_val_single, y_val_single,x_test, y_test = \
                                                self.data_p.trans_data(
                                                  self.past_history,
                                                  future_target = 1)

    train_data = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCHSIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data = val_data.batch(BATCHSIZE).repeat()

    model = Sequential()
    model.add(GRU(32,input_shape=x_train_single.shape[-2:],
                                  return_sequences=True))
    model.add(Dropout(RATE))
    model.add(GRU(32, return_sequences=True)) 
    model.add(Dropout(RATE))
    model.add(GRU(32))  
    model.add(Dropout(RATE))
    model.add(Dense(1))
    model.add(Activation('tanh'))

    model.compile(optimizer=RMSprop(), loss='mse',\
      metrics=self.metric)
    
    # 训练
    single_step_history = model.fit(train_data, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL, # 一个epoch过完一遍数据
                                                validation_data=val_data,
                                                validation_steps=50)
    # 评估模型                                       
    # results = model.evaluate(x_test, y_test)

    # 存储模型
    keras_model_path = os.path.join(GRUDIR_SINGLE,"step = {}".format(str(self.past_history)))
    model.save(keras_model_path)

  def train_mlp(self):
    x_train, y_train,x_test, y_test = \
                                self.data_p.trans_data(self.past_history,
                                                      future_target = FUTURE_TARGET,
                                                      time_dim=False)
    input_dim = x_train.shape[-1]
    model = tf.keras.models.Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(FUTURE_TARGET, activation='tanh'))

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=self.metric)

    single_step_history = model.fit(x_train, y_train,
              epochs=EPOCHS,
              batch_size=128)
    #results = model.evaluate(x_test, y_test, batch_size=128)
    # 存储模型
    keras_model_path = os.path.join(MLPDIR,"step = {},pre = {}".\
      format(str(self.past_history),str(FUTURE_TARGET)))
    model.save(keras_model_path)

  def train_svr(self,d = 1):

    x_train, y_train ,x_test, y_test = \
      self.data_p.trans_data(self.past_history,
                            future_target = d,
                            time_dim=False)

    if(d != 1):
      fdir =  'step = {},pre = {}'.format(str(self.past_history),str(d))
      y_train = y_train[:,d-1]
    else:
      fdir =  'step = {}'.format(str(self.past_history))

    clf = svm.SVR()
    clf.fit(x_train, y_train)
    #y_pre = clf.predict(x_test)

    #plot_one_method(y_test,y_pre)
    with open(os.path.join(SVRDIR,fdir),'wb') as f:
      pickle.dump(clf,f)

    return 

def train_all(net):
  net.train_mlp()
  net.train_gru_multi()
  net.train_lstm_multi()
  net.train_svr(1)
  net.train_svr(3)
  net.train_svr(6)

def pre_all(net):
  for p in [1,3,6]:

    plt.clf()

    net.pre_gru_multi(p)
    net.pre_lstm_multi(p)
    net.pre_mlp_multi(p)
    net.pre_svr_multi(p)


    plt.plot([0,1],[0,1],c = 'k',label = 'Diagonal')
    title = "Lead-time = {}".format(str(p))
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
  net = Net()
  pre_all(net)
