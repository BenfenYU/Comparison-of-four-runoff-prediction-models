import os,re,pickle,sys, imp, os,math,json
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from sklearn.preprocessing import MinMaxScaler  # 归一化编码
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from scipy import stats
'''
Returns of function pearsonr:
    rfloat
        Pearson’s correlation coefficient.
    p-valuefloat
        Two-tailed p-value.

'''


NPCC = "Cross correlation"
PCC = "Pearson cross correlation"
fol = "俄亥俄河"
num = 11
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR,"data",fol)

month = [31,28,31,30,31,30,31,31,30,31,30,31]

def adday(y,m,d,n = 1):
    if (y%4==0 and y%100!=0)or y%400==0:
        month[1] = 29
    else :
        month[1] = 28
    if (d == month[m-1]):
        if ( m == 12):
            y = y+1
            m = d = 1
        else:
            m = m+1
            d = 1
    else:
        d = d+1
    
    return y,m,d

def readRain(dire):
    '''
    读降雨的数据，返回成单位是元组的列表，元组中有日期和数据，用于和径流对齐
    dire是存放数据的目录
    '''
    fileList = os.listdir(dire)

    for fi in fileList:
        if os.path.splitext(fi)[1] == '.txt':
            fileRain = fi
    # 降雨
    frain =  open(os.path.join(dire,fileRain),'r')

    rain = []
    for l in frain:
        if not l:
            break
        if(l[0] == '#'):
            continue

        a = re.match("^\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)\n*", l, flags=0)
        if(not a):
            continue
        y = int(a.group(1))
        m = int(a.group(2))
        d = int(a.group(3))
        data = float(a.group(4))

        if(len(rain) == 0):
            rain.append((y,m,d,data))
            continue
        ly = rain[-1][0]
        lm = rain[-1][1]
        ld = rain[-1][2]
        ly,lm,ld = adday(ly,lm,ld)
        while(ly != y or lm != m or ld != d):
            rain.append((ly,lm,ld,-1))
            ly,lm,ld = adday(ly,lm,ld)
        rain.append((ly,lm,ld,data))

    frain.close()

    return rain

def readStream(dire):
    '''
    读径流的数据，返回成单位是元组的列表，元组中有日期和数据，用于和降雨对齐
    dire是存放数据的目录
    '''
    fileList = os.listdir(dire)

    for fi in fileList:
        if os.path.splitext(fi)[1] == '.csv':
            fileStream = fi
    # 径流
    fstream = open(os.path.join(dire,fileStream),'r')
    stream = []

    for l in fstream:
        l = l.replace(",",".")
        a = re.match("^(\d+)/(\d+)/(\d+)\.\"(.*)[A-Z].*\n*$", l, flags=0)
        if(not a):
            continue
        m = int(a.group(1))
        d = int(a.group(2))
        y = int(a.group(3))
        data = float(a.group(4))

        if(len(stream) == 0):
            stream.append((y,m,d,data))
            continue
        ly = stream[-1][0]
        lm = stream[-1][1]
        ld = stream[-1][2]
        ly,lm,ld = adday(ly,lm,ld)
        while(ly != y or lm != m or ld != d):
            stream.append((ly,lm,ld,-3))
            ly,lm,ld = adday(ly,lm,ld)
        stream.append((ly,lm,ld,data))

    fstream.close()

    return stream

def dayCmp(y1,m1,d1,y2,m2,d2):
    '''
    根据年月日比较两天，谁在前谁在后
    '''
    if(y1 > y2):
        return 1
    elif(y1 < y2):
        return -1
    else:
        if(m1 > m2):
            return 1
        elif(m1 < m2):
            return -1
        else:
            if(d1 > d2):
                return 1
            elif(d1 < d2):
                return -1
            else:
                return 0

# 对齐降雨和径流数据
def alignTime():
    '''
    分别读出同一目录下的降雨和径流数据，并在其目录下以列表的形式存储
    '''
    pathList = os.listdir(DATADIR)
    for dire in pathList:
        dire = os.path.join(DATADIR,dire)
        stream = readStream(dire)
        rain = readRain(dire)
        allData = []
        i = 0
        j = 0
        while(i<len(stream) and j < len(rain)):
            r = dayCmp(stream[i][0],stream[i][1],stream[i][2],rain[j][0],rain[j][1],rain[j][2])
            if ( r== 0):
                temp_data = [stream[i][-1],rain[j][-1]]
                if(-1 not in temp_data):
                    allData.append(temp_data)
                else:
                    print(temp_data)
                i += 1
                j += 1
            elif(r > 0):
                j += 1
            else:
                i += 1
        
        with open(os.path.join(dire,'data.pickle'), 'wb') as f:
            pickle.dump(allData, f)

    return
     
def loadData(path): # 加载pickle存储的二进制数据
    pkl_file = open(path, 'rb')
    dataList = pickle.load(pkl_file)
    pkl_file.close()

    return dataList

def mm(data):
    data = data.reshape(-1,2)
    mm = MinMaxScaler(feature_range=(0, 1))
    data = mm.fit_transform(data)

    return data

def do_pears(r = False,method = PCC): # Pearson Correlation 不同的时间lag得到不同的皮尔逊相关指数
    if(r):
        with open(os.path.join(BASEDIR,"pearsons.json"),"r") as f:
            x ,y=[],[] 
            pc = json.load(f)
            for key,value in pc.items():
                x.append(int(key))
                y.append(float(value))
    else:   
        dataset = standardize(np.array(loadData()))
        mean_pcor = {}
        x ,y=[],[] 
        for i in range(2,20):
            x1 = dataset[:,0]
            x2 = dataset[:,1]
            x1 =  np.squeeze(univariate_data(x1,0,None,i,0)[0],axis=2)
            x2 = np.squeeze(univariate_data(x2,0,None,i,0)[0],axis=2)
            num = 0
            pcors = 0
            for j in range(x1.shape[0]):
                if(method == NPCC):
                    pcor = np.correlate(x1[j,:],x2[j,:])
                elif(method == PCC):
                    pcor,_ = stats.pearsonr(x1[j,:],x2[j,:])
                if(math.isnan(pcor)):
                    continue
                num+=1
                pcors += pcor
            mean_pcor[str(i)] =float(pcors / num)
            x.append(i)
            y.append(mean_pcor[str(i)])

        with open(os.path.join(BASEDIR,"pearsons.json"),"w") as f:
            json.dump(mean_pcor,f)

    plt.plot(x,y)
    plt.suptitle("Pearson correlation of different lag")
    plt.show()

    return

def do_acf2(): # Autocorrelation Function,ACF
    dataset = np.array(loadData())
    x = dataset[:,0]

    plt.acorr(x, usevlines=True, normed=True, maxlags=100)
    plt.grid(True)

    plt.show()

    return 

def do_xcorr(): # cross-correlation
    dataset = np.array(loadData())
    x1 = dataset[:,0]
    x2 = dataset[:,1]

    plt.xcorr(x1, x2, usevlines=True, maxlags=50, normed=True, lw=2)
    plt.grid(True)

    plt.show()

    return 

def do_acf1(): # Autocorrelation Function, use statsmodels
    dataset = np.array(loadData())
    x = dataset[:,0]
    plot_acf(x)
    plt.show()

    return 

def do_pacf(): # Partial Autocorrelation Function
    dataset = np.array(loadData())
    x = dataset[:,0]
    plot_pacf(x)
    plt.show()

    return 

if __name__ == "__main__":
    alignTime()
