import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

class DataHandler:
    # コンストラクタ
    def __init__(self):
        print('data files are ...\n')
        file_names = glob('..\\data\\*')
        for i, name in enumerate(file_names): 
            print('{}: {}'.format(i,name.split('\\')[2]))
        indexes = []
        while True:
            try:
                selects = input('\nselect indexes...')
                indexes = []
                print(indexes)
                for index in selects.split(' '):
                    indexes.append(int(index))
                print(file_names)
                if max(indexes)>=len(file_names):
                    print('input number exceeds the number of files!')
                    continue
            except:
                print('invalid input!')
                continue
            break
        self.datas = []

        for index in indexes:
            file_name = file_names[index]
            self.datas.append((file_name,pd.read_csv(file_name,skiprows=4)))

    def show_selected_datas(self):
        markers = ['o','^','s','D','+','x']
        for i,data in enumerate(self.datas):
            plt.subplot(len(self.datas),1,i+1)
            label = data[0].split('\\')[-1].split('.')[0]
            plt.plot(data[1].index,data[1]['加速度X'].to_numpy(),label='accx:{}'.format(label),marker=markers[i],color='orangered',linestyle='',ms=1)
            plt.plot(data[1].index,data[1]['加速度Y'].to_numpy(),label='accy:{}'.format(label),marker=markers[i],color='deepskyblue',linestyle='',ms=1)
            plt.plot(data[1].index,data[1]['加速度Z'].to_numpy(),label='accz:{}'.format(label),marker=markers[i],color='limegreen',linestyle='',ms=1)
            plt.legend()
        plt.show()
    
    def show_fft_result(self):
        markers = ['o','^','s','D','+','x']
        for i,data in enumerate(self.datas):
            plt.subplot(len(self.datas),1,i+1)
            label = data[0].split('\\')[-1].split('.')[0]
            N = len(data[1])
            data[1]['accelX_fft']= abs(np.fft.fft(data[1]['加速度X'],axis=0)/(N/2))
            data[1]['accelY_fft']= abs(np.fft.fft(data[1]['加速度Y'],axis=0)/(N/2))
            data[1]['accelZ_fft']= abs(np.fft.fft(data[1]['加速度Z'],axis=0)/(N/2))
            freq = np.linspace(0,1000,N)
            plt.plot(freq,data[1]['accelX_fft'].to_numpy(),label='accx:{}'.format(label),marker=markers[i],color='orangered',ms=1)
            plt.plot(freq,data[1]['accelY_fft'].to_numpy(),label='accy:{}'.format(label),marker=markers[i],color='deepskyblue',ms=1)
            plt.plot(freq,data[1]['accelZ_fft'].to_numpy(),label='accz:{}'.format(label),marker=markers[i],color='limegreen',ms=1)
            plt.xscale("log")
            plt.legend()
        plt.show()

    def _calc_corr(self,name,df1,df2):
        d1 = df1[name]
        d2 = df2[name]
        mea_d1 = d1 - d1.mean()
        mea_d2 = d2 - d2.mean()
        corr = np.correlate(mea_d1,mea_d2,'same')
        corr /= (np.linalg.norm(mea_d1,ord=2)*np.linalg.norm(mea_d2,ord=2))
        return corr

    def plot_data_corr(self):
        if len(self.datas) % 2 != 0:
            return
        for i in range(0,len(self.datas),2):
            plt.subplot(int(len(self.datas)/2),1,int(i/2+1))
            label = self.datas[i][0].split('\\')[-1].split('.')[0].split('_')[-1]+'&'+self.datas[i+1][0].split('\\')[-1].split('.')[0].split('_')[-1]

            corrx = self._calc_corr('加速度X',self.datas[i][1],self.datas[i+1][1])
            corry = self._calc_corr('加速度Y',self.datas[i][1],self.datas[i+1][1])
            corrz = self._calc_corr('加速度Z',self.datas[i][1],self.datas[i+1][1])
            plt.plot(range(len(corrx)),corrx,label='accx:{}'.format(label),color='orangered',alpha = 0.3)
            plt.plot(range(len(corry)),corry,label='accy:{}'.format(label),color='deepskyblue',alpha = 0.3)
            plt.plot(range(len(corrz)),corrz,label='accz:{}'.format(label),color='limegreen',alpha = 0.3)
            plt.legend()
            print(label,'Xmax:{}'.format(max(corrx)),'Ymax:{}'.format(max(corry)),'Zmax:{}'.format(max(corrz)))
        plt.show()
            


        
def test():
    handler = DataHandler()
    handler.plot_data_corr()

def main():
    handler = DataHandler()
    for data in handler.datas:
        print(data[0],data[1].head())
    handler.show_selected_datas()
    handler.show_fft_result()
    handler.plot_data_corr()
    
# スクリプトが直接実行された場合にのみ main 関数を呼び出す
if __name__ == "__main__":
    test()



