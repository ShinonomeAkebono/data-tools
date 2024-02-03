import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

class DataHandler:
    # コンストラクタ
    def __init__(self):
        print('data files are ...\n')
        file_names = glob('../data/*')
        for i, name in enumerate(file_names): 
            print('{}: {}'.format(i,name.split('/')[2]))
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
            label = data[0].split('/')[-1].split('.')[0]
            plt.plot(data[1]['カウンター'].to_numpy(),data[1]['加速度X'].to_numpy(),label='accx:{}'.format(label),marker=markers[i],color='orangered',linestyle='',ms=1)
            plt.plot(data[1]['カウンター'].to_numpy(),data[1]['加速度Y'].to_numpy(),label='accy:{}'.format(label),marker=markers[i],color='deepskyblue',linestyle='',ms=1)
            plt.plot(data[1]['カウンター'].to_numpy(),data[1]['加速度Z'].to_numpy(),label='accz:{}'.format(label),marker=markers[i],color='limegreen',linestyle='',ms=1)
            plt.legend()
        plt.show()
    
    def show_fft_result(self):
        markers = ['o','^','s','D','+','x']
        for i,data in enumerate(self.datas):
            plt.subplot(len(self.datas),1,i+1)
            label = data[0].split('/')[-1].split('.')[0]
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

            
        
def main2():
    handler = DataHandler()
    for data in handler.datas:
        print(data[0],data[1].head())
    handler.show_selected_datas()
    handler.show_fft_result()

def main():

    print('data files are ...\n')
    file_names = glob('../data/*')
    count = 0
    for name in file_names: 
        print('{}: {}'.format(count,name.split('/')[2]))
        count+=1
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
    datas = []

    for index in indexes:
        file_name = file_names[index]
        datas.append(pd.read_csv(file_name))
    
# スクリプトが直接実行された場合にのみ main 関数を呼び出す
if __name__ == "__main__":
    main2()



