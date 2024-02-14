import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

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
                for index in selects.split(' '):
                    indexes.append(int(index))
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
        
        self.jp_en = {
            '加速度X':'accx',
            '加速度Y':'accy',
            '加速度Z':'accz',
            '角速度X':'gyrx',
            '角速度Y':'gyry',
            '角速度Z':'gyrz',
                         }
            
    def _show_datas(self,process:Callable,linestyle='',alpha = 1.0,xscale_log = False,pair = False):
        colors = ['orangered','deepskyblue','limegreen']

        if pair:
            if len(self.datas) % 2 != 0:
                print('please select an even number of datas!')
                return
            _,axs = plt.subplots(2 if int(len(self.datas)/2) == 1 else int(len(self.datas)/2),2)

            for i in range(0,len(self.datas),2):
                row_i = int(i/2)
                for j,key in enumerate(self.jp_en.keys()):
                    label,line,plt_data = process(key,self.datas[i],self.datas[i+1])
                    if j<3:
                        axs[i,0].plot(line,plt_data[1],label=label,color=colors[j%3],linestyle=linestyle,alpha=alpha,marker='o',ms=1)
                    else:
                        axs[i,1].plot(line,plt_data[1],label=label,color=colors[j%3],linestyle=linestyle,alpha=alpha,marker='o',ms=1)
                if xscale_log:
                    axs[row_i,0].set_xscale('log') 
                    axs[row_i,1].set_xscale('log') 
                axs[row_i,0].legend()
                axs[row_i,1].legend()
            if len(self.datas) == 2:
                [plt.delaxes(ax) for ax in axs[1, :]]

        else:
            _,axs = plt.subplots(2 if len(self.datas) == 1 else len(self.datas),2)
            for i,data in enumerate(self.datas):
                for j,key in enumerate(self.jp_en.keys()):
                    label,line,plt_data = process(key,data)
                    if j<3:
                        axs[i,0].plot(line,plt_data[1],label=label,color=colors[j%3],linestyle=linestyle,alpha=alpha,marker='o',ms=1)
                    else:
                        axs[i,1].plot(line,plt_data[1],label=label,color=colors[j%3],linestyle=linestyle,alpha=alpha,marker='o',ms=1)
                if xscale_log:
                    axs[i,0].set_xscale('log') 
                    axs[i,1].set_xscale('log') 
                axs[i,0].legend()
                axs[i,1].legend()

            if len(self.datas) == 1:
                [plt.delaxes(ax) for ax in axs[1, :]]
        plt.tight_layout()
        plt.show()
            
        
    def _not_calc(self,name,data):
        label = self.jp_en[name] + ':' + data[0].split('\\')[-1].split('.')[0]
        ret_line = data[1].index
        ret_data = (data[0],data[1][name].to_numpy())
        return label,ret_line,ret_data

    def _calc_fft(self,name,data):
        label = self.jp_en[name] + ':' + data[0].split('\\')[-1].split('.')[0]
        N = len(data[1])
        ret_line = np.linspace(0,N,N) 
        ret_data = (data[0],abs(np.fft.fft(data[1][name],axis=0)/(N/2)))
        return label,ret_line,ret_data

    def _calc_corr_test(self,name,data1,data2):
        label = self.jp_en[name] + ':' + data1[0].split('\\')[-1].split('.')[0].split('_')[-1]+'&'+data2[0].split('\\')[-1].split('.')[0].split('_')[-1]
        d1 = data1[1][name]
        d2 = data2[1][name]
        mea_d1 = d1 - d1.mean()
        mea_d2 = d2 - d2.mean()
        corr = np.correlate(mea_d1,mea_d2,'same')
        corr /= (np.linalg.norm(mea_d1,ord=2)*np.linalg.norm(mea_d2,ord=2))
        ret_line = range(len(corr))
        ret_data = (label,corr)
        print("{}'s max value is...:{}".format(label,max(corr)))
        return label,ret_line,ret_data
    
    def _proc_lpf(self,name,data):
        pass
        
    def show_selected_datas(self):
        self._show_datas(process=self._not_calc)
    
    def show_fft_result(self):
        self._show_datas(process=self._calc_fft,linestyle=None,xscale_log=True)

    def show_data_corr(self):
        self._show_datas(process=self._calc_corr_test,alpha=0.3,pair=True)
    
    def show_lpf_result(self):
        self._show_datas()
        
def test():
    handler = DataHandler()
    handler.show_data_corr()

def main():
    command = input('please select command number... \n1:just show\n2:fft\n3:calculate corriration\n')
    
    handler = DataHandler()
    
    if command == '1':
        handler.show_selected_datas()
    elif command == '2':
        handler.show_fft_result()
    elif command == '3':
        handler.show_data_corr()
    
if __name__ == "__main__":
    main()



