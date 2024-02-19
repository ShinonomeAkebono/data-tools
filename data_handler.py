import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from scipy import signal

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
        self.wp = 10**-1.0
        self.ws = 10**-0.5
        self.gpass = 3
        self.gstop = 40
            
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
                    label,line,plt_data,_ = process(key,self.datas[i],self.datas[i+1])
                    if j<3:
                        axs[row_i,0].plot(line,plt_data[1],label=label,color=colors[j%3],linestyle=linestyle,alpha=alpha,marker='o',ms=1)
                    else:
                        axs[row_i,1].plot(line,plt_data[1],label=label,color=colors[j%3],linestyle=linestyle,alpha=alpha,marker='o',ms=1)
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
                    label,line,plt_data,_ = process(key,data)
                    if j<3:
                        axs[i,0].plot(line,plt_data[1],label=label,color=colors[j%3],linestyle=linestyle,alpha=alpha,marker='o',ms=1)
                        axs[i,0].set_ylim(-30,30)
                    else:
                        axs[i,1].plot(line,plt_data[1],label=label,color=colors[j%3],linestyle=linestyle,alpha=alpha,marker='o',ms=1)
                        axs[i,1].set_ylim(-100,100)
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
        return label,ret_line,ret_data,name

    def _calc_fft(self,name,data):
        label = self.jp_en[name] + ':' + data[0].split('\\')[-1].split('.')[0]
        N = len(data[1])
        window = signal.windows.hann(N)
        ret_line = np.linspace(0,1,N//2) 
        fft_result = abs(np.fft.fft(data[1][name]*window,axis=0)/(N/2))
        plot_data = fft_result[:N//2]
        ret_data = (data[0],plot_data)
        return label,ret_line,ret_data,name

    def _calc_corr(self,name,data1,data2):
        label = self.jp_en[name] + ':' + data1[0].split('\\')[-1].split('.')[0].split('_')[-1]+'&'+data2[0].split('\\')[-1].split('.')[0].split('_')[-1]
        d1 = data1[1][name]
        d2 = data2[1][name]
        mea_d1 = d1 - d1.mean()
        mea_d2 = d2 - d2.mean()
        corr = signal.correlate(mea_d1,mea_d2)
        corr /= (np.linalg.norm(mea_d1,ord=2)*np.linalg.norm(mea_d2,ord=2))
        ret_line = range(len(corr))
        ret_data = (label,corr)
        print("{}'s max value is...:{}".format(label,max(corr)))
        return label,ret_line,ret_data,name
    
    def _calc_lpf_corr(self,name,data1,data2):
        _,_,lpf_data1,_=self._proc_lpf(name,data1)
        _,_,lpf_data2,_=self._proc_lpf(name,data2)
        return self._calc_corr(name,lpf_data1,lpf_data2)
       
    def _calc_lpf_corrmax(self,name,data1,data2):
        scale = np.linspace(-4,-0.6,100)
        max_list = []
        for i in range(len(scale)):
            self.wp=10**scale[i]
            self.ws=10**(scale[i]+0.6)# if scale[i] < -1 else 1
            df1_cp = data1[1].copy()
            df2_cp = data2[1].copy()
            #df1_cp[name] = self._proc_lpf_df(df1_cp[name])
            df2_cp[name] = self._proc_lpf_df(df2_cp[name])
            data1_cp=(data1[0],df1_cp)
            data2_cp=(data2[0],df2_cp)
            label,_,corr,name = self._calc_corr(name,data1_cp,data2_cp)
            max_list.append(max(corr[1]))
        ret_content = pd.DataFrame({name:max_list})
        ret_content = ret_content.fillna(ret_content[name].mean())
        ret_line = scale
        ret_data = (label,ret_content)
        return label,ret_line,ret_data,name
        
    def _proc_lpf(self,name,data):
        label = [data[1]]
        ret_line = data[1].index
        #バターワースフィルタの次数nと正規化周波数wnを計算
        n,wn = signal.buttord(wp=self.wp,ws=self.ws,gpass=self.gpass,gstop=self.gstop)
        #ローパスフィルタの伝達関数の分子分母b,aを計算
        b,a = signal.butter(n,wn,btype='low') 
        #フィルタリング
        data[1][name]=signal.filtfilt(b,a,data[1][name])
        return label,ret_line,data,name 

    def _proc_lpf_df(self,column):
        #バターワースフィルタの次数nと正規化周波数wnを計算
        n,wn = signal.buttord(wp=self.wp,ws=self.ws,gpass=self.gpass,gstop=self.gstop)
        #ローパスフィルタの伝達関数の分子分母b,aを計算
        b,a = signal.butter(n,wn,btype='low') 
        #フィルタリング
        return signal.filtfilt(b,a,column)
        

    def _calc_lpf_show(self,name,data):
        _,_,lpf_data,lpf_name = self._proc_lpf(name,data)
        label,ret_line,ret_data,ret_name = self._not_calc(lpf_name,lpf_data)
        return label,ret_line,ret_data,ret_name
        
    def _calc_lpf_fft(self,name,data):
        _,_,lpf_data,lpf_name = self._proc_lpf(name,data)
        label,ret_line,ret_data,ret_name = self._calc_fft(lpf_name,lpf_data)
        return label,ret_line,ret_data,ret_name

        
    def show_selected_datas(self):
        self._show_datas(process=self._not_calc)
    
    def show_fft_result(self):
        self._show_datas(process=self._calc_fft,linestyle=None,xscale_log=True)

    def show_data_corr(self):
        self._show_datas(process=self._calc_corr,alpha=0.3,pair=True)
    
    def show_lpf_result(self):
        self._show_datas(process=self._calc_lpf_show)
        self._show_datas(process=self._calc_lpf_fft,linestyle=None,xscale_log=True)
    
    def show_processed_corr(self):
        self._show_datas(process=self._calc_lpf_corr,alpha=0.3,pair=True)
        
    def show_corrmax_graph(self):
        self._show_datas(process=self._calc_lpf_corrmax,pair=True,linestyle=None)

        
        
def test():
    handler = DataHandler()
    handler.show_data_corr()

def main():
    command = input('please select command number... \n1:just show\n2:fft\n3:corriration\n4:LPF\n5:LPF corriration\n6:corriration max graph\n')
    
    handler = DataHandler()
    
    if command == '1':
        handler.show_selected_datas()
    elif command == '2':
        handler.show_fft_result()
    elif command == '3':
        handler.show_data_corr()
    elif command == '4':
        handler.show_lpf_result()
    elif command == '5':
        handler.show_processed_corr()
    elif command == '6':
        handler.show_corrmax_graph()
    
if __name__ == "__main__":
    main()



