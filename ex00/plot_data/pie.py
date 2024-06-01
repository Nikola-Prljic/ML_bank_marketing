import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from hist import hist_3

#sielent downcating warning
pd.set_option('future.no_silent_downcasting', True)


def norm(x: np.ndarray):
    x = (x - x.min()) / (x.max() - x.min())
    return x
    
def norm_list_of_ndarray(x: list):
    x = [norm(ary) for ary in x]
    return x

def violin_plot(ary_age, ary_balance, ary_education):
    la = norm_list_of_ndarray([ary_age, ary_balance])
    violin_parts = plt.violinplot(la)
    for vp in violin_parts['bodies']:
        vp.set_facecolor('r')
        vp.set_edgecolor('g')
        vp.set_linewidth(1)
        vp.set_alpha(0.5)
        
def hist_2D(age, balance, duration):
    #plt.hist2d(ary_age, ary_balance / 1000)
    plt.style.use('_mpl-gallery-nogrid')
    x, y, z = norm_list_of_ndarray([balance, duration, age])
    #z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
    levels = np.linspace(z.min(), z.max(), 7)
    
    plt.plot(x, y, 'o', markersize=2, color='lightgrey')
    plt.tricontour(x, y, z, levels=levels)
    


def main():
    df_feature = pd.read_parquet('../Dataset/bank_marketing_features.gzip')[:820000]
    df_label = pd.read_parquet('../Dataset/bank_marketing_label.gzip')[:820000]

    ary_age = df_feature['age'].to_numpy(dtype=np.int8)
    ary_balance = df_feature['balance'].to_numpy(dtype=np.int32)
    ary_duration = df_feature['duration'].to_numpy(dtype=np.int32)
    ary_label = (df_label['y'] == 'yes').astype(dtype=int).to_numpy(dtype=np.int8)
    ary_education = df_feature['education'].replace([None, 'secondary', 'primary', 'tertiary'], [0, 1, 2, 3]).to_numpy(dtype=np.int8)

    plt.figure(1, figsize=(16, 9))
    plt.subplot(231)
    colors = ['green', 'red', 'yellow']
    plt.hist([ary_age, ary_balance / 1000, ary_label], 15, density = True, 
            histtype ='bar',
            color = colors,
            label = ['age', 'balance', 'sub'])
    
    
    
    #plt.xlim((0, 200))
    plt.legend(prop ={'size': 10})
    
    plt.subplot(232)
    plt.title('duration / balance\n',
            fontweight = "bold")
    #hist_3(ary_age, ary_balance, ary_label, ary_education)
    #plt.hist(ary_age, label='age')
    #plt.hist(ary_balance / 1000, label='balance')
    col = np.where(ary_label == 0, 'r', 'g')
    plt.scatter(ary_duration, ary_balance, color=col, label='NO SUB', alpha=0.4)
    plt.scatter([1], [1], color='g', label='SUB')
    plt.ylim(-2000, 40000)
    plt.xlim(-100, 3000)
    plt.xlabel('Duration')
    plt.ylabel('Balance')
    plt.legend(prop ={'size': 10})
    #plt.xticks([0, 1, 2, 3], ['None', 'secondary', 'primary', 'tertiary'])
    #plt.show()

    yes_cnt = np.count_nonzero(ary_label)
    no_cnt = ary_label.size - yes_cnt
    
    
    """exit(1)"""
    plt.subplot(233)
    plt.pie([yes_cnt, no_cnt], labels=['Yes', 'No'], autopct='%1.0f%%')
    plt.title('Client will subscribe')
    #plt.show()

    plt.subplot(234)
    col = np.where(ary_label == 0, 'r', 'g')
    plt.scatter(ary_age, ary_balance, color=col)
    plt.xlabel('Age')
    plt.ylabel('Balance')
    #plt.hist(ary_label)
    
    plt.subplot(235)
    violin_plot(ary_age, ary_balance, ary_education)
    plt.subplot(236)
    hist_2D(ary_age, ary_balance, ary_duration)
    plt.show()


if __name__ == '__main__':
    main()