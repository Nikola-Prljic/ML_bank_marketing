import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def norm(x: np.ndarray):
    x = (x - x.min()) / (x.max() - x.min())
    return x
    
def norm_list(x: list):
    for ary in x:
        norm(ary)
    return x

    

def hist_3(ary_age, ary_balance, ary_label, ary_education):
    colors = ['green', 'red', 'yellow']

    """ry_age = norm(ary_age)
    ary_balance = norm(ary_balance)
    ary_education = norm(ary_education)
    print(ary_age)"""

    plt.hist([ary_age, ary_education, ary_balance / 1000], 15, density = True, 
            histtype ='bar',
            color = colors,
            label = ['age', 'education', 'sub'])
    
    
    
    #plt.xlim((0, 200))
    plt.legend(prop ={'size': 10})
    
    plt.title('matplotlib.pyplot.hist() function Example\n\n',
            fontweight = "bold")
    
    plt.show()
