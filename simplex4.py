import numpy as np
import matplotlib.pyplot as plt

# ======================================
# Ellipsoid 
# ======================================

# d=5   #it:100
# d = 10 #it:100
# d = 30 # it:1000
d = 50 # it:1000
iteration = 1000
x = np.random.uniform(-600, 600, (d+1, d))

def f(x):
    y1 = 0.0
    for i in range(d):
        y1 += x[i]**2/4000
    y2 = 1.0
    for i in range(d):
        y2 *= np.cos(x[i]/np.sqrt(i+1))
    return 1+y1+y2

def simplex(x):
    max_list = [] # 存储最大值
    min_list = [] # 存储最小值

    for kk in range(iteration):
        y = np.zeros(d+1)
        for i in range(len(x)):
            y[i] = f(x[i])
        idx = np.argsort(y) # 得到从小到大的索引值
        if np.linalg.norm(x[idx[-1]]-x[idx[0]]) < 1e-5:
            break;
        max_list.append(f(x[idx[-1]]))
        min_list.append(f(x[idx[0]]))

        res = x[idx[-1]].copy()

        # m = np.mean(x, axis=0)
        m = np.mean(x)

        reflect = 2*m-x[idx[-1]] # 令t=-1
        reflect_f = f(reflect)
        if y[idx[0]] <= reflect_f and reflect_f < y[idx[-2]]:   #如果在最小和次大之间
            x[idx[-1]] = reflect
        elif reflect_f < y[idx[0]]: # 如果比最小还要小
            s = m+2*(m-x[idx[-1]])
            s_f = f(s)
            if s_f < reflect_f:
                x[idx[-1]] = s
            else:
                x[idx[-1]] = reflect
        elif reflect_f >= y[idx[-2]]: #比次大还要大
            if reflect_f   < y[idx[-1]]: # 比最大要小
                # c1 = m+(reflect-m)/2
                c1 = 3*m/2-x[idx[-1]]/2
                c1_f = f(c1)
                if c1_f < reflect_f:
                    x[idx[-1]] = c1
                    continue
                for ii in range(1, len(x)):
                    x[ii] = 0.5*(x[0] + x[ii])
                # print('shrink')
            else : # 比最大更大
                c2 = m+(x[idx[-1]] -m )/2
                c2_f = f(c2)
                if c2_f < reflect_f:
                    x[idx[-1]] = c2
                    continue
                for ii in range(1, len(x)):
                    x[ii] = 0.5*(x[0] + x[ii])
                # print('shrink')
    return max_list, min_list, res

max_list, min_list, res = simplex(x)
print('函数最小值为:{}'.format(min_list[-1]))
print('在x={}的时候'.format(res))

plt.plot(min_list, 'r', label="min")
plt.plot(max_list, 'g', label="max")
plt.legend()
plt.show()


# 参考文献
# 1、http://wuli.wiki/online/NelMea.html
# 2、https://www.cnblogs.com/simplex/p/6777705.html


