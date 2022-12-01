#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import matplotlib.pyplot as plt
import numpy as np


def graph_draw(x, y):  # 折线图

    plt.plot(x, y, color="red", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def method(data):
    # print(data.__len__())
    tmp = data[0]
    for item in data:
        if tmp > item:
            tmp = item
    # for i, item in zip(range(data.__len__()), data):
       # data[i] = data[i] / tmp

    xs = list()
    for i in range(1, data.__len__() + 1):
        xs.append(i)
    ys = data

    # graph_draw(xs, ys)

    # 余弦相似度
    def cosin_distance(vector1, vector2):
        user_item_matric = np.vstack((vector1, vector2))
        sim = user_item_matric.dot(user_item_matric.T)
        norms = np.array([np.sqrt(np.diagonal(sim))])
        user_similarity = (sim / norms / norms.T)[0][1]
        return user_similarity

    # 波峰在8-16个
    ll = list()
    sim = list()
    # 周期长度
    win = 0
    wintmp = 0.

    print("数据量：",data.__len__() )

    for wd in range(int(data.__len__() / 12), int(data.__len__() / 8)):
    #设置峰个数，一般在8-16之间
        n = int(data.__len__() / wd)
        tmp = 0.

        for i in range(n):
            ll.append(data[wd * i:wd * (i + 1)])

        for l, i in zip(ll, range(ll.__len__() - 1)):
            # print(cosin_distance(ll[i], ll[i+1]))
            tmp = tmp + cosin_distance(ll[i], ll[i + 1])
        if (tmp / (n - 1) > wintmp):
            wintmp = tmp / (n - 1)
            win = wd

        sim.append(tmp / (n - 1))
        # print(wd, "--", tmp/(n-1))
        tmp = 0.
        ll.clear()

    # print(win)

    begin = 0
    begin_e = begin + win
    end = data.__len__()
    b_end = end - win

    bly = 100
    ely = 100
    blx = 0
    elx = 0
    for i in data[begin:begin_e]:
        if i < bly:
            bly = i
    for i, j in zip(range(win), data[begin:begin_e]):
        if bly == j:
            blx = i

    for i in data[b_end:end]:
        if i < ely:
            ely = i
    for i, j in zip(range(data.__len__() - win, data.__len__()), data[b_end:end]):
        if ely == j:
            elx = i

    # print(blx,"--", elx)
    # print(win)

    cycle_num = int((elx - blx) / win + 1)
    cycle_data = data[win * 0:win * 1]

    x_data = list()
    for i in range(1, 18):
        x_data.append(i)
    '''
    for i in range(20):
        cycle_data = data[win*i:win*(i+1)]
        graph_draw(x_data, cycle_data)
    '''
    # graph_draw(x_data, cycle_data)
    for i in range(1, cycle_num):
        for j in range(data[win * (i - 1):win * i].__len__()):
            cycle_data[j] = cycle_data[j] + data[win * (i - 1):win * i][j]

    for i in range(cycle_data.__len__()):
        cycle_data[i] = cycle_data[i] / cycle_num

    # print(cycle_data)
    # graph_draw(x_data, cycle_data)
    x = list()
    # 此处修改
    k = 0.00488
    #设定斜率：k=记录时间/（data.csv 所得数值行的个数-1）。（若实验记录时间为10s，data.csv 共 2048 行，k=10/（2048-1）=0.00488）
    import sys
    min_x = 0
    min_t = sys.maxsize
    for mp in range(win):
        if(cycle_data[mp] < min_t):
            min_t = cycle_data[mp]
            min_x = mp

    cycle_data = cycle_data[min_x:len(cycle_data)] + cycle_data[0:min_x]

    for i in range(win):
        # x调整
        x.append(i * k)

    x = np.array(x)
    y = np.array(cycle_data)

    # 用n次多项式拟合
    f1 = np.polyfit(x, y, 10)
    print('f1 is :\n', f1)

    p1 = np.poly1d(f1)
    print('p1 is :\n', p1)
    print()

    max = 0.
    max_pos = 0.
    min_pos = 0.
    for item, i in zip(cycle_data, range(win)):
        if item > max:
            max = item
            max_pos = i
    cycle_data_up = cycle_data[0:max_pos]

    min = max

    for item, i in zip(cycle_data_up, range(max_pos)):

        if item < min:
            min = item
            min_pos = i

    print("数据集y最大值：", max)

    x_max = 0.
    max_tmp = p1(x_max)
    x_p = 0.

    while True:
        x_max = x_max + k * 0.01
        if (p1(x_max) > max_tmp):
            max_tmp = p1(x_max)
            x_p = x_max

        if x_max >= win * k:
            break

    import sys
    x_min = 0.
    min_tmp = sys.maxsize + 0.0
    x_m = 0.

    while True:
        x_min = x_min + k * 0.01
        if (p1(x_min) < min_tmp):
            min_tmp = p1(x_min)
            x_m = x_min
        if x_min >= x_p:
            break

    cycle_data_up = cycle_data[min_pos:max_pos]
    cycle_data_down = cycle_data[max_pos:win]

    max = p1(x_p)
    print("拟合函数y最大值为：", max)

    x_l = x_m
    x_r = win * k

    slope_max = 0.
    slope_max_x = x_l

    xt = x_l

    #print(x_l, "--", x_r)

    while True:

        slope_tmp = float((p1(xt + (k * 0.01)) - p1(xt)) / (k * 0.01))
        # print(slope_tmp)

        if slope_tmp > slope_max:
            slope_max = slope_tmp
            slope_max_x = xt

        xt = xt + k * 0.01

        if xt >= x_r:
            break

    print("斜率最大值(Max Upstroke Velocity)为：", slope_max, "   x=", slope_max_x, " y=", p1(slope_max_x))



    x_l = x_m
    x_r = win*k

    import sys

    slope_min = sys.maxsize + 0.0
    slope_min_x = x_l

    xt = x_l

    # print(x_l, "--", x_r)
    while True:

        slope_tmp = float((p1(xt + (k * 0.01)) - p1(xt)) / (k * 0.01))
        # print(slope_tmp)

        if slope_tmp < slope_min:
            slope_min = slope_tmp
            slope_min_x = xt

        xt = xt + k * 0.01

        if xt >= x_r:
            break

    print("斜率最小值(Max Recovery Velocity)为：", slope_min, "   x=", slope_min_x, " y=", p1(slope_min_x))

    def method_apd(APD):

        APD = float(APD)

        if APD > p1(x_p):
            print("  大于最大值！")
            return
        if APD < min:
            print("  小于最小值！")
            return

        xl = list()

        bl2 = True
        mid = x_m

        l = float(x_m)
        r = x_p
        #print(l,"---",r)

        while (bl2):

            if(APD < p1(l)):
                bl2 = False
            if(APD > p1(r)):
                bl2 = False

            mid = (l + r) / 2
            #print(mid)

            if p1(mid) > APD:
                r = mid
                if abs(p1(mid) - APD) < 0.001 * k:
                    xl.append(mid)
                    bl2 = False

            elif p1(mid) < APD:
                l = mid
                if abs(p1(mid) - APD) < 0.001 * k:
                    xl.append(mid)
                    bl2 = False
            else:
                xl.append(mid)

        bl2 = True
        mid = min_pos

        l = x_p
        r = win * k
        #print(l,"---",r)

        while (bl2):

            if (APD > p1(l)):
                bl2 = False
            if (APD < p1(r)):
                bl2 = False

            mid = (l + r) / 2
            #print(mid)

            if p1(mid) > APD:
                l = mid
                if abs(p1(mid) - APD) < 0.001 * k:
                    xl.append(mid)
                    bl2 = False

            elif p1(mid) < APD:
                r = mid
                if abs(p1(mid) - APD) < 0.001 * k:
                    xl.append(mid)
                    bl2 = False
            else:
                xl.append(mid)
        if xl.__len__() == 0:
            print("    APD无对应x值")
        elif xl.__len__() == 1:
            print("    APD存在一个对应x值， x值为：", xl)
        elif xl.__len__() == 2:
            print("    APD存在两个对应x值  x值为：", xl, "  差值:", xl[1] - xl[0])
        else:
            print("    APD求解对应x值出现错误！")

    #下面是设定 APD 参数：若求 APD80，则修改 APD80=0+(max-0)*0.2。
    APD80 = 0 + (max - 0) * 0.2
    print("APD80:", APD80, end="")
    method_apd(APD80)

    APD50 = 0 + (max - 0) * 0.5
    print("APD50", APD50, end="")
    method_apd(APD50)


    yvals = p1(x)  # 拟合y值
    # print('yvals is :\n',yvals)
    # 绘图
    plot1 = plt.plot(x, y, 's', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=4)  # 指定legend的位置右下角
    plt.title('polyfitting')
    plt.show()

    print()
    bl1 = True
    while bl1:
        s = input(" x or y(退出请按0): ")

        if s == 'x':
            x = input("请输入x： ")
            x = float(x)
            y = p1(x)
            print("y = ", y)

        elif s == 'y':
            y = input("请输入y： ")
            y = float(y)

            xl = list()

            if y > p1(x_p):
                print("大于最大值！")
                continue
            if y < min:
                print("小于最小值！")
                continue

            bl2 = True
            mid = x_m

            l = float(x_m)
            r = x_p

            while (bl2):

                mid = (l + r) / 2
                # print(mid)

                if p1(mid) > y:
                    r = mid
                    if abs(p1(mid) - y) < 0.001 * k:
                        xl.append(mid)
                        bl2 = False

                elif p1(mid) < y:
                    l = mid
                    if abs(p1(mid) - y) < 0.001 * k:
                        xl.append(mid)
                        bl2 = False
                else:
                    xl.append(mid)

            bl2 = True
            mid = x_m

            l = x_p
            r = win * k

            while (bl2):

                mid = (l + r) / 2

                if p1(mid) > y:
                    l = mid
                    if abs(p1(mid) - y) < 0.001 * k:
                        xl.append(mid)
                        bl2 = False

                elif p1(mid) < y:
                    r = mid
                    if abs(p1(mid) - y) < 0.001 * k:
                        xl.append(mid)
                        bl2 = False
                else:
                    xl.append(mid)
            print("x值为：", xl)

        elif s == '0':
            bl1 = False
        else:
            print("wrong!")

        print()


def main():
    # 读取excel数据
    csv_file = csv.reader(open("data.csv", 'r'))

    i = 0

    all_data = list()

    # n = list(csv_file).__len__()

    row_num = ""
    for line in csv_file:
        row_num = len(line)
        break

    tmp_data = list()
    for line in csv_file:
        tmp_data.append(line)
        #print(line)
    for j in range(row_num):
        data = list()
        for tmp in tmp_data:

            if (list(tmp[j]).__len__() != 0 and type(tmp[j]) is str):
                data.append(float(tmp[j]))
        all_data.append(data)
    '''
    for j in range(row_num):
        data = list()
        for line in csv_file:
            print(j,"-00-", line)
            if i == 0:
                i = 1
                continue


            if (list(line[j]).__len__() != 0 and type(line[j]) is str):
                data.append(float(line[j]))
                #print(j,"--",float(line[j]))

        all_data.append(data)

        i = i + 1
    '''

    for data in all_data:
        method(data)




if __name__ == '__main__':
    main()

