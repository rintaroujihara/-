#!/usr/bin/env python
# coding: utf-8

# 訪日外国人流動データの可視化

# In[4]:


#get_ipython().run_line_magic('matplotlib', 'inline')
from gurobipy import *
import pandas as pd
import pylab
import math
import networkx as NX
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.cm import ScalarMappable
import matplotlib.colors as colors
import os
from sklearn import linear_model


# In[5]:


#元の都道府県の座標(平面直交座標JGD2000)
df1 = pd.read_excel('kentyou_latest.xlsx')

#都道府県間の平均移動費用 = c_{jk}
df2 = pd.read_excel('idouhiyou_average.xlsx')
df3 = df2.fillna(0)

#都道府県間の移動費用の元データ(鳥海先生より拝借)
df4 = pd.read_excel('idouhiyou_original.xlsx',header = 0)

#都道府県間の移動量(前後の都道府県のみに着目した場合 or トリップチェーンに含まれる全ての県に着目した場合)
df5 = pd.read_excel('OD_CompleteGraph.xlsx', index_col=0)

def make_data():
    #都道府県の集合
    I = range(47)
    J = range(47)
    x,y,w,v,c = {},{},{},{},{}
    
    #元の都道府県の座標(緯度経度から直交座標に変更済み)
    for i in I:
        y[i] = df1.iat[i,1]
        x[i] = df1.iat[i,2]
     
    #都道府県iの付置前と付置後の間の重み = α_i ( = 1 )
    for i in I:
        w[i] = 1
                    
    #都道府県j, k間のトリップ量の合計(上三角行列に変形) = v_{jk}            
    for j in range(47):
        for k in range(j, 47):
            if j != k:
                v[j,k] = df5.iat[j,k]+ df5.iat[k,j]
            else:
                v[j,k] = df5.iat[k,j]
                
    #都道府県間の移動費用を定義
    #元のデータではトリップ量がない都道府県間の移動費用も定義(バス、鉄道、乗用車、航空の合計を4で除算)
    #内々の移動費用は3000円
    p = 0
    for j in range(47):
        for k in range(j, 47):
            if j != k:
                if df3.iat[p,3] != 0:
                    c[j,k] = df3.iat[p,3]
                else:
                    q = 0
                    for r in range(4):
                        if df4.iat[p,35+r] == -1:
                            q += 1
                    c[j,k] = sum([(df4.iat[p,l]) for l in range(35,39)])/(4-q)
                p += 1
            else:
                c[j,k] = 3000
    
    #トリップ量×費用の合計 (β_{jk}の分母)
    summ = [0 for i in range(47)]
    for j in range(47):
        for k in range(j,47):
            summ[j] += v[j,k] * c[j,k]
        for i in range(0,j):
            summ[j] += v[i,j] * c[i,j]
    #summが0のとき
    for i in range(47):
        if summ[i] == 0:
            summ[i] = 1
            
    return I,J,x,y,w,v,c,summ


# In[6]:


#euclidean maltifacility problem
def weber(I,x,y,w,v,c,summ):
    model = Model("MFWP")
    X,Y,z,z1,xaux,yaux,x1aux,y1aux = {},{},{},{},{},{},{},{}
    
    for i in I:
        #X[i],Y[i]は配置後の位置
        X[i] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="X_{0}".format(i))
        Y[i] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="Y_{0}".format(i))
        #z[i]は元の位置との距離
        z[i] = model.addVar(vtype="C", name="z_{0}".format(i)) 
        xaux[i] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="xaux_{0}".format(i))
        yaux[i] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="yaux_{0}".format(i))
    
    for j in range(46):
        for k in range(j+1,47):
            #新しい位置同士の距離
            z1[j,k] = model.addVar(vtype="C", name="z1_{0}{1}".format(j,k))
            x1aux[j,k] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="x1aux_{0}{1}".format(j,k))
            y1aux[j,k] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="y1aux_{0}{1}".format(j,k))
            
    model.update()
    
    for i in I:
        model.addConstr(xaux[i]*xaux[i] + yaux[i]*yaux[i] <= z[i]*z[i], "MinDist_{0}".format(i))
        model.addConstr(xaux[i] == (x[i]-X[i]), "xAux_{0}".format(i))
        model.addConstr(yaux[i] == (y[i]-Y[i]), "yAux_{0}".format(i))
        
    for j in range(46):
        for k in range(j+1,47):
            model.addConstr(x1aux[j,k]*x1aux[j,k] + y1aux[j,k]*y1aux[j,k] <= z1[j,k]*z1[j,k], "MinDist1_{0}{1}".format(j,k))
            model.addConstr(x1aux[j,k] == (X[j]-X[k]), "x1Aux_{0}{1}".format(j,k))
            model.addConstr(y1aux[j,k] == (Y[j]-Y[k]), "y1Aux_{0}{1}".format(j,k))

    #目的関数の係数を設定(前半は都道府県を留まらせようとする関数、後半は都道府県同士をトリップ量と移動費用に応じて近づける関数)     
    model.setObjective(quicksum(w[i]*z[i] * 1.25 for i in I) 
                       +quicksum((1/summ[j]+1/summ[k])*v[j,k]*c[j,k]*z1[j,k] * 1 for j in range(46) for k in range(j+1, 47) ), GRB.MINIMIZE)

    model.update()
    model.__data = X,Y,z,z1
    return model

if __name__ == "__main__":
    I,J,x,y,w,v,c,summ = make_data()
    model = weber(I,x,y,w,v,c,summ)
    model.optimize()
    X,Y,z,z1 = model.__data
    
    try: #netwokxで結果を描画
        
        #元の都道府県のグラフ
        G = NX.Graph()
        #新しい都道府県のグラフ
        G1 = NX.Graph()
        #新しい都道府県と元の都道府県を結ぶ線のグラフ
        G2 = NX.Graph()
        
        pylab.figure(figsize=(30, 30))
        
        #G,G2に数字でノード名を追加
        G.add_nodes_from(I)
        G2.add_nodes_from(I)
        
        #G1とG2に都道府県名でノード名を追加
        for i in I:
            G1.add_nodes_from(["{0}".format(df1.iat[i,0])])
            G2.add_nodes_from(["{0}".format(df1.iat[i,0])])
        
        #G, G1, G2に位置情報を入力
        position = {}
        position1 = {}
        position2 = {}
        for i in I:
            position[i] = (y[i],x[i])
            position1["{0}".format(df1.iat[i,0])] = ((Y[i].X),(X[i].X))
            #position2 = position + position1
            position2[i] = (y[i],x[i])
            position2["{0}".format(df1.iat[i,0])] = ((Y[i].X),(X[i].X))
        
        #G2に元の都道府県と配置された都道府県を結ぶ線を定義
        for i in range(47):
            G2.add_edge(i, "{0}".format(df1.iat[i,0])) ,
        
        #線だけを表示(ノードを白にして消す)
        NX.draw_networkx(G2, pos=position2, node_color="w",alpha=0.5, font_size =0, with_labels=False)
        
        #都道府県の元の位置を表示
        NX.draw_networkx(G, with_labels=False, pos=position, node_size=150, node_color="green", alpha = 0.5 ,label = "付置前の位置")
        
#        都道府県の移動後の位置を表示
        NX.draw_networkx(G1, with_labels=True,font_family = "IPAexGothic", pos=position1, node_size=400, node_color="lightblue", font_size =18, label = "付置後の位置" )
        
        
        #タイトルと凡例を表示
        plt.title('パターン4', size = 60)
        plt.legend(fontsize = 40)

        #描画時のアスペクト比を１対１に設定
        plt.axes().set_aspect('equal','box')
        plt.axes().set_axis_off()
        
        #出力された地図を保存
#        plt.savefig("Trip_Expenditure_CompleteGraph.svg", transparent = True)
#        plt.savefig("Trip_Expenditure_CompleteGraph.svg", transparent = True)
    
        #LPファイルを生成
        #model.update()
        #model.write("mkp.lp")

    except ImportError:
        print ("install 'networkx' and 'matplotlib' for plotting")


        
#自分で考えた指標を表示(与えた重みβ_jkと実際に縮んだ割合を比較) 


clf = linear_model.LinearRegression()

#距離の収縮率(ピッタリ重なるときが100)
xx1 = [0 for i in range(1081)]

#2点間の引き合う強さ
yy1 = [0 for i in range(1081)]

i = 0
for j in range(46):
    for k in range(j+1,47):
        a = ((X[j].X - X[k].X)*(X[j].X - X[k].X)+(Y[j].X - Y[k].X)*(Y[j].X - Y[k].X))/((x[j]-x[k])*(x[j]-x[k])+(y[j]-y[k])*(y[j]-y[k]))
        if a > 1:
            xx1[i] =0
        else:
            xx1[i] = (1-a)*100
        yy1[i] = (1/summ[j] + 1/summ[k])*v[j,k]*c[j,k]
        i += 1
        
#リストに変換
xx1 = np.reshape(xx1,(-1,1))

#fitさせる
clf.fit(xx1, yy1)

#figureサイズの設定
fig = plt.figure(figsize =(8, 8))

#直線の式を書き込む
#ax = fig.add_subplot(111)
#ax.text(60, 0.13, "{0}".format(round(clf.coef_[0],4))+"x+"+"{0}".format(round(clf.intercept_,4)), size = 20, color = "red")

#散布図の表示
plt.scatter(xx1,yy1)

#回帰直線の表示
plt.plot(xx1, clf.predict(xx1),color = "red")

#x軸，y軸の範囲
plt.xlim(0,100)
plt.ylim(0,1)

#x軸，y軸の名前
plt.xlabel('shrinkage ratio of distances betwen each pair of prefectures', fontsize=15)
plt.ylabel('weight between each pair of prefectures', fontsize=15)

#画像を保存
#plt.savefig("Trip_Expenditure_CompleteGraph_sokan.png", transparent = True)
#plt.savefig("homon_trip_hiyou_sokan.svg", transparent = True)
plt.show()

#相関係数を算出
xx1 = np.reshape(xx1,(-1))
s1=pd.Series(xx1)
s2=pd.Series(yy1)
res=s1.corr(s2)
print(res)


# 距離の縮み具合と重みとの比較

## モデルがちゃんと動いているかを確かめるためのWeber問題
#
## In[51]:
#
#
##調査したい県の番号(0-46)
#p = 26
#
##最適解の値
#given_xplace = [0 for i in range(47)]
#given_yplace = [0 for i in range(47)]
#for i in range(47):
#    given_xplace[i] = X[i].X
#    given_yplace[i] = Y[i].X
#
##調査したい県の最適解の値
#given_x = X[p].X
#given_y = Y[p].X
#
##調査したい県の値を抜く
#given_xplace.pop(p)
#given_yplace.pop(p)
#
##結果を表示
#for i in range(46):
#    #print(i,df2.iat[i,0],x[i],X[i].X,y[i], Y[i].X)
#    print(i,given_xplace[i],given_yplace[i])
#
#
## In[52]:
#
#
#def weber(I,J,x1,y1,w1,w2,x,y):
#    """weber: model for solving the single source weber problem using soco.
#    Parameters:
#        - I,J: 都道府県の集合
#        - x1:　最適解のx座標
#        - y1: 最適解のy座標
#        - w1: トリップ量と移動費用の重み
#        - w2: 自分の県との重み
#        - x: 各都道府県の元のx座標
#        - y: 各都道府県の元のy座標
#    Returns a model, ready to be solved.
#    """
#    
#    model = Model("weber")
#    X,Y,z2,xaux3,yaux3,z3,xaux4,yaux4 = {},{},{},{},{},{},{},{}
#    #Weber点の座標
#    X = model.addVar(lb=-GRB.INFINITY, vtype="C", name="X")
#    Y = model.addVar(lb=-GRB.INFINITY, vtype="C", name="Y")
#    
#    #提案モデル最適解における座標とweber点の距離
#    for i in I:
#        z2[i] = model.addVar(vtype="C", name="z(%s)"%(i))    
#        xaux3[i] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="xaux(%s)"%(i))
#        yaux3[i] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="yaux(%s)"%(i))    
#    model.update()
#    
#    for i in I:
#        model.addConstr(xaux3[i]*xaux3[i] + yaux3[i]*yaux3[i] <= z2[i]*z2[i], "MinDist(%s)"%(i))
#        model.addConstr(xaux3[i] == (x1[i]-X), "xAux(%s)"%(i))
#        model.addConstr(yaux3[i] == (y1[i]-Y), "yAux(%s)"%(i))
#   
#    #Weber点と元の都道府県の位置との距離
#    z3[p] = model.addVar(vtype="C", name="z(%s)"%(p))    
#    xaux4[p] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="xaux4(%s)"%(p))
#    yaux4[p] = model.addVar(lb=-GRB.INFINITY, vtype="C", name="yaux4(%s)"%(p))    
#    model.update()
#    
#    model.addConstr(xaux4[p]*xaux4[p] + yaux4[p]*yaux4[p] <= z3[p]*z3[p], "MinDist1(%s)"%(p))
#    model.addConstr(xaux4[p] == (x[p]-X), "xAux4(%s)"%(p))
#    model.addConstr(yaux4[p] == (y[p]-Y), "yAux4(%s)"%(p))
#
#    model.setObjective(quicksum(w1[i]*z2[i] for i in I) + w2*z3[p]*1.25, GRB.MINIMIZE)
#
#    model.update()
#    model.__data = X,Y,z2,z3
#    return model
#
#
#def make_data(summ,v,w):
#    I = range(46)
#    J = range(1)
#    x1,y1,w1,w2 = {},{},{},{}
#    
#    for i in I:
#        x1[i] = given_xplace[i]
#        y1[i] = given_yplace[i]
#        if p <= i:
#            w1[i] = (v[p,i+1]/summ[p]+v[p,i+1]/summ[i+1])*c[p,i+1]
#        elif p > i:
#            w1[i] = (v[i,p]/summ[p]+v[i,p]/summ[i])*c[i,p]
#    
#    #元の都道府県との重み
#    w2 = w[p]
#    return I,J,x1,y1,w1,w2,x,y
#
#                
#if __name__ == "__main__":
#    I,J,x1,y1,w1,w2,x,y = make_data(summ,v,w)
#   
#    model = weber(I,J,x1,y1,w1,w2,x,y)
#    model.optimize()
#    
#    X,Y,z2,z3 = model.__data
#    print("Optimal value=",model.ObjVal)
#    print ("Selected position:",)
#    print ("\t",(round(X.X),round(Y.X)))
#    print ("Solution:")
#    print ("%s\t%8s" % ("i","z2[i]"))
#    for i in I:
#        print ("%s\t%8g" % (i,z2[i].X))
#        
#    try: #解をnetworkx上に表示
#        #最適解の都道府県のグラフ
#        G = NX.Graph()
#        #weber点
#        G1 = NX.Graph()
#        #weber点と最適解を結ぶ点
#        G2 = NX.Graph()
#        #元の都道府県の位置
#        G3 = NX.Graph()
#        
#        pylab.figure(figsize=(35, 35))
#        
#        #対象としている都道府県も含めてうまく描画
#        for i in I:
#            G.add_nodes_from(["{0}".format(df1.iat[i,0])])
#            G2.add_nodes_from(["{0}".format(df1.iat[i,0])])
#        
#        G.remove_node("{0}".format(df1.iat[p,0]))
#        G2.remove_node("{0}".format(df1.iat[p,0]))
#        G2.add_node("最適解の位置")
#        G.add_node("最適解の位置")
#        G3.add_node("元の位置")
#        
#        G1.add_nodes_from(["{0}".format(df1.iat[p,0])])
#        G2.add_nodes_from(["{0}".format(df1.iat[p,0])])
#        
#        position = {}
#        position1 = {}
#        position2 = {}
#        position3 = {}
#        
#        for i in range(p):
#            position["{0}".format(df1.iat[i,0])] = (y1[i],x1[i])
#            position2["{0}".format(df1.iat[i,0])] = (y1[i],x1[i])
#        for i in range(p,46):
#            position["{0}".format(df1.iat[i+1,0])] = (y1[i],x1[i])
#            position2["{0}".format(df1.iat[i+1,0])] = (y1[i],x1[i])
#        
#        
#        position["最適解の位置"] = (given_y, given_x)
#        position2["最適解の位置"] = (given_y, given_x)
#        position1["{0}".format(df1.iat[p,0])] = ((Y.X),(X.X))
#        position2["{0}".format(df1.iat[p,0])] = ((Y.X),(X.X))
#        position3["元の位置"] = (y[p],x[p])
#        G2.add_edge("最適解の位置", "{0}".format(df1.iat[p,0]))
#        
#        #線だけを表示(ノードを白にして消す)
#        NX.draw_networkx(G2,pos=position2,node_color="w")
#        
#        #点だけを表示
#        NX.draw_networkx(G3,pos=position3,node_color="blue",node_size=800, with_labels=True, font_family = 'IPAexGothic', font_size =30, alpha = 0.5,label = "元の位置")
#       
#        #都道府県の元の位置を表示
#        NX.draw_networkx(G, with_labels=True, font_family = 'IPAexGothic',pos=position, node_size=800,node_color="green", alpha = 0.5, font_size =30 ,label = "最適解の位置")
#        
#        #都道府県の移動後の位置を表示
#        NX.draw_networkx(G1, with_labels=True, font_family = 'IPAexGothic',pos=position1,node_size=800,node_color="red", font_size =30, label = "weber点の位置" )
#        
#        #凡例を描画
#        plt.legend(fontsize = 40)
#        
#        #描画時のアスペクト比を１対１に設定
#        plt.axes().set_aspect('equal','box')
#        plt.axes().set_axis_off()
#        
#        #タイトル表示
#        plt.title('weber問題', size = 60)
#        
#        #出力された地図を表示
#        #plt.savefig("MFWP_2019_4_12_houmon_trip.eps", transparent = True)
#    
#        model.update()
#        model.write("mkp1.lp")
#        
#    except ImportError:
#        print ("install 'networkx' and 'matplotlib' for plotting")
#        
#
#
## In[ ]:
#



