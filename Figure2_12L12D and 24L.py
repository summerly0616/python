# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 09:35:37 2021

@author: Huangting
"""

from scipy.integrate import odeint
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

#%% 使用 pandas 读取csv文件
file = pd.read_csv('Clockgenes_12L12D.csv', header = None, sep = ',') 
df = pd.DataFrame(file)
t_data = df[0]




#%% light input

def lightdark(t):
    if np.mod(t,24)>=12:
        L = 0
        D = 1
    else:
        L = 1
        D = 0  
    return L,D

#%% vector field of minimal model
def fun(y,t):
    L,D = lightdark(t)
    
    v1=4.6
    v1L=2.0
    v2A=2.27
    v2L=5.0
    k1L=0.53
    k1D=0.021  
    k2=0.35
    k3=0.72
    k4=0.05 
    k5=3.4
    p1=0.76
    p1L=0.42
    p2=1.01
    p4=1.01
    p9=0.3  
    p10c=0.23 
    p11=0.14 
    p13=0.63
    d1=0.48
    d2D=0.5
    d2L=0.29
    d3D=0.48
    d3L=0.78
    d4D=1.21
    d4L=0.38
    d5=0.2
    d9=5.2 
    d10L=0.3 
    d10c=3
    d10n=0.1
    d10d=0.3
    d11=6.6
    d11b=0.8
    K0=5.07
    K1=0.3
    K2=1.3
    K4=0.4   
    K5=0.62
    K5b=1.8

    m9=0.1  
    m10=0.8  
    m11=3.4  
    m12=0.1
    r1=0.6
    r2=1
    r3=0.1
    
    
    v3=1.5
    v4=2.47
    v4L=1.2
    v5=12.5
    v5L=7.56
    v6=  3.3
    k6=  1.5
    p3=0.64
    p5=0.51
    p6=   4
    p7L=   1.3
    p7D=   3.3
    p8=   0.2
    p12=0.13
    d6=  1.5
    d7=  0.4
    d8=  0.3
    d12=0.5
    K6 =0.46
    K7=5
    K7a=  1
    K7b=3.5
    K8=1.36
    K9=0.9
    K10=1.9
    K11=5
    K11b= 4.5  
    K12=0.4
    K13=3
    K14=  2.1
    K15=  5
    K16=  1
    m1=0.3
    m2=0.7
    m3=3.4
    m4=0.1
    m5=  0.5
    m6=   0.6
    m7=   0.1  
    m8=    1.3

    
    K13b = 10.5
    K13c = 1.1
    K7c=0.5
    
    MCL   = y[0]
    CL    = y[1]
    MP97 = y[2]
    P97  = y[3]
    MP51 = y[4]
    P51  = y[5]
    MEL  = y[6]
    EL   = y[7]
    MGI  = y[8]
    GI   = y[9]
    MR8  = y[10]
    R8   = y[11]
    LNK1 = y[12]
    RL   = y[13]
    EC   = y[14]
    COP1c= y[15]
    COP1n= y[16]
    COP1d= y[17]
    ZTL  = y[18]
    ZG   = y[19]    
    P    = y[20]
    K2a = 0.5
  
    z=np.array([(v1+v1L*L*P)/(1+(CL/K0)**2+(P97/K1)**2+(P51/K2)**2)*(GI/K2a)**2/(1+(GI/K2a)**2)-(k1L*L+k1D*D)*MCL,
                  (p1+p1L*L)*MCL-d1*CL,
                  (v2L*L*P+v2A)/(1+(P51/K4)**2+(EL/K5)**2+(CL/K5b)**2)-k2*MP97,
                  p2*MP97-(d2D*D+d2L*L)*P97,
                  v3*(RL/K7a)**2/(1+(RL/K7a)**2)*(GI/K7c)**2/(1+(GI/K7c)**2)/(1+(CL/K6)**2+(P51/K7)**2+(EC/K7b)**2)-k3*MP51,
                  p3*MP51-(m1+m2*D)*P51*(ZTL+ZG)-(d3D*D+d3L*L)*P51,
                  (v4+v4L*L)/(1+(CL/K8)**2+(P51/K9)**2+(EL/K10)**2+(EC/K11)**2)*(R8/K11b)**2/(1+(R8/K11b)**2)-k4*MEL,
                  p4*MEL-(d4D*D+d4L*L)*EL,
                  (v5L*L*P+v5)/(1+(CL/K12)**2+(EC/K13)**2+(P97/K13b)**2+(P51/K13c)**2)-k5*MGI,
                  p5*MGI-m3*L*ZTL*GI+m4*D*ZG-d5*GI,
                  v6/(1+(P97/K14)**2+(P51/K15)**2)-k6*MR8,
                  p6*MR8-m5*R8*LNK1+m6*RL-d6*R8,
                  (p7L*L+p7D*D)*(EC/K16)**2/(1+(EC/K16)**2)-m7*R8*LNK1+m8*RL-d7*LNK1,
                  p8*R8*LNK1-d8*RL,
                  p9*EL**2-m9*EC*COP1n-m10*EC*COP1d-d9*EC,
                  p10c-r1*COP1c-(d10L*L+d10c)*COP1c,
                  r1*COP1c-(r2*L*P+r3)*COP1n-d10n*COP1n,
                  (r2*L*P+r3)*COP1n-d10d*COP1d,
                  p11-m11*L*ZTL*GI+m12*D*ZG-d11*ZTL,
                  m11*L*ZTL*GI-m12*D*ZG-d11b*ZG,
                  (p12-p13*P)*D-d12*P*L])
    

    return z





tspan=np.array([0,240]);
h=0.01
t=np.arange(tspan[0],tspan[1]+h,h)
y0 = np.array([1.52318295e-01,3.39972549e-01,5.95471718e-01,2.15006756e+00,
  1.35930447e+00,1.09418276e+00,3.15492116e-01,8.46288136e-01,
  2.16044375e+00, 5.47355844e+00,1.03846426e+00,5.46457762e-01,
  5.43789886e+01,1.92123958e+01,4.08696884e-02,5.89743590e-02,
  1.76881577e-01,5.89341476e-02,5.55050747e-03,1.29566879e-01,
  7.78654796e-12])



y = odeint(fun, y0, t)

CL_data = df[1]
P97_data = df[2]
P51_data = df[3]
EL_data = df[4]
GI_data = df[5]
RL_data = df[6]


CLse_data = df[8]
P97se_data = df[9]
P51se_data = df[10]
ELse_data = df[11]
GIse_data = df[12]
RLse_data = df[13]


font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,}
font1 = {'family' : 'Times New Roman',
              'weight' : 'normal',
              'size'   : 18,}


plt.subplot(261)

x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,0]-min(y[16800:,0]))/(max(y[16800:,0])-min(y[16800:,0])),color='k',label='SlCL(Sim.)')
plt.scatter(t_data,CL_data,marker='o',color='blue',label='SlCCA1(Exp.)')
plt.errorbar(t_data,CL_data,CLse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,CL_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative transcriptional level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,0]-CL_data[i])**2
    
# print(MSE)



plt.subplot(262)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,2]-min(y[16800:,2]))/(max(y[16800:,2])-min(y[16800:,2])),color='k',label='SlP97(Sim.)')
plt.scatter(t_data,P97_data,marker='o',color='blue',label='SlPRR9(Exp.)')
plt.errorbar(t_data,P97_data,P97se_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,P97_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative PRR9/PRR7 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,2]-P97_data[i])**2
    
# print(MSE)




plt.subplot(263)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,4]-min(y[16800:,4]))/(max(y[16800:,4])-min(y[16800:,4])),color='k',label='SlP51(Sim.)')
plt.scatter(t_data,P51_data,marker='o',color='blue',label='SlPRR5(Exp.)')
plt.errorbar(t_data,P51_data,P51se_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,P51_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative PRR5/TOC1 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100+16000,4]-P51_data[i])**2
    
# print(MSE)




plt.subplot(264)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,6]-min(y[16800:,6]))/(max(y[16800:,6])-min(y[16800:,6])),color='k',label='SlEL(Sim.)')
plt.scatter(t_data,EL_data,marker='o',color='blue',label='SlELF4(Exp.)')
plt.errorbar(t_data,EL_data,ELse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,EL_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative ELF4/LUX mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,6]-EL_data[i])**2
    
# print(MSE)





plt.subplot(265)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,8]-min(y[16800:,8]))/(max(y[16800:,8])-min(y[16800:,8])),color='k',label='SlGI(Sim.)')
plt.scatter(t_data,GI_data,marker='o',color='blue',label='SlGI(Exp.)')
plt.errorbar(t_data,GI_data,GIse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,GI_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative GI mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)


# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,8]-GI_data[i])**2
    
# print(MSE)



plt.subplot(266)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,10]-min(y[16800:,10]))/(max(y[16800:,10])-min(y[16800:,10])),color='k',label='SlRL(Sim.)')
plt.scatter(t_data,RL_data,marker='o',color='blue',label='SlRVE8(Exp.)')
plt.errorbar(t_data,RL_data,RLse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,RL_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative RVE8 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,10]-RL_data[i])**2
    
# print(MSE)

#%% 使用 pandas 读取csv文件
file = pd.read_csv('Clockgenes_24L.csv', header = None, sep = ',') 
df = pd.DataFrame(file)
t_data = df[0]




#%% light input
def lightdark1(t):
    if t>44:
        L = 1
        D = 0
    elif np.mod(t,24)>=12:
        L = 0
        D = 1
    else:
        L = 1
        D = 0  
    return L,D

#%% vector field of minimal model
def fun(y,t):
    L,D = lightdark1(t)
    
    v1=4.6
    v1L=2.0
    v2A=2.27
    v2L=5.0
    k1L=0.53
    k1D=0.021
    k2=0.35
    k3=0.72
    k4=0.10
    k5=3.4
    p1=0.76
    p1L=0.42
    p2=1.01
    p4=1.01
    p9=0.3  
    p10c=0.23 
    p11=0.14 
    p13=0.63
    d1=0.48
    d2D=0.5
    d2L=0.29
    d3D=0.48
    d3L=0.78
    d4D=1.21
    d4L=0.38
    d5=0.2
    d9=5.2 
    d10L=0.3 
    d10c=3
    d10n=0.1
    d10d=0.3
    d11=6.6
    d11b=0.8
    K0=5.07
    K1=0.3
    K2=1.3
    K4=0.4  #取0.28时可以发生持续振荡
    K5=0.62
    K5b=1.8

    m9=0.1  
    m10=0.8  
    m11=3.4  
    m12=0.1
    r1=0.6
    r2=1
    r3=0.1
    
    
    v3=1.5
    v4=2.47
    v4L=1.2
    v5=12.5
    v5L=7.56
    v6=  3.3
    k6=  1.5
    p3=0.64
    p5=0.51
    p6=   4
    p7L=   1.3
    p7D=   3.3
    p8=   0.2
    p12=0.13
    d6=  1.5
    d7=  0.4
    d8=  0.3
    d12=0.5
    K6 =0.46
    K7=5
    K7a=  1
    K7b=3.5
    K8=1.36
    K9=0.1
    K10=1.9
    K11=5
    K11b= 4.5
    K12=0.4
    K13=3
    K14=  2.1
    K15=  5
    K16=  1
    m1=0.3
    m2=0.7
    m3=3.4
    m4=0.1
    m5=  0.3
    m6=   0.6
    m7=   0.1  
    m8=    1.3

    
    K13b = 10.5
    K13c = 1.1
    K7c=0.5
    
    MCL   = y[0]
    CL    = y[1]
    MP97 = y[2]
    P97  = y[3]
    MP51 = y[4]
    P51  = y[5]
    MEL  = y[6]
    EL   = y[7]
    MGI  = y[8]
    GI   = y[9]
    MR8  = y[10]
    R8   = y[11]
    LNK1 = y[12]
    RL   = y[13]
    EC   = y[14]
    COP1c= y[15]
    COP1n= y[16]
    COP1d= y[17]
    ZTL  = y[18]
    ZG   = y[19]    
    P    = y[20]
  
    K2a = 0.5
  
    z=np.array([(v1+v1L*L*P)/(1+(CL/K0)**2+(P97/K1)**2+(P51/K2)**2)*(GI/K2a)**2/(1+(GI/K2a)**2)-(k1L*L+k1D*D)*MCL,
                  (p1+p1L*L)*MCL-d1*CL,
                  (v2L*L*P+v2A)/(1+(P51/K4)**2+(EL/K5)**2+(CL/K5b)**2)-k2*MP97,
                  p2*MP97-(d2D*D+d2L*L)*P97,
                  v3*(RL/K7a)**2/(1+(RL/K7a)**2)*(GI/K7c)**2/(1+(GI/K7c)**2)/(1+(CL/K6)**2+(P51/K7)**2+(EC/K7b)**2)-k3*MP51,
                  p3*MP51-(m1+m2*D)*P51*(ZTL+ZG)-(d3D*D+d3L*L)*P51,
                  (v4+v4L*L)/(1+(CL/K8)**2+(P51/K9)**2+(EL/K10)**2+(EC/K11)**2)*(R8/K11b)**2/(1+(R8/K11b)**2)-k4*MEL,
                  p4*MEL-(d4D*D+d4L*L)*EL,
                  (v5L*L*P+v5)/(1+(CL/K12)**2+(EC/K13)**2+(P97/K13b)**2+(P51/K13c)**2)-k5*MGI,
                  p5*MGI-m3*L*ZTL*GI+m4*D*ZG-d5*GI,
                  v6/(1+(P97/K14)**2+(P51/K15)**2)-k6*MR8,
                  p6*MR8-m5*R8*LNK1+m6*RL-d6*R8,
                  (p7L*L+p7D*D)*(EC/K16)**2/(1+(EC/K16)**2)-m7*R8*LNK1+m8*RL-d7*LNK1,
                  p8*R8*LNK1-d8*RL,
                  p9*EL**2-m9*EC*COP1n-m10*EC*COP1d-d9*EC,
                  p10c-r1*COP1c-(d10L*L+d10c)*COP1c,
                  r1*COP1c-(r2*L*P+r3)*COP1n-d10n*COP1n,
                  (r2*L*P+r3)*COP1n-d10d*COP1d,
                  p11-m11*L*ZTL*GI+m12*D*ZG-d11*ZTL,
                  m11*L*ZTL*GI-m12*D*ZG-d11b*ZG,
                  (p12-p13*P)*D-d12*P*L])
    

    return z






tspan=np.array([0,240]);
h=0.01
t=np.arange(tspan[0],tspan[1]+h,h)
#y0 = np.array([1.94479216e-01,0.1,3.19894883e-01,0.1,1.23615127e+00,0.1,0.1,0.1,0.1,0.1,1.63254020e+00,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
y0 = np.array([1.52318295e-01,3.39972549e-01,5.95471718e-01,2.15006756e+00,
 1.35930447e+00,1.09418276e+00,3.15492116e-01,8.46288136e-01,
 2.16044375e+00, 5.47355844e+00,1.03846426e+00,5.46457762e-01,
 5.43789886e+01,1.92123958e+01,4.08696884e-02,5.89743590e-02,
 1.76881577e-01,5.89341476e-02,5.55050747e-03,1.29566879e-01,
 7.78654796e-12])



y = odeint(fun, y0, t)

CL_data = df[1]
P97_data = df[2]
P51_data = df[3]
EL_data = df[4]
GI_data = df[5]
RL_data = df[6]


CLse_data = df[8]
P97se_data = df[9]
P51se_data = df[10]
ELse_data = df[11]
GIse_data = df[12]
RLse_data = df[13]


font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 18,}


plt.subplot(267)

x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,0]-min(y[16800:,0]))/(max(y[16800:,0])-min(y[16800:,0])),color='k',label='SlCL(Sim.)')
plt.scatter(t_data,CL_data,marker='o',color='blue',label='SlCCA1(Exp.)')
plt.errorbar(t_data,CL_data,CLse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,CL_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative transcriptional level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,0]-CL_data[i])**2
    
# print(MSE)



plt.subplot(268)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,2]-min(y[16800:,2]))/(max(y[16800:,2])-min(y[16800:,2])),color='k',label='SlP97(Sim.)')
plt.scatter(t_data,P97_data,marker='o',color='blue',label='SlPRR9(Exp.)')
plt.errorbar(t_data,P97_data,P97se_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,P97_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative PRR9/PRR7 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,2]-P97_data[i])**2
    
# print(MSE)




plt.subplot(269)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,4]-min(y[16800:,4]))/(max(y[16800:,4])-min(y[16800:,4])),color='k',label='SlP51(Sim.)')
plt.scatter(t_data,P51_data,marker='o',color='blue',label='SlPRR5(Exp.)')
plt.errorbar(t_data,P51_data,P51se_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,P51_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative PRR5/TOC1 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100+16000,4]-P51_data[i])**2
    
# print(MSE)




plt.subplot(2,6,10)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,6]-min(y[16800:,6]))/(max(y[16800:,6])-min(y[16800:,6])),color='k',label='SlEL(Sim.)')
plt.scatter(t_data,EL_data,marker='o',color='blue',label='SlELF4(Exp.)')
plt.errorbar(t_data,EL_data,ELse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,EL_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative ELF4/LUX mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,6]-EL_data[i])**2
    
# print(MSE)





plt.subplot(2,6,11)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,8]-min(y[16800:,8]))/(max(y[16800:,8])-min(y[16800:,8])),color='k',label='SlGI(Sim.)')
plt.scatter(t_data,GI_data,marker='o',color='blue',label='SlGI(Exp.)')
plt.errorbar(t_data,GI_data,GIse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,GI_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative GI mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)


# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,8]-GI_data[i])**2
    
# print(MSE)



plt.subplot(2,6,12)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,10]-min(y[16800:,10]))/(max(y[16800:,10])-min(y[16800:,10])),color='k',label='SlRL(Sim.)')
plt.scatter(t_data,RL_data,marker='o',color='blue',label='SlRVE8(Exp.)')
plt.errorbar(t_data,RL_data,RLse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,RL_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
# plt.ylabel('Relative RVE8 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)


plt.show()