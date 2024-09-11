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
file = pd.read_csv('photosyntheticgenes_12L12D.csv', header = None, sep = ',') 
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
    k4=0.1  
    k5=3.4
    p1=0.76
    p1L=0.42
    p2=1.01
    p4=1.01
    p9=0.3  
    p10c=0.23 
    p11=0.14 
    p12a=0.63
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
    
    
    v7=12.6
    v8=10.8
    v9=6.7
    v10=9.6
    k7=1.6
    k8=0.22
    k9=0.15
    k10=0.13
    p13=0.68  #
    p14=0.98 #
    p15=0.16
    p16=0.56
    d13=0.86
    d14=0.35
    d15=0.46
    d16=0.48
    K17=0.5
    K18=0.8
    K19=1.5
    K19a=1.5
    K20=0.5
    K21=1.12
    K20a=2.5
    # K21=0.12
    K22=4.8
    
    
    MLhcb1 = y[21]
    Lhcb1  = y[22]
    MpsbA  = y[23]
    psbA   = y[24]
    MRbcS1 = y[25]
    RbcS1  = y[26]
    MatpA  = y[27]
    atpA   = y[28]
  
    K23= 0.1
  
    z=np.array([(v1+v1L*L*P)/(1+(CL/K0)**2+(P97/K1)**2+(P51/K2)**2)-(k1L*L+k1D*D)*MCL,
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
                 (p12-p12a*P)*D-d12*P*L,
                 v7*(CL/K17)**2/(1+(CL/K17)**2)/(1+(GI/K18)**2)-k7*MLhcb1,
                 p13*MLhcb1-d13*Lhcb1,
                 v8/(1+(CL/K19)**2+(GI/K19a)**2)-k8*MpsbA,
                 p14*MpsbA-d14*psbA,
                 v9/(1+(CL/K20)**2+(GI/K20a)**2)-k9*MRbcS1,
                 p15*MRbcS1-d15*RbcS1,
                 v10*(GI/K22)**2/(1+(GI/K22)**2)*(P97/K23)**2/(1+(P97/K23)**2)/(1+(CL/K21)**2)-k10*MatpA,
                 p16*MatpA-d16*atpA])
    

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
 7.78654796e-12,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])



y = odeint(fun, y0, t)

lhcb1_data = df[1]
psbA_data = df[2]
RbcS1_data = df[3]
atpA_data = df[4]



lhcb1se_data = df[6]
psbAse_data = df[7]
RbcS1se_data = df[8]
atpAse_data = df[9]


font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 16,}


plt.subplot(241)

x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.scatter(t_data,lhcb1_data,marker='o',color='blue',label='SlLhcb1(Exp.)')
plt.errorbar(t_data,lhcb1_data,lhcb1se_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.plot(t[16800:]-168,y[16800:,21]/1.3,color='k',label='SlLhcb1(Sim.)')
plt.xlim(0,48)
plt.ylim(0,5)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,lhcb1_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative Lhcb1 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,0]-CL_data[i])**2
    
# print(MSE)



plt.subplot(242)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 20*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,23]-min(y[16800:,23]))/(max(y[16800:,23])-min(y[16800:,23]))*17,color='k',label='SlpsbA(Sim.)')
plt.scatter(t_data,psbA_data,marker='o',color='blue',label='SlpsbA(Exp.)')
plt.errorbar(t_data,psbA_data,psbAse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,20)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,psbA_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative psbA mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,2]-P97_data[i])**2
    
# print(MSE)




plt.subplot(243)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 2*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,25]-min(y[16800:,25]))/(max(y[16800:,25])-min(y[16800:,25]))*1.5,color='k',label='SlRbcS1(Sim.)')
plt.scatter(t_data,RbcS1_data,marker='o',color='blue',label='SlRbcS1(Exp.)')
plt.errorbar(t_data,RbcS1_data,RbcS1se_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,2)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,RbcS1_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative RbcS1 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100+16000,4]-P51_data[i])**2
    
# print(MSE)




plt.subplot(244)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='silver')
plt.fill_between(x1,y1,y2,facecolor='silver')
plt.fill_between(x2,y1,y2,facecolor='silver')
plt.plot(t[16800:]-168,(y[16800:,27]-min(y[16800:,27]))/(max(y[16800:,27])-min(y[16800:,27]))*1.2,color='k',label='SlatpA(Sim.)')
plt.scatter(t_data,atpA_data,marker='o',color='blue',label='SlatpA(Exp.)')
plt.errorbar(t_data,atpA_data,atpAse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,atpA_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative atpA mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,6]-EL_data[i])**2
    
# print(MSE)

#%% 使用 pandas 读取csv文件
file = pd.read_csv('photosyntheticgenes_24L.csv', header = None, sep = ',') 
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
    p12a=0.63
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
    K9=0.1
    K10=1.9
    K11=5
    K11b= 8.5
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
  
    v7=12.6
    v8=5.8
    v9=6.7
    v10=9.6
    k7=1.6
    k8=0.52
    k9=0.15
    k10=0.63
    p13=0.68  
    p14=0.98 
    p15=0.16
    p16=0.56
    d13=0.86
    d14=0.35
    d15=0.46
    d16=0.48
    K17=0.5
    K18=0.8
    K19=0.3
    K20=0.5
    K21=1.12
    
    
    MLhcb1 = y[21]
    Lhcb1  = y[22]
    MpsbA  = y[23]
    psbA   = y[24]
    MRbcS1 = y[25]
    RbcS1  = y[26]
    MatpA  = y[27]
    atpA   = y[28]
  
    v7=12.6
    v8=10.8
    v9=6.7
    v10=9.6
    k7=1.6
    k8=0.22
    k9=0.15
    k10=0.13
    p13=0.68  
    p14=0.98 
    p15=0.16
    p16=0.56
    d13=0.86
    d14=0.35
    d15=0.46
    d16=0.48
    K17=0.5
    K18=0.8
    K19=1.5
    K19a=1.5
    K20=0.5
    K20a=2.5
    K21=1.12
    K22=4.8
    
    
    
    
    MLhcb1 = y[21]
    Lhcb1  = y[22]
    MpsbA  = y[23]
    psbA   = y[24]
    MRbcS1 = y[25]
    RbcS1  = y[26]
    MatpA  = y[27]
    atpA   = y[28]
  
    K23= 10.1
  
    z=np.array([(v1+v1L*L*P)/(1+(CL/K0)**2+(P97/K1)**2+(P51/K2)**2)-(k1L*L+k1D*D)*MCL,
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
                 (p12-p12a*P)*D-d12*P*L,
                 v7*(CL/K17)**2/(1+(CL/K17)**2)/(1+(GI/K18)**2)-k7*MLhcb1,
                 p13*MLhcb1-d13*Lhcb1,
                 v8/(1+(CL/K19)**2+(GI/K19a)**2)-k8*MpsbA,
                 p14*MpsbA-d14*psbA,
                 v9/(1+(CL/K20)**2+(GI/K20a)**2)-k9*MRbcS1,
                 p15*MRbcS1-d15*RbcS1,
                 v10*(GI/K22)**2/(1+(GI/K22)**2)*(P97/K23)**2/(1+(P97/K23)**2)/(1+(CL/K21)**2)-k10*MatpA,
                 p16*MatpA-d16*atpA])
    

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
 7.78654796e-12,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])



y = odeint(fun, y0, t)

lhcb1_data = df[1]
psbA_data = df[2]
RbcS1_data = df[3]
atpA_data = df[4]



lhcb1se_data = df[6]
psbAse_data = df[7]
RbcS1se_data = df[8]
atpAse_data = df[9]

font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,}
font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
                'size'   : 16,}


plt.subplot(245)

x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,21]-min(y[16800:,21]))/(max(y[16800:,21])-min(y[16800:,21]))*3,color='k',label='SlLhcb1(Sim.)')
plt.scatter(t_data,lhcb1_data,marker='o',color='blue',label='SlLhcb1(Exp.)')
plt.errorbar(t_data,lhcb1_data,lhcb1se_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,5)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,lhcb1_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative Lhcb1 mRNA level',font1)


x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,0]-CL_data[i])**2
    
# print(MSE)



plt.subplot(246)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 20*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,23]-min(y[16800:,23]))/(max(y[16800:,23])-min(y[16800:,23]))*17,color='k',label='SlpsbA(Sim.)')
plt.scatter(t_data,psbA_data,marker='o',color='blue',label='SlpsbA(Exp.)')
plt.errorbar(t_data,psbA_data,psbAse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,20)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,psbA_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative psbA mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100,2]-P97_data[i])**2
    
# print(MSE)




plt.subplot(247)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 2*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,25]-min(y[16800:,25]))/(max(y[16800:,25])-min(y[16800:,25]))*1.9+0.1,color='k',label='SlRbcS1(Sim.)')
plt.scatter(t_data,RbcS1_data,marker='o',color='blue',label='SlRbcS1(Exp.)')
plt.errorbar(t_data,RbcS1_data,RbcS1se_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,2)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,RbcS1_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative RbcS1 mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# MSE=0
# for i in range(17):
#     MSE = MSE+(y[i*100+16000,4]-P51_data[i])**2
    
# print(MSE)




plt.subplot(248)
x = np.linspace(-12,0,2)
x1 = np.linspace(12,24,2)
x2 = np.linspace(36,48,2)
y1 = 0*np.ones(len(x))
y2 = 1.5*np.ones(len(x))
plt.fill_between(x,y1,y2,facecolor='whitesmoke')
plt.fill_between(x1,y1,y2,facecolor='whitesmoke')
plt.fill_between(x2,y1,y2,facecolor='whitesmoke')
plt.plot(t[16800:]-168,(y[16800:,27]-min(y[16800:,27]))/(max(y[16800:,27])-min(y[16800:,27]))*1.2,color='k',label='SlatpA(Sim.)')
plt.scatter(t_data,atpA_data,marker='o',color='blue',label='SlatpA(Exp.)')
plt.errorbar(t_data,atpA_data,atpAse_data,lw = 1,ecolor='navy',elinewidth=2,ms=7,capsize=3)

plt.xlim(0,48)
plt.ylim(0,1.5)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.plot(t_data,atpA_data,linestyle = '-.',color='steelblue')
plt.xticks(fontproperties = 'Times New Roman', size = 18)
plt.yticks(fontproperties = 'Times New Roman', size = 18)
# plt.xlabel('Time(h)',font)
plt.ylabel('Relative atpA mRNA level',font1)

x_major_locator=MultipleLocator(12)
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)



plt.show()