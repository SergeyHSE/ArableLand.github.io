# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:01:18 2023

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import statsmodels.stats.diagnostic as dg
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
np.set_printoptions(suppress=True)
from statsmodels.compat import lzip

df = pd.read_excel(r'C:\Users\User\Documents\pyton-projects\spider\final.xlsx')

print(df)

summury = df.describe()
summury.to_excel('summury.xlsx', sheet_name='list0', index= False)
summury.to_csv('summury.csv')
print (summury)

plt.hist(df['gdp_p_person'], color = 'blue', edgecolor = 'black')
plt.hist(df['logistic_perf'], color = 'blue', edgecolor = 'black')
plt.hist(df['land_ha_person'], color = 'blue', edgecolor = 'black')
plt.hist(df['urban_pop'], color = 'blue', edgecolor = 'black')
plt.hist(df['crop_output_ha'], color = 'blue', edgecolor = 'black')

everageLand = summury.iloc[1]
everageLand.to_csv('everageLand.csv')
everageLand.to_excel('everageLand.xlsx', sheet_name='list0', index= False)

reg = 'land_price_ha~gdp_p_person+logistic_perf+land_ha_person+urban_pop+crop_output_ha'

df.plot(kind='scatter', x = 'gdp_p_person', y = 'land_price_ha')
df.plot(kind='scatter', x = 'logistic_perf', y = 'land_price_ha')
df.plot(kind='scatter', x = 'land_ha_person', y = 'land_price_ha')
df.plot(kind='scatter', x = 'urban_pop', y = 'land_price_ha')
df.plot(kind='scatter', x = 'crop_output_ha', y = 'land_price_ha')

regoutput = smf.ols(reg,df).fit()
print(regoutput.summary())

#Стьюдентезированые остатки, выбросы
stud_res = regoutput.outlier_test()
df['student_resid'] = stud_res['student_resid']
df[df['student_resid'] > 2].count()
Vertical_outlier=df[df['student_resid'] > 2]
df.plot(kind='scatter', x = 'land_price_ha', y = 'student_resid')

influence = regoutput.get_influence()
cooks = influence.cooks_distance
df['cooks_distance'] = cooks[0]
df[df['cooks_distance'] > 0.008565].count()
df.plot(kind='scatter', x = 'land_price_ha', y = 'cooks_distance')

#Удаление остатков
df_noresid=df.drop(index=df[df['student_resid']>2].index)
df_noresid=df_noresid.drop(index=df_noresid[df_noresid['cooks_distance']>0.008565].index)

reg_noresid_output = smf.ols(reg,df_noresid).fit()
print(reg_noresid_output.summary())

#ТЕст Рамсея
reset = dg.linear_reset(regoutput, power=2, test_type='fitted',use_f=True)
print(np.round(reset.fvalue,4))
print(np.round(reset.pvalue,4))

#мультиколлинеарность

corr_data = df[['land_price_ha','gdp_p_person','logistic_perf','land_ha_person','urban_pop','crop_output_ha']]

corr= corr_data.corr()
corr.to_csv('Correlation.csv')

df['Lngdp'] = np.log(df['gdp_p_person'])
reg2 = 'land_price_ha~Lngdp+land_ha_person+urban_pop+crop_output_ha'
reg2output = smf.ols(reg2,df).fit()
print(reg2output.summary())
print(df)
reset = dg.linear_reset(reg2output, power=2, test_type='fitted',use_f=True)
print(np.round(reset.fvalue,4))
print(np.round(reset.pvalue,4))

#Гетероскедастичность
residuals = regoutput.resid**2
breush = sm.OLS(residuals,corr_data).fit()
breush.summary()

names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']

bp2_test = het_breuschpagan(reg2output.resid,reg2output.model.exog)
print(bp2_test)
lzip (names, bp2_test)
wh2_test = het_white(reg2output.resid,reg2output.model.exog)
print(wh2_test)
lzip (names, wh2_test)

#эендогенность




