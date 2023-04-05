import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
import statsmodels.api as sm
import streamlit as st

print('Setup complete')

st.title("Real Estate Dataset")
df = pd.read_csv('data.csv')
st.dataframe(df)

st.title("House price")
col1, col2 = st.columns(2)
with col1:
    
    st.dataframe(df[['CRIM','ZN','INDUS']])
with col2:
    st.header('CRIM , ZN and INDUS' )
    fig, ax = plt.subplots()
    ax.scatter(df.iloc[:,0],df.iloc[:,1])
    st.pyplot(fig)

    
col1, col2 = st.columns(2)
with col1:
    
    st.dataframe(df[['CHAS','NOX']])
with col2:
    st.header('CHAS AND NOX')
    fig, ax = plt.subplots()
    ax.scatter(df.iloc[:,2],df.iloc[:,3])
    st.pyplot(fig)
st.title("histogram, barchart,areachart,and linechart")
t=df["MEDV"].value_counts()
st.write(t)
st.bar_chart(t)
st.area_chart(df.iloc[:,3])
st.line_chart(df.iloc[:,0])

st.area_chart(df.iloc[:,4])
st.line_chart(df.iloc[:,1])

fig,ax=plt.subplots()
ax.hist(df.iloc[:,1],color="green",bins=12)
st.line_chart(df.iloc[:,1])
st.pyplot(fig)


t=df["DIS"].value_counts()
st.write(t)
st.bar_chart(t)
st.area_chart(df.iloc[:,5])
st.line_chart(df.iloc[:,2])
st.area_chart(df.iloc[:,3])
st.line_chart(df.iloc[:,4])

fig,ax=plt.subplots()
ax.hist(df.iloc[:,0],color="red",bins=12)
st.line_chart(df.iloc[:,1])
st.pyplot(fig)
import matplotlib as plt




t=df["AGE"].value_counts()
st.write(t)
st.bar_chart(t)
st.area_chart(df.iloc[:,5])
st.line_chart(df.iloc[:,2])
st.area_chart(df.iloc[:,3])
st.line_chart(df.iloc[:,4])

fig,ax=plt.subplots()
ax.hist(df.iloc[:,3],color="orange",bins=12)
st.line_chart(df.iloc[:,2])
st.pyplot(fig)

t=df["PTRATIO"].value_counts()
st.write(t)
st.bar_chart(t)
st.area_chart(df.iloc[:,0])
st.line_chart(df.iloc[:,2])
st.area_chart(df.iloc[:,3])
st.line_chart(df.iloc[:,4])

fig,ax=plt.subplots()
ax.hist(df.iloc[:,0],color="orange",bins=12)
st.line_chart(df.iloc[:,1])
st.pyplot(fig)

st.write(t)
st.bar_chart(t)
st.area_chart(df.iloc[:,5])
st.line_chart(df.iloc[:,2])
st.area_chart(df.iloc[:,3])
st.line_chart(df.iloc[:,4])

fig,ax=plt.subplots()
ax.hist(df.iloc[:,0],color="orange",bins=12)
st.line_chart(df.iloc[:,1])
st.pyplot(fig)

t=df["DIS"].value_counts()
st.write(t)
st.bar_chart(t)
st.area_chart(df.iloc[:,5])
st.line_chart(df.iloc[:,2])
st.area_chart(df.iloc[:,3])
st.line_chart(df.iloc[:,4])

fig,ax=plt.subplots()
ax.hist(df.iloc[:,0],color="orange",bins=12)
st.line_chart(df.iloc[:,1])
st.pyplot(fig)


import matplotlib as plt
labels=["DIS","AGE","RM"]
fig,ax=plt.subplots()
ax.pie(t,labels=labels,autopct="%1.2f%%")
st.pyplot(fig)




df = px.data.age()
fig = px.pie(df, values='AGE', names='House price')
fig.show()


st.write(t)
st.header('Box Plot')
col1,col2,col3,col4 =st.columns(4)
with col1:
    fig, ax = plt.subplots()
    ax.boxplot(df.iloc[:,0])
    st.pyplot(fig)

with col1:
    fig,ax = plt.subplots()
    ax.boxplot(df.iloc[:,0])
    st.pyplot(fig)
with col2:
        fig, ax = plt.subplots()
        ax.boxplot(df.iloc[:,1])
        st.pyplot(fig)
with col3:
    fig, ax = plt.subplots()
    ax.boxplot(df.iloc[:,2])
    st.pyplot(fig)
with col4:
    fig, ax = plt.subplots()
    ax.boxplot(df.iloc[:,3])
    st.pyplot(fig)

import matplotlib.pyplot as plt
import streamlit as st

labels=["DIS","AGE","RM"]
t = [20, 30, 50]

fig,ax=plt.subplots()
ax.pie(t,labels=labels,autopct="%1.2f%%")
st.pyplot(fig)


sns.boxplot(data=boston_df, x='age_label', y='MEDV')
plt.title('Median value of owner-occupied homes in different age groups')
plt.ylabel('homes')
plt.xlabel('age group')
st.pyplot(fig)

sns.histplot(data=boston_df, x='PTRATIO', bins=20)
st.pyplot(fig)




import matplotlib.pyplot as plt
t = [7, 10, 8]  
labels = ["DIS", "AGE", "RM"]
fig, ax = plt.subplot(211)
ax.pie(t, labels=labels, autopct="%1.2f%%")
plt.subplot(211)

