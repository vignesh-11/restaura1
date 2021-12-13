#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -q keras
# !pip install -U scikit-learn
# !pip install scikit-learn-pipeline-utils
# !pip install shap
# get_ipython().system('pip install folium')
# get_ipython().system('pip install geopy')


# <h1>IMPORTING LIBRARIES</h1>

# In[2]:


from fpdf import FPDF

# Standard libraries
from datetime import date
import pandas as pd
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
from collections import Counter
from PIL import Image

# Viz libs
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.inset_locator import InsetPosition
import folium
from folium.plugins import HeatMap, FastMarkerCluster
# from wordcloud import WordCloud
import plotly.graph_objs as go
import plotly.offline as py

# Geolocation libs
# from geopy.geocoders import Nominatim

# Utils modules
# from custom_transformers import *
from viz_utils import *
# from ml_utils import *

# # ML libs
# from sklearn.model_selection import train_test_split
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# import lightgbm as lgb
# import shap


# <H1>USER INPUTS</H1>

# In[3]:


user_location = "Banashankari"
user_rest_type = ['Cafe', 'Casual Dining']
user_cuisine = ['North Indian', 'Chinese']
succ = 33

# <H1>READING DATA</H1>

# In[4]:


# Reading restaurants data
df_restaurants = pd.read_csv('zomato.csv')
df = pd.read_csv('zomato.csv')
# Changing the data type from approx_cost columns
df_restaurants['approx_cost'] = df_restaurants['approx_cost(for two people)'].astype(str).apply(lambda x: x.replace(',', ''))
df_restaurants['approx_cost'] = df_restaurants['approx_cost'].astype(float)

# Extracting the rate in a float column
df_restaurants['rate_num'] = df_restaurants['rate'].astype(str).apply(lambda x: x.split('/')[0])
while True:
    try:
        df_restaurants['rate_num'] = df_restaurants['rate_num'].astype(float)
        break
    except ValueError as e1:
        noise_entry = str(e1).split(":")[-1].strip().replace("'", "")
        # print(f'Threating noisy entrance on rate: {noise_entry}')
        df_restaurants['rate_num'] = df_restaurants['rate_num'].apply(lambda x: x.replace(noise_entry, str(np.nan)))

# Dropping old columns
df_restaurants.drop(['approx_cost(for two people)', 'rate'], axis=1, inplace=True)


# <H1>QUERIES</H1>

# <H2>LOCATION FILTER</H2>

# In[5]:


x_l1 = []
for index, row in df.iterrows():
  if user_location in str(row['location']):
    x_l1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])

x_l2 = pd.DataFrame(x_l1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


# <h3>ONLINE ORDER IN USER LOCATION</h3>

# In[6]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
pie, ax = plt.subplots(figsize=[8,8])
x=x_l2['online_order'].value_counts()
try: 
    x['No']
except:
    x['No'] = 0
try:
    x['Yes']
except:
    x["Yes"] = 0
if(x['No'] == 0) and (x['Yes'] == 0):
    x['Yes'] = 1
colors = ['#e6d8b5', '#998759']
labels = ["Yes","No"]
plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
ax.set_title('Restaurants\n that accept online orders in \n' + str(user_location))
plt.legend(loc="upper right")
plt.savefig("loc_oo.png", bbox_inches="tight")


# <h3>TABLE BOOKINGS IN USER LOCATION</h3>

# In[7]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
pie, ax = plt.subplots(figsize=[8,8])
x=x_l2['book_table'].value_counts()
try: 
    x['No']
except:
    x['No'] = 0
try:
    x['Yes']
except:
    x["Yes"] = 0
if(x['No'] == 0) and (x['Yes'] == 0):
    x['Yes'] = 1
colors = ['#4f7554', '#7fb085']
labels = ["No", "Yes"]
plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
ax.set_title('Restaurants\n that accept online table bookings in \n' + str(user_location))
plt.legend(loc="upper right")
plt.savefig("loc_tb.png", bbox_inches="tight")


# <H2>RESTAURANT TYPE FILTER OVER LOCATION FILTER</H2>

# In[8]:


x_l_rt1 = []
for index, row in x_l2.iterrows():
  for i in user_rest_type:
    if i in str(row['rest_type']):
      x_l_rt1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])
x_l_rt2 = pd.DataFrame(x_l_rt1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


# <H2>CUISINES FILTER OVER RESTAURANT TYPE AND LOCATION FILTERS</H2>

# In[9]:


x_l_rt_c1 = []
for index, row in x_l_rt2.iterrows():
  for i in user_cuisine:
    if i in str(row['cuisines']):
      x_l_rt_c1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])
x_l_rt_c2 = pd.DataFrame(x_l_rt_c1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


# <h3>ONLINE ORDERS BASED ON USER RESTAURANT TYPES, CUISINES AND LOCATION</h3>

# In[10]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
pie, ax = plt.subplots(figsize=[8,8])
x=x_l_rt_c2['online_order'].value_counts()
try: 
    x['No']
except:
    x['No'] = 0
try:
    x['Yes']
except:
    x["Yes"] = 0
if(x['No'] == 0) and (x['Yes'] == 0):
    x['Yes'] = 1
colors = ['#e6d8b5', '#998759']
labels = ["Yes","No"]
plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
ax.set_title(
    str(user_rest_type) 
    + ' & ' + str(user_cuisine) + '\n restuarants that accept online order in \n' 
    + str(user_location))
plt.legend(loc="upper right")
plt.savefig("loc_rt_c_oo.png", bbox_inches="tight")


# <H3>ONLINE TABLE BOOKINGS BASED ON USER RESTAURANT TYPES, CUISINES AND LOCATION</H3>

# In[11]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
pie, ax = plt.subplots(figsize=[8,8])
x=x_l_rt_c2['book_table'].value_counts()
try: 
    x['No']
except:
    x['No'] = 0
try:
    x['Yes']
except:
    x["Yes"] = 0
if(x['No'] == 0) and (x['Yes'] == 0):
    x['Yes'] = 1
colors = ['#4f7554', '#7fb085']
labels = ["No", "Yes"]
plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
ax.set_title(str(user_rest_type) 
             + ' & ' + str(user_cuisine) + ' \nrestaurants that accept table bookings in\n ' 
             + str(user_location))
plt.legend(loc="upper right")
plt.savefig("loc_rt_c_tb.png", bbox_inches="tight")


# <H3>POPULAR CUISINES IN USER LOCATION</H3>

# In[12]:


plt.figure(figsize=(7,7))
full_cuisines = x_l2['cuisines'].value_counts()
cuisines=x_l2['cuisines'].value_counts()[:10]
final_answer = {}
for i in user_cuisine: 
  final_answer[i] = 0
for i in user_cuisine:
  for j in full_cuisines.keys():
    if str(i) in str(j):
      final_answer[i] += full_cuisines[j]

sns.barplot(cuisines,cuisines.index, palette="rocket")
plt.xlabel('Count')
plt.title("Most popular cuisines in " + str(user_location))
plt.savefig("loc_cuisines.png", bbox_inches="tight", transparent=False)


# <h2>COST FILTER OVER LOCATION FILTER</h2>

# In[13]:


x_l_cost1 = []
for index, row in df.iterrows():
  if user_location in str(row['location']):
    x_l_cost1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate'], row['approx_cost(for two people)']])

x_l_cost2 = pd.DataFrame(x_l_cost1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate', 'approx_cost(for two people)'])


# In[14]:


cost_dist=x_l_cost2[['rate','approx_cost(for two people)','online_order']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
plt.figure(figsize=(6,6))
plt.title('Cost distribution of restaurants in ' + str(user_location))
sns.color_palette("flare", as_cmap=True)
sns.distplot(cost_dist['approx_cost(for two people)'], color="#998759")
plt.savefig("loc_cost.png", bbox_inches="tight")


# <h2>BANGALORE QUERIES</h2>

# <H3>ONLINE ORDERS</H3>

# In[15]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
pie, ax = plt.subplots(figsize=[8,8])
x=df['online_order'].value_counts()
colors = ['#e6d8b5', '#998759']
labels = ["Yes","No"]
plt.pie(x, labels=labels,colors=colors, startangle=90, autopct='%1.2f%%')
ax.set_title('Restaurants \nthat accept online orders in \nBengaluru')
plt.legend(loc="upper right")
plt.savefig("oo.png", bbox_inches="tight")


# <H3>ONLINE TABLE BOOKINGS</H3>

# In[16]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
pie, ax = plt.subplots(figsize=[8,8])
x=df['book_table'].value_counts()
colors = ['#4f7554', '#7fb085']
labels = ["No", "Yes"]
plt.pie(x, labels=labels, colors=colors, autopct='%1.2f%%')
ax.set_title('Restaurants \nthat accept online table bookings in \nBengaluru')
plt.legend(loc="upper right")
plt.savefig("tb.png", bbox_inches="tight")


# <H2>RESTAURANT TYPE FILTER</H2>

# In[17]:


x_rt1 = []
for index, row in df.iterrows():
  for i in user_rest_type:
    if i in str(row['rest_type']):
      x_rt1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])

x_rt2 = pd.DataFrame(x_rt1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


# <H2>CUISINES FILTER OVER RESTAURANT TYPE FILTER</H2>

# In[18]:


x_rt_c1 = []
for index, row in x_rt2.iterrows():
  for i in user_cuisine:
    if i in str(row['cuisines']):
      x_rt_c1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])

x_rt_c2 = pd.DataFrame(x_rt_c1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


# <H3>ONLINE ORDER BASED ON RESTAURANT TYPE AND CUISINES FILTERS IN BENGALURU</H3>

# In[19]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
pie, ax = plt.subplots(figsize=[8,8])
x=x_rt_c2['online_order'].value_counts()
try: 
    x['No']
except:
    x['No'] = 0
try:
    x['Yes']
except:
    x["Yes"] = 0
if(x['No'] == 0) and (x['Yes'] == 0):
    x['Yes'] = 1
colors = ['#e6d8b5', '#998759']
labels = ["Yes","No"]
plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
ax.set_title(str(user_rest_type) 
             + " & " + str(user_cuisine) + ' \nrestaurants that accept online order in \nBengaluru')
plt.legend(loc="upper right")
plt.savefig("rt_c_oo.png", bbox_inches="tight")


# <H3>TABLE BOOKING BASED ON RESTAURANT TYPE AND CUISINES FILTERS IN BENGALURU</H3>

# In[20]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
pie, ax = plt.subplots(figsize=[8,8])
x=x_rt_c2['book_table'].value_counts()
try: 
    x['No']
except:
    x['No'] = 0
try:
    x['Yes']
except:
    x["Yes"] = 0
if(x['No'] == 0) and (x['Yes'] == 0):
    x['Yes'] = 1
colors = ['#4f7554', '#7fb085']
labels = ["No", "Yes"]
plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
ax.set_title(str(user_rest_type) 
             + " & " + str(user_cuisine) + ' \nrestaurants that accept online table bookings in \nBengaluru')
plt.legend(loc="upper right")
plt.savefig("rt_c_tb.png", bbox_inches="tight")


# <h3>POPULAR CUISINES IN BENGALURU</h3>

# In[21]:


plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
plt.figure(figsize=(7,7))
cuisines=df['cuisines'].value_counts()[:10]
sns.barplot(cuisines,cuisines.index, palette="rocket")
plt.xlabel('Count')
plt.title("Most popular cuisines in Bengaluru")
plt.savefig("cuisines.png", bbox_inches="tight")


# <h3>FOODY AREAS OF BENGALURU</h3>

# In[22]:


plt.figure(figsize=(7,7))
Rest_locations=df['location'].value_counts()[:20]
sns.barplot(Rest_locations,Rest_locations.index,palette="rocket")
plt.title("Foody areas of Bengaluru")
plt.savefig("foody_areas.png", bbox_inches="tight")


# <h3>COST DISTRIBUTION</h3>

# In[23]:


cost_dist=df[['rate','approx_cost(for two people)','online_order']].dropna()
cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
plt.figure(figsize=(6,6))
plt.title("Cost distribution of restaurants in Bengaluru")
sns.distplot(cost_dist['approx_cost(for two people)'], color="#998759")
plt.savefig("cost.png", bbox_inches="tight")


# <H3>AREAS VS RATINGS VS COST</H3>

# In[24]:


plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = 'arial'
# Grouping data by city
city_group = df_restaurants.groupby(by='listed_in(city)', as_index=False).agg({'rate_num': 'mean',
                                                                               'approx_cost': 'mean'})
city_group.sort_values(by='rate_num', ascending=False, inplace=True)

# Ploting
fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(x='listed_in(city)', y='approx_cost', data=city_group, palette='flare_r', 
            order=city_group['listed_in(city)'])
ax2 = ax.twinx()
sns.lineplot(x='listed_in(city)', y='rate_num', data=city_group, color='gray', ax=ax2, sort=False)

# Labeling line chart (rate)
xs = np.arange(0, len(city_group), 1)
ys = city_group['rate_num']
for x,y in zip(xs, ys):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center', # horizontal alignment can be left, right or center
                 color='black')
    
# Labeling bar chart (cost)
for p in ax.patches:
    x = p.get_bbox().get_points()[:, 0]
    y = p.get_bbox().get_points()[1, 1]
    ax.annotate('{}'.format(int(y)), (x.mean(), 15), va='bottom', rotation='vertical', color='white', 
                fontweight='bold')

# Customizing chart
format_spines(ax)
format_spines(ax2)
ax.tick_params(axis='x', labelrotation=90)
ax.set_title('Bengaluru Cities and all its Restaurants by Approx Cost (bars) and Rate (line)')
plt.savefig("areas_rating_cost.png", bbox_inches="tight")


# <h3>RATINGS AND COST VS ONLINE ORDER</h3>

# In[25]:


plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = 'arial'
# Building a figure
fig = plt.figure(constrained_layout=True, figsize=(15, 12))

# Axis definition with GridSpec
gs = GridSpec(2, 5, figure=fig)
ax2 = fig.add_subplot(gs[0, :3])
ax3 = fig.add_subplot(gs[0, 3:])
ax4 = fig.add_subplot(gs[1, :3])
ax5 = fig.add_subplot(gs[1, 3:])

# First Line (01) - Rate
sns.kdeplot(df_restaurants.query('rate_num > 0 & online_order == "Yes"')['rate_num'], ax=ax2,
             color='darkslateblue', shade=True, label='With Online Order Service')
sns.kdeplot(df_restaurants.query('rate_num > 0 & online_order == "No"')['rate_num'], ax=ax2,
             color='lightsalmon', shade=True, label='Without Online Order Service')
ax2.set_title('Restaurants Rate Distribution by Online Order Service Offer', color='dimgrey', size=14)
sns.boxplot(x='online_order', y='rate_num', data=df_restaurants, palette=['darkslateblue', 'lightsalmon'], ax=ax3)
ax3.set_title('Box Plot for Rate and Online Order Service', color='dimgrey', size=14)

# First Line (01) - Cost
sns.kdeplot(df_restaurants.query('approx_cost > 0 & online_order == "Yes"')['approx_cost'], ax=ax4,
             color='darkslateblue', shade=True, label='With Online Order Service')
sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "No"')['approx_cost'], ax=ax4,
             color='lightsalmon', shade=True, label='Without Online Order Service')
ax4.set_title('Restaurants Approx Cost Distribution by Online Order Service Offer', color='dimgrey', size=14)
sns.boxplot(x='online_order', y='approx_cost', data=df_restaurants, palette=['darkslateblue', 'lightsalmon'], ax=ax5)
ax5.set_title('Box Plot for Cost and Online Order Service', color='dimgrey', size=14)


# Customizing plots
for ax in [ax2, ax3, ax4, ax5]:
    format_spines(ax, right_border=False)

plt.tight_layout()
plt.savefig("complex_oo.png", bbox_inches="tight")


# <h3>RATINGS AND COST VS ONLINE TABLE BOOKING</h3>

# In[26]:


plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = 'arial'
# Building a figure
fig = plt.figure(constrained_layout=True, figsize=(15, 12))

# Axis definition with GridSpec
gs = GridSpec(2, 5, figure=fig)
ax2 = fig.add_subplot(gs[0, :3])
ax3 = fig.add_subplot(gs[0, 3:])
ax4 = fig.add_subplot(gs[1, :3])
ax5 = fig.add_subplot(gs[1, 3:])

# First Line (01) - Rate
sns.kdeplot(df_restaurants.query('rate_num > 0 & book_table == "Yes"')['rate_num'], ax=ax2,
             color='mediumseagreen', shade=True, label='With Book Table Service')
sns.kdeplot(df_restaurants.query('rate_num > 0 & book_table == "No"')['rate_num'], ax=ax2,
             color='crimson', shade=True, label='Without Book Table Service')
ax2.set_title('Restaurants Rate Distribution by Book Table Service Offer', color='dimgrey', size=14)
sns.boxplot(x='book_table', y='rate_num', data=df_restaurants, palette=['mediumseagreen', 'crimson'], ax=ax3)
ax3.set_title('Box Plot for Rate and Book Table Service', color='dimgrey', size=14)

# First Line (01) - Cost
sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "Yes"')['approx_cost'], ax=ax4,
             color='mediumseagreen', shade=True, label='With Book Table Service')
sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "No"')['approx_cost'], ax=ax4,
             color='crimson', shade=True, label='Without Book Table Service')
ax4.set_title('Restaurants Approx Cost Distribution by Book Table Service Offer', color='dimgrey', size=14)
sns.boxplot(x='book_table', y='approx_cost', data=df_restaurants, palette=['mediumseagreen', 'crimson'], ax=ax5)
ax5.set_title('Box Plot for Cost and Book Table Service', color='dimgrey', size=14)


# Customizing plots
for ax in [ax2, ax3, ax4, ax5]:
    format_spines(ax, right_border=False)
    
plt.tight_layout()
plt.savefig("complex_tb.png", bbox_inches="tight")


# In[27]:


# <H1>PDF</H1>

# <h2>PAGE DIMENSIONS</h2>

# In[28]:


WIDTH = 210
HEIGHT = 297


# In[ ]:


# In[29]:


class PDF(FPDF):
	# Page header
	def header(self):
		self.image('restaura-blue.png', 10, 10, 50)
		self.set_font('helvetica', 'I', 10)
		pdf.set_text_color(68, 68, 68)
		self.cell(0, 10, 'Analytics Report', ln=1, align='C')
		self.ln(10)

	def footer(self):
		self.set_y(-15)
		self.set_font('helvetica', 'I', 10)
		pdf.set_text_color(68, 68, 68)
		self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
	def chapter_body(self, name):
		with open(name, 'rb') as fh:
			txt = fh.read().decode('latin-1')
		self.set_font('arial', '', 12)
		self.multi_cell(0, 5, txt)
		self.ln()
    
	def chapter_title(self, ch_num, ch_title, link):
		self.set_link(link)
		pdf.set_font('arial', 'B', 14)
		pdf.set_text_color(68, 68, 68)
		chapter_title = f'{ch_num}. {ch_title}'
		self.cell(0, 10, chapter_title, ln=1)
        
	def print_chapter(self, ch_num, ch_title, name, link):
		self.chapter_title(ch_num, ch_title, link)
		self.chapter_body(name)

	def add_row(self, s_no, name, pg, link_name):
		pdf.set_font('arial', '', 14)
		pdf.set_text_color(68, 68, 68)
		pdf.cell(20, 15, str(s_no), border=1, align='C', link=link_name)
		pdf.set_font('arial', '', 12)
		pdf.cell(150, 15, name, border=1, link=link_name)
		pdf.set_font('arial', '', 14)
		pdf.cell(25, 15, str(pg), border=1, ln=1, align='C', link=link_name)


# In[30]:


today = date.today()
d1 = today.strftime("%d/%m/%Y")


# In[31]:
plt.clf()
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = 'arial'
explode = [0, 0]
color_good = ["white", "#63bf32"]
color_okay = ["white", "#fa8f02"]
color_bad = ["white", "#e61919"]

if (succ > 70):
    plt.pie([100 - succ, succ], labels = ['', str(succ)+"%"], colors = color_good, startangle=90, shadow=1)
elif (succ < 33):
    plt.pie([100 - succ, succ], labels = ['', str(succ)+"%"], colors = color_bad, startangle=90, shadow=1)
else:
    plt.pie([100 - succ, succ], labels = ['', str(succ)+"%"], colors = color_okay, startangle=90, shadow=1)
circle = plt.Circle((0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.savefig('succ.png', bbox_inches="tight")


pdf = PDF()


# In[32]:


one = pdf.add_link()
two = pdf.add_link()
three = pdf.add_link()
four = pdf.add_link()
five = pdf.add_link()
six = pdf.add_link()
seven = pdf.add_link()
eight = pdf.add_link()
nine = pdf.add_link()
ten = pdf.add_link()
eleven = pdf.add_link()


# In[33]:


'''Page 1'''
'''Cover Page'''

pdf.add_page()

pdf.image('cover.png', x = -0.5, w = pdf.w + 1)

pdf.set_font('arial', 'BI', 15)
pdf.set_text_color(169, 99, 36)

pdf.cell(30)
pdf.cell(0, 10, 'Restaurant Name', ln=1)

pdf.cell(30)
pdf.cell(0, 10, 'Restaurant Type', ln=1)

pdf.cell(30)
pdf.cell(0, 10, 'Cuisines', ln=1)

pdf.cell(30)
pdf.cell(0, 10, 'Location', ln=1)

pdf.cell(30)
pdf.cell(0, 10, str(d1), ln=1)

pdf.set_text_color(0, 0, 0)


# In[34]:


'''Page 2'''
'''Content Page'''

pdf.add_page()

pdf.set_font('arial', 'B', 20)
pdf.set_text_color(68, 68, 68)

pdf.cell(0, 15, 'Content Page', ln=1, align='C')

pdf.set_font('arial', 'B', 15)
pdf.set_text_color(68, 68, 68)

pdf.cell(20, 15, 'Sl. No.', border=1, align='C')
pdf.cell(150, 15, 'Content Name', border=1, align='C')
pdf.cell(25, 15, 'Page No.', border=1, ln=1, align='C')

pdf.add_row(1, 'Online platform conformity details of restaurants in ' + str(user_location), 3, one)
pdf.add_row(2, 'Online platform conformity details of similar restaurants in ' + str(user_location), 4, two)
pdf.add_row(3, 'Most popular cuisines in ' + str(user_location), 5, three)
pdf.add_row(4, 'Cost distribution of restaurants in ' + str(user_location), 5, four)
pdf.add_row(5, 'Importance of online platform conformity', 6, five)
pdf.add_row(6, 'Online conformity details of restaurants in Bengaluru', 8, six)
pdf.add_row(7, 'Online platform conformity details of similar restaurants in Bengaluru', 9, seven)
pdf.add_row(8, 'Most popular cuisines in Bengaluru', 10, eight)
pdf.add_row(9, 'Cost distribution of restaurants in Bengaluru', 10, nine)
pdf.add_row(10, 'Analysis of foody areas of Bengaluru', 11, ten)
pdf.add_row(11, 'Success Probability', 12, eleven)


# In[35]:


'''Page 3'''
'''Online conformity details of restaurants in user location'''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '1) Online platform conformity details of restaurants in ' + str(user_location))
# pdf.chapter_body('loc_oo_tb.txt')
pdf.print_chapter(1, 
                  'Online platform conformity details of restaurants in ' + str(user_location), 
                  'loc_oo_tb.txt', one)

# pdf.set_font('Courier', 'B', 24)
pdf.cell(50)
pdf.image("loc_oo.png", h=100)
pdf.ln()
pdf.cell(50)
pdf.image("loc_tb.png", h=100)
pdf.ln(5)


# In[36]:


'''Page 4'''
'''Online conformity details of matched restaurants in user location'''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '2) Online platform conformity details of similar restaurants in ' + str(user_location))
# pdf.chapter_body("loc_rt_c_oo_tb.txt")

pdf.print_chapter(2, 
                  'Online platform conformity details of similar restaurants in ' + str(user_location),
                  'loc_rt_c_oo_tb.txt', two)

pdf.cell(50)
pdf.image("loc_rt_c_oo.png", h=100)
pdf.ln()
pdf.cell(50)
pdf.image("loc_rt_c_tb.png", h=100)
pdf.ln(5)


# In[37]:


'''Page 5'''
'''Most popular cuisines and cost distribution in user location '''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '3) Most popular cuisines in ' + str(user_location))
# pdf.chapter_body("loc_c.txt")

pdf.print_chapter(3, 
                  'Most popular cuisines in ' + str(user_location),
                  'loc_c.txt', three)

pdf.cell(5)
pdf.image("loc_cuisines.png", h=90)
pdf.ln()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '4) Cost distribution of restaurants in ' + str(user_location))
# pdf.chapter_body("loc_cost.txt")

pdf.print_chapter(4, 
                  'Cost distribution of restaurants in ' + str(user_location),
                  'loc_cost.txt', four)

pdf.cell(35)
pdf.image("loc_cost.png", h=90)


# In[38]:


'''Page 6'''
'''Importance of online platform conformity'''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '5) Importance of online platform conformity')
# pdf.chapter_body("complex1.txt")

pdf.print_chapter(5, 
                  'Importance of online platform conformity',
                  'complex1.txt', five)

pdf.ln(13)

pdf.image("complex_oo.png", h=150)


'''Page 7'''
pdf.image("complex_tb.png", h=150)
pdf.ln()
pdf.chapter_body("complex2.txt")


# In[39]:


'''Page 8'''
'''Online conformity details of restaurants in Bengaluru'''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '6) Online conformity details of restaurants in Bengaluru')
# pdf.chapter_body("oo_tb.txt")

pdf.print_chapter(6, 
                  'Online conformity details of restaurants in Bengaluru',
                  'oo_tb.txt', six)

pdf.cell(50)
pdf.image("oo.png", h=100)
pdf.ln()

pdf.cell(50)
pdf.image("tb.png", h=100)
pdf.ln()


# In[40]:


'''Page 9'''
'''Online conformity details of matched restaurants in user location'''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '7) Online platform conformity details of similar restaurants in Bengaluru')
# pdf.chapter_body("rt_c_oo_tb.txt")

pdf.print_chapter(7, 
                  'Online platform conformity details of similar restaurants in Bengaluru',
                  'rt_c_oo_tb.txt', seven)

pdf.cell(50)
pdf.image("rt_c_oo.png", h=100)
pdf.ln()
pdf.cell(50)
pdf.image("rt_c_tb.png", h=100)
pdf.ln(5)


# In[41]:


'''Page 10'''
'''Most popular cuisines and cost distribution in Bengaluru'''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '8) Most popular cuisines in Bengaluru')
# pdf.chapter_body("blore_c.txt")

pdf.print_chapter(8, 
                  'Most popular cuisines in Bengaluru',
                  'blore_c.txt', eight)

pdf.cell(25)
pdf.image("cuisines.png", h=90)
pdf.ln()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '9) Cost distribution of restaurants in Bengaluru')
# pdf.chapter_body("blore_cost.txt")

pdf.print_chapter(9, 
                  'Cost distribution of restaurants in Bengaluru',
                  'blore_cost.txt', nine)

pdf.cell(35)
pdf.image("cost.png", h=90)


# pdf.add_page()
# pdf.image("cuisines.png", 5, 50, h=100)
# pdf.image("foody_areas.png", 5, 150, h=100)


# In[42]:


'''Page 11'''
'''Analysis of foody areas of Bengaluru'''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

# pdf.multi_cell(0, 10, '10) Analysis of foody areas of Bengaluru')
# pdf.chapter_body("foody_areas.txt")

pdf.print_chapter(10, 
                  'Analysis of foody areas of Bengaluru',
                  'foody_areas.txt', ten)

pdf.cell(20)
pdf.image("foody_areas.png", h=100)
pdf.ln()
pdf.cell(20)
pdf.image("areas_rating_cost.png", h=100)
pdf.ln()


# In[43]:


'''Page 12'''
'''Success Probability'''

pdf.add_page()

pdf.set_font('arial', 'B', 14)
pdf.set_text_color(68, 68, 68)

pdf.cell(0, 10, '11) Success Probability : ' + str(succ) + "%", ln=1)

pdf.cell((WIDTH/2)-40)
pdf.image("succ.png", h=60)
pdf.ln()

if (succ > 70):
    pdf.print_chapter('\b', '', 'succ_good.txt', eleven)
elif (succ < 33):
    pdf.print_chapter('\b', '', 'succ_bad.txt', eleven)
else:
    pdf.print_chapter('\b', '', 'succ_okay.txt', eleven)


# In[44]:


pdf.output('REPORT.pdf', 'F')


# In[ ]:




