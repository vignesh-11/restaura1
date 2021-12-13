import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import json
from datetime import date

from fpdf import FPDF

# Standard libraries
import pandas as pd
import numpy as np
pd.options.display.max_colwidth = 200
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
from geopy.geocoders import Nominatim

# Utils modules
# from custom_transformers import *
from viz_utils import *
# from main import *

from main import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model1 = pickle.load(open('ml_model.pkl','rb'))
cols = ["url","address","name","online_order","book_table","rate","votes","phone","location","rest_type","dish_liked","cuisines","approx_cost(for two people)","reviews_list","menu_item","listed_in(type)","listed_in(city)"]
@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    #features = [x for x in request.form.values()]
    #features = np.array(features)
    features = json.loads(request.data)
    list_features = []
    for col in cols:
        list_features.append(features[col])
    #print(list_features)
    features1 = np.array(list_features)
    #print(features1)
    final_features = pd.DataFrame([features1],columns=cols)
    #print(final_features)
    #temp = pd.read_csv('Dataset/Test.csv')
    #print(temp)
    rd2 = pd.read_csv('Dataset/zomato.csv')
    rd3 = rd2.append(final_features, ignore_index = True)
    tr1, nr1 = common_pipeline.fit_transform(rd3)
    #print(nr1.shape)
    nr_prep = full_pipeline.fit_transform(nr1.drop('target', axis=1))
    if nr_prep.shape[1]==54:
         nr_prep = np.delete(nr_prep, 53, 1)
    #nr_prep.drop(nr_prep.columns[[54]], axis=1, inplace=True)
    print(nr_prep.shape)
    y_pd = model1.predict(nr_prep)
    y_pb = model1.predict_proba(nr_prep)
    y_sc = y_pb[:, 1]
    nr1['success_class'] = y_pd
    nr1['success_proba'] = y_sc
    nr1['success_proba'].iloc[-1]

    nrd = nr1.reset_index().merge(rd3.reset_index()[['name', 'index']], how='left', on='index')    
    val = nrd.iloc[-1]["success_proba"]
    val = 100 * val
    #print(nrd.iloc[-1])
    return jsonify({"success_prob": val})
    #return render_template('index.html', prediction_text=nrd.iloc[-1]["name"] + ' Restaurant Success Probability {}'.format(val))
    


@app.route('/report',methods=["GET", "POST"])
def download():
    features = json.loads(request.data)
    # In[3]:
    user_cuisine = features["cuisines"].split(', ') #['North Indian', 'South Indian']
    user_rest_type = features["rest_type"].split(', ') #['Casual Dining', 'Cafe']
    user_location = features["location"] #"Banashankari"

    print(user_cuisine)
    print(user_rest_type)
    print(user_location)

    name = features["name"]
    succ = round(features["success_prob"], 2)

    print(succ)


    # <H1>READING DATA</H1>

    # In[4]:


    # Reading restaurants data
    df_restaurants = pd.read_csv('Dataset/zomato.csv')
    df = pd.read_csv('Dataset/zomato.csv')
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


    # <h2>COST FILTER OVER LOCATION FILTER</h2>

    # In[6]:


    x_l_cost1 = []
    for index, row in df.iterrows():
        if user_location in str(row['location']):
            x_l_cost1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate'], row['approx_cost(for two people)']])

    x_l_cost2 = pd.DataFrame(x_l_cost1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate', 'approx_cost(for two people)'])


    # <H2>RESTAURANT TYPE FILTER OVER LOCATION FILTER</H2>

    # In[7]:


    x_l_rt1 = []
    for index, row in x_l2.iterrows():
        for i in user_rest_type:
            if i in str(row['rest_type']):
                x_l_rt1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])
    x_l_rt2 = pd.DataFrame(x_l_rt1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


    # <H2>CUISINES FILTER OVER LOCATION FILTER</H2>

    # In[8]:


    x_l_c1 = []
    for index, row in x_l2.iterrows():
        for i in user_cuisine:
            if i in str(row['cuisines']):
                x_l_c1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])

    x_l_c2 = pd.DataFrame(x_l_c1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


    # <H2>CUISINES FILTER OVER RESTAURANT TYPE AND LOCATION FILTERS</H2>

    # In[9]:


    x_l_rt_c1 = []
    for index, row in x_l_rt2.iterrows():
        for i in user_cuisine:
            if i in str(row['cuisines']):
                x_l_rt_c1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])
    x_l_rt_c2 = pd.DataFrame(x_l_rt_c1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


    # <H2>RESTAURANT TYPE FILTER</H2>

    # In[10]:


    x_rt1 = []
    for index, row in df.iterrows():
        for i in user_rest_type:
            if i in str(row['rest_type']):
                x_rt1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])

    x_rt2 = pd.DataFrame(x_rt1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


    # <H2>CUISINES FILTER OVER RESTAURANT TYPE FILTER</H2>

    # In[11]:


    x_rt_c1 = []
    for index, row in x_rt2.iterrows():
        for i in user_cuisine:
            if i in str(row['cuisines']):
                x_rt_c1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])

    x_rt_c2 = pd.DataFrame(x_rt_c1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


    # <H2>CUISINES FILTER</H2>

    # In[12]:


    x_c1 = []
    for index, row in df.iterrows():
        for i in user_cuisine:
            if i in str(row['cuisines']):
                x_c1.append([row['rest_type'], row['location'], row['cuisines'], row['online_order'], row['book_table'], row['rate']])

    x_c2 = pd.DataFrame(x_c1, columns = ['rest_type', 'location', 'cuisines', 'online_order', 'book_table', 'rate'])


    # <h3>ONLINE ORDER IN USER LOCATION</h3>

    # In[13]:


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
    colors = ['#998759', '#e6d8b5']
    labels = ["No", "Yes"]
    plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
    ax.set_title('Restaurants\n that accept online orders in \n' + str(user_location))
    plt.legend(loc="upper right")
    plt.savefig("img/loc_oo.png", bbox_inches="tight")


    # <h3>TABLE BOOKINGS IN USER LOCATION</h3>

    # In[14]:


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
    plt.savefig("img/loc_tb.png", bbox_inches="tight")


    # <H3>ONLINE ORDERS BASED ON USER RESTAURANT TYPES AND LOCATION</H3>

    # In[15]:


    plt.rcParams["font.size"] = 16
    pie, ax = plt.subplots(figsize=[8,8])
    x=x_l_rt2['online_order'].value_counts()
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
    colors = ['#998759', '#e6d8b5']
    labels = ["No", "Yes"]
    plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
    ax.set_title(str(user_rest_type) + ' \nrestaurants that accept online orders in \n' + str(user_location))
    plt.legend(loc="upper right")
    plt.savefig("img/loc_rt_oo.png", bbox_inches="tight")


    # <H3>ONLINE TABLE BOOKINGS BASED ON USER RESTAURANT TYPES AND LOCATION</H3>

    # In[16]:


    plt.rcParams["font.size"] = 16
    pie, ax = plt.subplots(figsize=[8,8])
    x=x_l_rt2['book_table'].value_counts()
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
    ax.set_title(str(user_rest_type) + ' \nrestaurants that accept online table booking in \n' + str(user_location))
    plt.legend(loc="upper right")
    plt.savefig("img/loc_rt_tb.png", bbox_inches="tight")


    # <H3>ONLINE ORDERS BASED ON USER CUISINES AND LOCATION</H3>

    # In[17]:


    plt.rcParams["font.size"] = 16
    pie, ax = plt.subplots(figsize=[8,8])
    x=x_l_c2['online_order'].value_counts()
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
    colors = ['#998759', '#e6d8b5']
    labels = ["No", "Yes"]
    plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
    ax.set_title(str(user_cuisine) + ' \nrestaurants that accept online orders in \n' + str(user_location))
    plt.legend(loc="upper right")
    plt.savefig("img/loc_c_oo.png", bbox_inches="tight")


    # <H3>ONLINE TABLE BOOKINGS BASED ON USER CUISINES AND LOCATION</H3>

    # In[18]:


    plt.rcParams["font.size"] = 16
    pie, ax = plt.subplots(figsize=[8,8])
    x=x_l_c2['book_table'].value_counts()
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
    ax.set_title(str(user_cuisine) + ' \nrestaurants that accept online table booking in \n' + str(user_location))
    plt.legend(loc="upper right")
    plt.savefig("img/loc_c_tb.png", bbox_inches="tight")


    # <h3>ONLINE ORDERS BASED ON USER RESTAURANT TYPES, CUISINES AND LOCATION</h3>

    # In[19]:


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
    colors = ['#998759', '#e6d8b5']
    labels = ["No", "Yes"]
    plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
    ax.set_title(
        str(user_rest_type) 
        + ' & ' + str(user_cuisine) + '\n restuarants that accept online order in \n' 
        + str(user_location))
    plt.legend(loc="upper right")
    plt.savefig("img/loc_rt_c_oo.png", bbox_inches="tight")


    # <H3>ONLINE TABLE BOOKINGS BASED ON USER RESTAURANT TYPES, CUISINES AND LOCATION</H3>

    # In[20]:


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
    plt.savefig("img/loc_rt_c_tb.png", bbox_inches="tight")


    # <H3>POPULAR CUISINES IN USER LOCATION</H3>

    # In[21]:


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
    plt.savefig("img/loc_cuisines.png", bbox_inches="tight", transparent=False)


    # In[22]:


    cost_dist=x_l_cost2[['rate','approx_cost(for two people)','online_order']].dropna()
    cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
    cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
    plt.figure(figsize=(6,6))
    plt.title('Cost distribution of restaurants in ' + str(user_location))
    sns.color_palette("flare", as_cmap=True)
    sns.distplot(cost_dist['approx_cost(for two people)'], color="#998759")
    plt.savefig("img/loc_cost.png", bbox_inches="tight")


    # <h2>BANGALORE QUERIES</h2>

    # <H3>ONLINE ORDERS</H3>

    # In[23]:


    # plt.rcParams["font.size"] = 16
    # plt.rcParams["font.family"] = 'arial'
    # pie, ax = plt.subplots(figsize=[8,8])
    # x=df['online_order'].value_counts()
    # colors = ['#998759', '#e6d8b5']
    # labels = ["No", "Yes"]
    # plt.pie(x, labels=labels,colors=colors, startangle=90, autopct='%1.2f%%')
    # ax.set_title('Restaurants \nthat accept online orders in \nBengaluru')
    # plt.legend(loc="upper right")
    # plt.savefig("img/oo.png", bbox_inches="tight")


    # <H3>ONLINE TABLE BOOKINGS</H3>

    # In[24]:


    # plt.rcParams["font.size"] = 16
    # plt.rcParams["font.family"] = 'arial'
    # pie, ax = plt.subplots(figsize=[8,8])
    # x=df['book_table'].value_counts()
    # colors = ['#4f7554', '#7fb085']
    # labels = ["No", "Yes"]
    # plt.pie(x, labels=labels, colors=colors, autopct='%1.2f%%')
    # ax.set_title('Restaurants \nthat accept online table bookings in \nBengaluru')
    # plt.legend(loc="upper right")
    # plt.savefig("img/tb.png", bbox_inches="tight")


    # <H3>ONLINE ORDER BASED ON RESTAURANT TYPE IN BENGALURU</H3>

    # In[25]:


    plt.rcParams["font.size"] = 16
    pie, ax = plt.subplots(figsize=[8,8])
    x=x_rt2['online_order'].value_counts()
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
    colors = ['#998759', '#e6d8b5']
    labels = ["No", "Yes"]
    plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
    ax.set_title(str(user_rest_type) + ' \nrestaurants that accept online orders in \nBangalore')
    plt.legend(loc="upper right")
    plt.savefig("img/rt_oo.png", bbox_inches="tight")


    # <H3>TABLE BOOKING BASED ON RESTAURANT TYPE IN BENGALURU</H3>

    # In[26]:


    plt.rcParams["font.size"] = 16
    pie, ax = plt.subplots(figsize=[8,8])
    x=x_rt2['book_table'].value_counts()
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
    ax.set_title(str(user_rest_type) + ' \nrestaurants that accept online table bookings in \nBangalore')
    plt.legend(loc="upper right")
    plt.savefig("img/rt_tb.png", bbox_inches="tight")


    # <H3>ONLINE ORDER BASED ON CUISINES IN BENGALURU</H3>

    # In[27]:


    plt.rcParams["font.size"] = 16
    pie, ax = plt.subplots(figsize=[8,8])
    x=x_c2['online_order'].value_counts()
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
    colors = ['#998759', '#e6d8b5']
    labels = ["No", "Yes"]
    plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
    ax.set_title(str(user_cuisine) + ' \nrestaurants that accept online orders in \nBangalore')
    plt.legend(loc="upper right")
    plt.savefig("img/c_oo.png", bbox_inches="tight")


    # <H3>ONLINE ORDER BASED ON CUISINES IN BENGALURU</H3>

    # In[28]:


    plt.rcParams["font.size"] = 16
    pie, ax = plt.subplots(figsize=[8,8])
    x=x_c2['book_table'].value_counts()
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
    ax.set_title(str(user_cuisine) + ' \nrestaurants that accept online table bookings in \nBangalore')
    plt.legend(loc="upper right")
    plt.savefig("img/c_tb.png", bbox_inches="tight")


    # <H3>ONLINE ORDER BASED ON RESTAURANT TYPE AND CUISINES FILTERS IN BENGALURU</H3>

    # In[29]:


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
    colors = ['#998759', '#e6d8b5']
    labels = ["No", "Yes"]
    plt.pie(x, labels=labels, colors=colors, startangle=90, autopct='%1.2f%%')
    ax.set_title(str(user_rest_type) 
                + " & " + str(user_cuisine) + ' \nrestaurants that accept online order in \nBengaluru')
    plt.legend(loc="upper right")
    plt.savefig("img/rt_c_oo.png", bbox_inches="tight")


    # <H3>TABLE BOOKING BASED ON RESTAURANT TYPE AND CUISINES FILTERS IN BENGALURU</H3>

    # In[30]:


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
    plt.savefig("img/rt_c_tb.png", bbox_inches="tight")


    # <h3>POPULAR CUISINES IN BENGALURU</h3>

    # In[31]:


    # plt.rcParams["font.size"] = 16
    # plt.rcParams["font.family"] = 'arial'
    # plt.figure(figsize=(7,7))
    # cuisines=df['cuisines'].value_counts()[:10]
    # sns.barplot(cuisines,cuisines.index, palette="rocket")
    # plt.xlabel('Count')
    # plt.title("Most popular cuisines in Bengaluru")
    # plt.savefig("img/cuisines.png", bbox_inches="tight")


    # <h3>FOODY AREAS OF BENGALURU</h3>

    # In[32]:


    # plt.figure(figsize=(7,7))
    # Rest_locations=df['location'].value_counts()[:20]
    # sns.barplot(Rest_locations,Rest_locations.index,palette="rocket")
    # plt.title("Foody areas of Bengaluru")
    # plt.savefig("img/foody_areas.png", bbox_inches="tight")


    # <h3>COST DISTRIBUTION</h3>

    # In[33]:


    # cost_dist=df[['rate','approx_cost(for two people)','online_order']].dropna()
    # cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)
    # cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
    # plt.figure(figsize=(6,6))
    # plt.title("Cost distribution of restaurants in Bengaluru")
    # sns.distplot(cost_dist['approx_cost(for two people)'], color="#998759")
    # plt.savefig("img/cost.png", bbox_inches="tight")


    # <H3>AREAS VS RATINGS VS COST</H3>

    # In[34]:


    # plt.rcParams["font.size"] = 12
    # plt.rcParams["font.family"] = 'arial'
    # # Grouping data by city
    # city_group = df_restaurants.groupby(by='listed_in(city)', as_index=False).agg({'rate_num': 'mean',
    #                                                                                'approx_cost': 'mean'})
    # city_group.sort_values(by='rate_num', ascending=False, inplace=True)

    # # Ploting
    # fig, ax = plt.subplots(figsize=(15, 8))
    # sns.barplot(x='listed_in(city)', y='approx_cost', data=city_group, palette='flare_r', 
    #             order=city_group['listed_in(city)'])
    # ax2 = ax.twinx()
    # sns.lineplot(x='listed_in(city)', y='rate_num', data=city_group, color='gray', ax=ax2, sort=False)

    # # Labeling line chart (rate)
    # xs = np.arange(0, len(city_group), 1)
    # ys = city_group['rate_num']
    # for x,y in zip(xs, ys):
    #     label = "{:.2f}".format(y)
    #     plt.annotate(label, # this is the text
    #                  (x,y), # this is the point to label
    #                  textcoords="offset points", # how to position the text
    #                  xytext=(0,10), # distance from text to points (x,y)
    #                  ha='center', # horizontal alignment can be left, right or center
    #                  color='black')
        
    # # Labeling bar chart (cost)
    # for p in ax.patches:
    #     x = p.get_bbox().get_points()[:, 0]
    #     y = p.get_bbox().get_points()[1, 1]
    #     ax.annotate('{}'.format(int(y)), (x.mean(), 15), va='bottom', rotation='vertical', color='white', 
    #                 fontweight='bold')

    # # Customizing chart
    # format_spines(ax)
    # format_spines(ax2)
    # ax.tick_params(axis='x', labelrotation=90)
    # ax.set_title('Bengaluru Cities and all its Restaurants by Approx Cost (bars) and Rate (line)')
    # plt.savefig("img/areas_rating_cost.png", bbox_inches="tight")


    # <h3>RATINGS AND COST VS ONLINE ORDER</h3>

    # In[35]:


    # plt.rcParams["font.size"] = 12
    # plt.rcParams["font.family"] = 'arial'
    # # Building a figure
    # fig = plt.figure(constrained_layout=True, figsize=(15, 12))

    # # Axis definition with GridSpec
    # gs = GridSpec(2, 5, figure=fig)
    # ax2 = fig.add_subplot(gs[0, :3])
    # ax3 = fig.add_subplot(gs[0, 3:])
    # ax4 = fig.add_subplot(gs[1, :3])
    # ax5 = fig.add_subplot(gs[1, 3:])

    # # First Line (01) - Rate
    # sns.kdeplot(df_restaurants.query('rate_num > 0 & online_order == "Yes"')['rate_num'], ax=ax2,
    #              color='darkslateblue', shade=True, label='With Online Order Service')
    # sns.kdeplot(df_restaurants.query('rate_num > 0 & online_order == "No"')['rate_num'], ax=ax2,
    #              color='lightsalmon', shade=True, label='Without Online Order Service')
    # ax2.set_title('Restaurants Rate Distribution by Online Order Service Offer', color='dimgrey', size=14)
    # sns.boxplot(x='online_order', y='rate_num', data=df_restaurants, palette=['darkslateblue', 'lightsalmon'], ax=ax3)
    # ax3.set_title('Box Plot for Rate and Online Order Service', color='dimgrey', size=14)

    # # First Line (01) - Cost
    # sns.kdeplot(df_restaurants.query('approx_cost > 0 & online_order == "Yes"')['approx_cost'], ax=ax4,
    #              color='darkslateblue', shade=True, label='With Online Order Service')
    # sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "No"')['approx_cost'], ax=ax4,
    #              color='lightsalmon', shade=True, label='Without Online Order Service')
    # ax4.set_title('Restaurants Approx Cost Distribution by Online Order Service Offer', color='dimgrey', size=14)
    # sns.boxplot(x='online_order', y='approx_cost', data=df_restaurants, palette=['darkslateblue', 'lightsalmon'], ax=ax5)
    # ax5.set_title('Box Plot for Cost and Online Order Service', color='dimgrey', size=14)


    # # Customizing plots
    # for ax in [ax2, ax3, ax4, ax5]:
    #     format_spines(ax, right_border=False)

    # plt.tight_layout()
    # plt.savefig("img/complex_oo.png", bbox_inches="tight")


    # <h3>RATINGS AND COST VS ONLINE TABLE BOOKING</h3>

    # In[36]:


    # plt.rcParams["font.size"] = 12
    # plt.rcParams["font.family"] = 'arial'
    # # Building a figure
    # fig = plt.figure(constrained_layout=True, figsize=(15, 12))

    # # Axis definition with GridSpec
    # gs = GridSpec(2, 5, figure=fig)
    # ax2 = fig.add_subplot(gs[0, :3])
    # ax3 = fig.add_subplot(gs[0, 3:])
    # ax4 = fig.add_subplot(gs[1, :3])
    # ax5 = fig.add_subplot(gs[1, 3:])

    # # First Line (01) - Rate
    # sns.kdeplot(df_restaurants.query('rate_num > 0 & book_table == "Yes"')['rate_num'], ax=ax2,
    #              color='mediumseagreen', shade=True, label='With Book Table Service')
    # sns.kdeplot(df_restaurants.query('rate_num > 0 & book_table == "No"')['rate_num'], ax=ax2,
    #              color='crimson', shade=True, label='Without Book Table Service')
    # ax2.set_title('Restaurants Rate Distribution by Book Table Service Offer', color='dimgrey', size=14)
    # sns.boxplot(x='book_table', y='rate_num', data=df_restaurants, palette=['mediumseagreen', 'crimson'], ax=ax3)
    # ax3.set_title('Box Plot for Rate and Book Table Service', color='dimgrey', size=14)

    # # First Line (01) - Cost
    # sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "Yes"')['approx_cost'], ax=ax4,
    #              color='mediumseagreen', shade=True, label='With Book Table Service')
    # sns.kdeplot(df_restaurants.query('approx_cost > 0 & book_table == "No"')['approx_cost'], ax=ax4,
    #              color='crimson', shade=True, label='Without Book Table Service')
    # ax4.set_title('Restaurants Approx Cost Distribution by Book Table Service Offer', color='dimgrey', size=14)
    # sns.boxplot(x='book_table', y='approx_cost', data=df_restaurants, palette=['mediumseagreen', 'crimson'], ax=ax5)
    # ax5.set_title('Box Plot for Cost and Book Table Service', color='dimgrey', size=14)


    # # Customizing plots
    # for ax in [ax2, ax3, ax4, ax5]:
    #     format_spines(ax, right_border=False)
        
    # plt.tight_layout()
    # plt.savefig("img/complex_tb.png", bbox_inches="tight")


    # In[37]:

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


    # <H1>PDF</H1>

    # <h2>PAGE DIMENSIONS</h2>

    # In[38]:


    WIDTH = 210
    HEIGHT = 297


    # In[39]:


    class PDF(FPDF):
        # Page header
        def header(self):
            self.image('img/restaura-blue.png', 10, 10, 50)
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
            pdf.cell(20, 10, str(s_no), border=1, align='C', link=link_name)
            pdf.set_font('arial', '', 12)
            pdf.cell(150, 10, name, border=1, link=link_name)
            pdf.set_font('arial', '', 14)
            pdf.cell(25, 10, str(pg), border=1, ln=1, align='C', link=link_name)


    # In[40]:


    today = date.today()
    d1 = today.strftime("%d/%m/%Y")


    # In[41]:


    pdf = PDF()


    # In[42]:


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
    twelve = pdf.add_link()
    thirteen = pdf.add_link()
    fourteen = pdf.add_link()
    fifteen = pdf.add_link()
    sixteen = pdf.add_link()


    # In[43]:


    '''Page 1'''
    '''Cover Page'''

    pdf.add_page()

    pdf.image('img/cover.png', x = -0.5, w = pdf.w + 1)

    pdf.set_font('arial', 'BI', 15)
    pdf.set_text_color(169, 99, 36)

    pdf.cell(30)
    pdf.cell(0, 10, 'Restaurant Name: ' + str(name), ln=1)

    pdf.cell(30)
    pdf.cell(0, 10, 'Restaurant Type: ' + str(user_rest_type), ln=1)

    pdf.cell(30)
    pdf.cell(0, 10, 'Cuisines: ' + str(user_cuisine), ln=1)

    pdf.cell(30)
    pdf.cell(0, 10, 'Location: ' + str(user_location), ln=1)

    pdf.cell(30)
    pdf.cell(0, 10, str(d1), ln=1)

    pdf.set_text_color(0, 0, 0)


    # In[44]:


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

    pdf.add_row(1, 'Online conformity details of restaurants in ' + str(user_location), 3, one)
    pdf.add_row(2, 'Online conformity details of similar restaurant types in ' + str(user_location), 4, two)
    pdf.add_row(3, 'Online conformity details of similar cuisine restaurants in ' + str(user_location), 5, three)
    pdf.add_row(4, 'Online conformity details of similar restaurants in ' + str(user_location), 6, four)
    pdf.add_row(5, 'Most popular cuisines in ' + str(user_location), 7, five)
    pdf.add_row(6, 'Cost distribution of restaurants in ' + str(user_location), 7, six)
    pdf.add_row(7, 'Importance of online platform conformity', 8, seven)
    pdf.add_row(8, 'Online conformity details of restaurants in Bengaluru', 10, eight)
    pdf.add_row(9, 'Online conformity details of similar restaurant types in Bengaluru', 11, nine)
    pdf.add_row(10, 'Online conformity details of similar cuisine restaurants in Bengaluru', 12, ten)
    pdf.add_row(11, 'Online conformity details of similar restaurants in Bengaluru', 13, eleven)
    pdf.add_row(12, 'Most popular cuisines in Bengaluru', 14, twelve)
    pdf.add_row(13, 'Cost distribution of restaurants in Bengaluru', 14, thirteen)
    pdf.add_row(14, 'Analysis of foody areas of Bengaluru', 15, fourteen)
    pdf.add_row(15, 'Success Probability', 16, fifteen)
    pdf.add_row(16, 'Market Competition', 17, sixteen)

    pdf.set_font('arial', 'I', 10)
    pdf.set_text_color(68, 68, 68)
    pdf.ln(40)
    pdf.cell(0, 10, 
            'Disclaimer: All the graphs and analyses present in this report have been made based on the data available to us.')


    # In[45]:


    '''Page 3'''
    '''Online conformity details of restaurants in user location'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(1, 
                    'Online conformity details of restaurants in ' + str(user_location), 
                    'loc_oo_tb.txt', one)

    pdf.cell(50)
    pdf.image("img/loc_oo.png", h=100)
    pdf.ln()
    pdf.cell(50)
    pdf.image("img/loc_tb.png", h=100)
    pdf.ln(5)


    # In[46]:


    '''Page 4'''
    '''Online conformity details of similar restaurant types in user location'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(2, 
                    'Online conformity details of similar restaurant types in ' + str(user_location), 
                    'loc_rt_oo_tb.txt', two)

    pdf.cell(50)
    pdf.image("img/loc_rt_oo.png", h=100)
    pdf.ln()
    pdf.cell(50)
    pdf.image("img/loc_rt_tb.png", h=100)
    pdf.ln(5)


    # In[47]:


    '''Page 5'''
    '''Online conformity details of similar cuisine restaurants in user location'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(3, 
                    'Online conformity details of similar cuisine restaurants in ' + str(user_location), 
                    'loc_c_oo_tb.txt', three)

    pdf.cell(50)
    pdf.image("img/loc_c_oo.png", h=100)
    pdf.ln()
    pdf.cell(50)
    pdf.image("img/loc_c_tb.png", h=100)
    pdf.ln(5)


    # In[48]:


    '''Page 6'''
    '''Online conformity details of matched restaurants in user location'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(4, 
                    'Online platform conformity details of similar restaurants in ' + str(user_location),
                    'loc_rt_c_oo_tb.txt', four)

    pdf.cell(50)
    pdf.image("img/loc_rt_c_oo.png", h=100)
    pdf.ln()
    pdf.cell(50)
    pdf.image("img/loc_rt_c_tb.png", h=100)
    pdf.ln(5)


    # In[49]:


    '''Page 7'''
    '''Most popular cuisines and cost distribution in user location '''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(5, 
                    'Most popular cuisines in ' + str(user_location),
                    'loc_c.txt', five)

    pdf.cell(5)
    pdf.image("img/loc_cuisines.png", h=90)
    pdf.ln()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(6, 
                    'Cost distribution of restaurants in ' + str(user_location),
                    'loc_cost.txt', six)

    pdf.cell(35)
    pdf.image("img/loc_cost.png", h=90)


    # In[50]:


    '''Page 8'''
    '''Importance of online platform conformity'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(7, 
                    'Importance of online platform conformity',
                    'complex1.txt', seven)

    pdf.ln(13)

    pdf.image("img/complex_oo.png", h=150)


    '''Page 9'''
    pdf.image("img/complex_tb.png", h=150)
    pdf.ln()
    pdf.chapter_body("complex2.txt")
    pdf.ln()
    pdf.ln()

    pdf.set_font('arial', 'B', 12)
    pdf.set_text_color(68, 68, 68)
    pdf.multi_cell(0, 5, txt="In the following sections, we'll look at statistics similar to the ones we saw above, but with respect to Bengaluru. ")


    # In[51]:


    '''Page 10'''
    '''Online conformity details of restaurants in Bengaluru'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(8, 
                    'Online conformity details of restaurants in Bengaluru',
                    'oo_tb.txt', eight)

    pdf.cell(50)
    pdf.image("img/oo.png", h=100)
    pdf.ln()
    pdf.cell(50)
    pdf.image("img/tb.png", h=100)
    pdf.ln(5)


    # In[52]:


    '''Page 11'''
    '''Online conformity details of similar restaurant types in Bengaluru'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(9, 
                    'Online conformity details of similar restaurant types in Bengaluru',
                    'rt_oo_tb.txt', nine)

    pdf.cell(50)
    pdf.image("img/rt_oo.png", h=100)
    pdf.ln()
    pdf.cell(50)
    pdf.image("img/rt_tb.png", h=100)
    pdf.ln(5)


    # In[53]:


    '''Page 12'''
    '''Online conformity details of similar cuisine restaurants in user location'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(10, 
                    'Online platform conformity details of similar cuisine restaurants in Bengaluru',
                    'c_oo_tb.txt', ten)

    pdf.cell(50)
    pdf.image("img/c_oo.png", h=100)
    pdf.ln()
    pdf.cell(50)
    pdf.image("img/c_tb.png", h=100)
    pdf.ln(5)


    # In[54]:


    '''Page 13'''
    '''Online conformity details of matched restaurants in user location'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)


    pdf.print_chapter(11, 
                    'Online platform conformity details of similar restaurants in Bengaluru',
                    'rt_c_oo_tb.txt', eleven)

    pdf.cell(50)
    pdf.image("img/rt_c_oo.png", h=100)
    pdf.ln()
    pdf.cell(50)
    pdf.image("img/rt_c_tb.png", h=100)
    pdf.ln(5)


    # In[55]:


    '''Page 14'''
    '''Most popular cuisines and cost distribution in Bengaluru'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(12, 
                    'Most popular cuisines in Bengaluru',
                    'blore_c.txt', twelve)

    pdf.cell(25)
    pdf.image("img/cuisines.png", h=90)
    pdf.ln()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(13, 
                    'Cost distribution of restaurants in Bengaluru',
                    'blore_cost.txt', thirteen)

    pdf.cell(35)
    pdf.image("img/cost.png", h=90)


    # In[56]:


    '''Page 15'''
    '''Analysis of foody areas of Bengaluru'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.print_chapter(14, 
                    'Analysis of foody areas of Bengaluru',
                    'foody_areas.txt', fourteen)

    pdf.cell(20)
    pdf.image("img/foody_areas.png", h=100)
    pdf.ln()
    pdf.cell(20)
    pdf.image("img/areas_rating_cost.png", h=100)
    pdf.ln()


    # In[57]:


    '''Page 16'''
    '''Success Probability'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68, 68, 68)

    pdf.cell(0, 10, '15. Success Percentage : ' + str(succ) + "%", ln=1)

    pdf.cell((WIDTH/2)-40)
    pdf.image("img/succ.png", h=60)
    pdf.ln()

    if (succ > 70):
        pdf.print_chapter('\b', '', 'succ_good.txt', fifteen)
    elif (succ < 33):
        pdf.print_chapter('\b', '', 'succ_bad.txt', fifteen)
    else:
        pdf.print_chapter('\b', '', 'succ_okay.txt', fifteen)


    # In[58]:


    '''Page 13'''
    '''Competition'''

    pdf.add_page()

    pdf.set_font('arial', 'B', 14)
    pdf.set_text_color(68,68,68)

    pdf.print_chapter(16, 
                    'Market Competition',
                    'comp.txt', sixteen)


    # In[59]:


    C = round(df_restaurants['rate_num'].mean(), 2)
    m = 1000
    v = df_restaurants['votes']
    R = df_restaurants['rate_num']
    df_restaurants['imdb'] = (v / (v+m)) * R + (m / (v+m)) * C

    unique_rest_type = df_restaurants['rest_type'].unique()
    final_unique_rest_type = []
    for i in unique_rest_type:
        j = str(i).split(",")
        for k in j:
            if k.strip() not in final_unique_rest_type:
                final_unique_rest_type.append(k.strip())

    cleaned_rest_type = [x for x in final_unique_rest_type if str(x) != 'nan']
    empty_rest_type = []
    for i in cleaned_rest_type:
        if i != "None":
            empty_rest_type.append(str(i))
    final_rest_type_list = sorted(empty_rest_type)

    comp_rt = {}
    i = 0
    for j in final_rest_type_list:
        if j not in comp_rt:
            comp_rt[j] = i
            i = i + 1

    unique_cuisines1 = df_restaurants['cuisines'].unique()
    unique_cuisines2 = []
    for i in unique_cuisines1:
        split_cuisines = str(i).split(",")
        for j in split_cuisines:
            if j.strip() not in unique_cuisines2:
                unique_cuisines2.append(j.strip())

    cleaned_cuisines = [x for x in unique_cuisines2 if str(x) != 'nan']
    empty_cuisines = []
    for i in cleaned_cuisines:
        if i != "None":
            empty_cuisines.append(str(i))
    final_cuisines_list = sorted(empty_cuisines)

    comp_c = {}
    i = 0
    for j in final_cuisines_list:
        if j not in comp_c:
            comp_c[j] = i
            i = i + 1

    new_df = df_restaurants.sort_values('imdb', ascending=False)
    new_df.drop_duplicates(subset =["address", "rest_type", "cuisines"],
                        keep = 'first', inplace = True)

    index_of_true_rt = []
    for i in final_rest_type_list:
        lsls = new_df['rest_type'].str.contains(i, regex=False)
        ls_keys = list(lsls[lsls == True].keys())[0:5]
        index_of_true_rt.append(ls_keys)

    max_entries_rt = {}
    i = 0
    for j in comp_rt:
        if j not in max_entries_rt:
            max_entries_rt[j] = len(index_of_true_rt[i])
            i = i + 1

    index_of_true_c = []
    for i in final_cuisines_list:
        lsls = new_df['cuisines'].str.contains(i, regex=False)
        ls_keys = list(lsls[lsls == True].keys())[0:5]
        index_of_true_c.append(ls_keys)

    max_entries_c = {}
    i = 0
    for j in comp_c:
        if j not in max_entries_c:
            max_entries_c[j] = len(index_of_true_c[i])
            i = i + 1
        
    cm_rt = {}
    cm = 0
    for i in max_entries_rt:
        cm_rt[i] = cm
        cm += max_entries_rt[i]

    cm_c = {}
    cm = 0
    for i in max_entries_c:
        cm_c[i] = cm
        cm += max_entries_c[i]


    # In[60]:


    file1 = open("RT.txt","w")
    file2 = open("C.txt", "w")

    rt = pd.read_csv('comp_rt.csv')
    c = pd.read_csv('comp_c.csv')

    for i in user_rest_type:
        entries = max_entries_rt[i]
        start = cm_rt[i]
        for j in range(entries):
            row_rt = rt[start+j: start+j+1]
            file1.write("\n\nRestaurant name : " + row_rt['name'].to_string(index=False))
            file1.write("\nAddress : " + row_rt['address'].to_string(index=False))
            file1.write("\nRating : " + row_rt['rate_num'].to_string(index=False))
            file1.write("\nNumber of votes : " + row_rt['votes'].to_string(index=False))
            file1.write("\nRestaurant type : " + row_rt['rest_type'].to_string(index=False))
            file1.write("\nCuisines : " + row_rt['cuisines'].to_string(index=False))
            file1.write("\nDishes liked : " + row_rt['dish_liked'].to_string(index=False))
            file1.write("\nApproximate cost for two people : " + row_rt['approx_cost'].to_string(index=False) + '\n')
    file1.close()

    for i in user_cuisine:
        entries = max_entries_c[i]
        start = cm_c[i]
        for j in range(entries):
            row_c = c[start+j: start+j+1]
            file2.write("\n\nRestaurant name : " + row_c['name'].to_string(index=False))
            file2.write("\nAddress : " + row_c['address'].to_string(index=False))
            file2.write("\nRating : " + row_c['rate_num'].to_string(index=False))
            file2.write("\nNumber of votes : " + row_c['votes'].to_string(index=False))
            file2.write("\nRestaurant type : " + row_c['rest_type'].to_string(index=False))
            file2.write("\nCuisines : " + row_c['cuisines'].to_string(index=False))
            file2.write("\nDishes liked : " + row_c['dish_liked'].to_string(index=False))
            file2.write("\nApproximate cost for two people : " + row_c['approx_cost'].to_string(index=False) + '\n')
    file2.close()


    # In[61]:


    '''Restaurant Type based Competition'''
    pdf.set_font('arial', 'B', 12)
    pdf.set_text_color(68, 68, 68)
    pdf.cell(40, 10, '16.1. Restaurant Type based Competition')
    pdf.ln()

    with open('RT.txt', 'rb') as fh:
        txt = fh.read().decode('latin-1')
        pdf.set_font('arial', '', 12)
        pdf.multi_cell(0, 5, txt)
        pdf.ln()


    # In[62]:


    '''Cuisine based Competition'''
    pdf.set_font('arial', 'B', 12)
    pdf.set_text_color(68, 68, 68)
    pdf.cell(40, 10, '16.2. Cuisine based Competition')
    pdf.ln()

    with open('C.txt', 'rb') as fh:
        txt = fh.read().decode('latin-1')
        pdf.set_font('arial', '', 12)
        pdf.multi_cell(0, 5, txt)
        pdf.ln()


    # In[63]:


    pdf.output('REPORT.pdf', 'F')

    return send_file('REPORT.pdf', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)



