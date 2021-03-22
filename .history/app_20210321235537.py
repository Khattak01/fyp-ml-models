import pickle
import numpy as np
# import pyodbc
#import pandas as pd
#import random
#import datetime

from flask import Flask
from flask import jsonify
from keras.models import load_model
import tensorflow as tf


app = Flask(__name__)

# model = pickle.load(open('./BalancedModel.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
model = load_model('model.h5')

@app.route('/hello/', methods=['GET', 'POST'])
def welcome():
    return "Hello World"


# @app.before_request
# def before():
#     print("This is executed BEFORE each predict request.")

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    dataready = [24,1,39,0,0,0,0,1,0,1,0,1,0,1,0,22000,1,0,0,0,0,1,12,1,1,1]
    data = np.array(dataready)

    data = data.reshape(1,-1) #column to row

    df = scaler.transform(data)

    # #del input_features[0:2]
    result = model.predict_classes(df)
    #print(result[0])
    # # score = result[0][0]
    
    print ("result >>> ",result)
    # print ("data >>> ",data)
    print ("df >>> ",df)
    print ("data >>> ",data)
    return jsonify({'name':'khattak01',
                    'address':'Nowshera'})


if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = 7000     
    app.run(debug = True , use_reloader=False)
app.run(host=HOST,port=PORT)

# from flask import Flask, request, render_template, jsonify
#from tensorflow.keras.models import load_model
#import tensorflow as tf

# con = pyodbc.connect(
#     "Driver={SQL Server Native Client 11.0};"
#     "Server=HAIER-PC\SQLEXPRESS;"
#     "Database=DenguePrediction;"
#     "Trusted_Connection=yes;"
#     )

# #(UserID, Age, Gender, Temperature, Headache, Nausea, Joint, Muscle, Rash, Bleeding, Vomit, Drowsey, Cough, Abdominal, Diarrhea, Orbit, Platelets, Allergy, Hypertension, Asthma, Cancer, Diabetes, HistDengue,Result)
# app = flask.Flask(__name__)
# model = pickle.load(open('BalancedModel.pkl','rb'))
# scaler = pickle.load(open('StandardScaler.pkl','rb'))
# #model = load_model('newModel2.h5')

# def report_database(con, userid , val , model_result ):

#     cursor = con.cursor()
#     #print(val)
#     current_time = datetime.datetime.now()
#     current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
#     sql = "INSERT INTO Report VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

#     cursor.execute(sql,((current_time),(userid),(val[0]),(val[1]),(val[2]),(val[3]),(val[4]),(val[5]),(val[6]),(val[7]),(val[8]),(val[9]),(val[10]),
#                         (val[11]),(val[12]),(val[13]),(val[14]),(val[15]),(val[16]),(val[17]),(val[18]),(val[19]),
#                         (val[20]),(val[21]),(val[22]),(val[23]),(model_result)))
#     con.commit()

# @app.route('/',methods = ['POST','GET'])
# def home():
#     #final_prediction(con)
#     return render_template('index.html')

# @app.route('/predict', methods = ['POST','GET'])
# def predict():
#     #encoding = 'utf-8'
#     months = ['January','February','March','April','May','June','July','August','September','Octuber','November','December']
#     #gets value in string
#     curr_month = (flask.request.args['month'])
#     print(curr_month)
#     house_type = flask.request.args['house_type']

#     sources = ['Tap water','Borehole','Open well','Rain water','River/Pond/Canal','Other Sources']
#     #gets value in string
#     water_source = flask.request.args['water_source']
#     print(water_source)
#     area_type = flask.request.args['area_type']

#     user_id = int(flask.request.args['userID'])

#     gender = int(flask.request.args['userGender'])
#     #f1 = gender.encode(encoding)
#     drowsy = int(flask.request.args['drowsy'])
#     #f2 = drowsy.encode(encoding)
#     headache = int(flask.request.args['headache'])
#     #f3 = headache.encode(encoding)
#     nausea = int(flask.request.args['nausea'])
#     #f4 = nausea.encode(encoding)
#     joint = int(flask.request.args['joint'])
#     #f5 = joint.encode(encoding)
#     muscle = int(flask.request.args['muscle'])
#     #f6 =muscle.encode(encoding)
#     rash = int(flask.request.args['rash'])
#     #f7 = rash.encode(encoding)
#     bleed = int(flask.request.args['bleed'])
#     #f8 = bleed.encode(encoding)
#     vomit = int(flask.request.args['vomit'])
#     #f9 = vomit.encode(encoding)
#     fever = int(flask.request.args['fever'])
#     #f10 = fever.encode(encoding)
#     temp = request.args.get('temperature')
#     #f11 = temp.encode(encoding)
#     age = int(flask.request.args['userAge'])
#     #f12 = age.encode(encoding)
#     cough = int(flask.request.args['cough'])
#     #f13 = cough.encode(encoding)
#     abdominal = int(flask.request.args['abs'])
#     #f14 = abdominal.encode(encoding)
#     diarrhea =int(flask.request.args['diarrhea'])
#     #f15 = diarrhea.encode(encoding)
#     orbital = int(flask.request.args['orbital'])
#     #f16 = orbital.encode(encoding)
#     allergy = int(flask.request.args['allergy'])
#     #f17 = allergy.encode(encoding)
#     hypertension = int(flask.request.args['hyper'])
#     #f18 = hypertension.encode(encoding)
#     asthma = int(flask.request.args['asthma'])
#     #f19 = asthma.encode(encoding)
#     cancer = int(flask.request.args['cancer'])
#     #f20 = cancer.encode(encoding)
#     diabetes = int(flask.request.args['diabetes'])
#     #f21 = diabetes.encode(encoding)
#     dengue_hist = int(flask.request.args['dengue'])
#     #f22 = dengue_hist.encode(encoding)
#     plt = int(request.args.get('plt'))
#     #f23 = platelets.encode(encoding)
#     platelets = request.args.get('plt_count')

#     #if user didn't knows his/her temperature
#     if fever == 0:
#         temp = random.uniform(36.5,37.5)
#         temp = round(temp,1)
#     elif fever ==1:
#         temp = float(temp)

#     #if user didn't know his/her platelets
#     if plt == 0:
#         platelets = random.randint(150000,450000)
#     elif plt == 1:
#         platelets = int(platelets)

#     #Gets the value of month that user enter in integer actual value is in string
#     for i in range(len(months)):
#         if months[i] == curr_month:
#             month_val = i+1

#     #gets the value of water_source that user enters in integer actual value is in string
#     for i in range(len(sources)):
#         if sources[i] == water_source:
#             water_val = i+1

#     if (water_source) == 'Tube well':
#         water_val = 2

#     input_features = [age,gender,temp,headache,nausea,joint,muscle,rash,bleed,vomit,drowsy,cough,
#                       abdominal,diarrhea,orbital,platelets,allergy,hypertension,asthma,cancer,diabetes,
#                       dengue_hist,month_val,house_type,area_type,water_val]

#     #dataready = [24,1,40,1,0,1,0,1,0,1,0,1,0,1,0,14000,1,0,0,0,0,1,12,1,1,1]
#     #data = data.reshape(1,-1)
#     #encoded_val = [f12,f1,f11,f3,f4,f5,f6,f7,f8,f9,f2,f13,f14,f15,f16,f23,f17,f18,f19,f20,f21,f22]
#     #dataset = pd.read_excel('newData.xlsx')
#     #Train = dataset.iloc[:,:-1].values
#     #from sklearn.preprocessing import StandardScaler
#     #sc = StandardScaler()
#     #scaler = sc.fit(Train[:,:])

#     data = np.array(input_features)

#     data = data.reshape(1,-1) #column to row

#     df = scaler.transform(data)

#     #data = data.reshape(-1,1) row to column

#     #del input_features[0:2]

#     #feature needs to be store on database
#     input_features = [curr_month,house_type,area_type,water_source,temp,headache,nausea,joint,muscle,rash,bleed,vomit,drowsy,cough,
#                       abdominal,diarrhea,orbital,platelets,allergy,hypertension,asthma,cancer,diabetes,
#                       dengue_hist]

#     result = model.predict_classes([df])
#     #print(result[0])
#     score = result[0][0]
#     report_database(con,(user_id),input_features,int(score))

#     #return render_template('index.html',prediction_result=result)
#     return flask.redirect('http://localhost:14630/disease_report.aspx')

# if __name__ == '__main__':
#     app.run(debug = True , use_reloader=False)
#     #HOST = '127.0.0.1'
#     #PORT = 4000      #make sure this is an integer

# app.run(HOST, PORT);
