from flask import Flask, jsonify, request, render_template
from werkzeug import secure_filename

import numpy as np
import pandas as pd
import math
import os
from keras.models import load_model
from keras.preprocessing import image
import wget

result = []

#Load Convolutions
convolutions = load_model('/Users/justinchien/ds/metis/Project_Kojak/app/models/convolutions.h5')
print('loaded convolutions')

#Load front or angled
front_or_angle_top = load_model('/Users/justinchien/ds/metis/Project_Kojak/app/models/front_or_angle_top.h5')
print('loaded front or angle model')

#Load fronts model
fronts = load_model('/Users/justinchien/ds/metis/Project_Kojak/app/models/fronts.h5')
print('loaded fronts model')

#load angled model
angled = load_model('/Users/justinchien/ds/metis/Project_Kojak/app/models/angled.h5')
print('loaded angled model')

#load BMW model
bmw_top = load_model('/Users/justinchien/ds/metis/Project_Kojak/app/models/bmws_top.h5')
print('loaded bmw top model')

#processing image for entry into convolution layer
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def prediction(image_path):

    wget.download(image_path, out = 'app/static/car/car.jpg')

    img = image.load_img('app/static/car/car.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    convoluted = convolutions.predict(x)
    prediction = front_or_angle_top.predict(convoluted)

    predictions = []

    for i in range(len(prediction[0])):
        predictions.append((i, prediction[0][i]))

    predictions.sort(key=lambda x: x[1], reverse = True)

    os.remove('app/static/car/car.jpg')

    #If angled view
    if predictions[0][1] > 0.5:
        print ('angled')
        #Send to Angled models for sangled
        angled_prediction = angled.predict(convoluted)
        angled_predictions = []

        for i in range(len(angled_prediction[0])):
            angled_predictions.append((i, angled_prediction[0][i]))

        angled_predictions.sort(key=lambda x:x[1], reverse = True)

        car_dict = {
            0:'aston',
            1:'audi',
            2:'bmw',
            3:'bugatti',
            4:'ferrari',
            5:'lamborghini',
            6:'mcLaren'
        }

        #if BMW, go to model designation
        if angled_predictions[0][0] == 2:
            print('BMW!')
            bmw_prediction = bmw_top.predict(convoluted)
            bmw_predictions = []

            for i in range(len(bmw_prediction[0])):
                bmw_predictions.append((i, bmw_prediction[0][i]))

            bmw_predictions.sort(key=lambda x: x[1], reverse = True)

            bmw_dict = {
                0:'2_Series',
                1:'3_Series',
                2:'4_Series',
                3:'5_Series',
                4:'6_Series',
                5:'7_Series',
                6:'i3',
                7:'i8'
            }

            result = (bmw_dict[bmw_predictions[0][0]], bmw_predictions[0][1]*100, bmw_dict[bmw_predictions[1][0]], bmw_predictions[1][1]*100)
            # print(bmw_dict[bmw_predictions[0][0]], bmw_predictions[0][1]*100)
            # print(bmw_dict[bmw_predictions[1][0]], bmw_predictions[1][1]*100)

        #Print top two predictions for front if not BMW
        else:
            result = (car_dict[angled_predictions[0][0]], angled_predictions[0][1]*100, car_dict[angled_predictions[1][0]], angled_predictions[1][1]*100)
            # print(car_dict[angled_predictions[0][0]], angled_predictions[0][1]*100)
            # print(car_dict[angled_predictions[1][0]], angled_predictions[1][1]*100)


    #If front view
    else:
        print ('front')

        #Send to top models for front
        front_prediction = fronts.predict(convoluted)
        front_predictions = []

        for i in range(len(front_prediction[0])):
            front_predictions.append((i, front_prediction[0][i]))

        front_predictions.sort(key=lambda x: x[1], reverse = True)

        car_dict = {
            0:'aston',
            1:'audi',
            2:'bmw',
            3:'bugatti',
            4:'ferrari',
            5:'lamborghini',
            6:'mcLaren'
        }

        #Print top two predictions for front
        result = (car_dict[front_predictions[0][0]], front_predictions[0][1]*100, car_dict[front_predictions[1][0]], front_predictions[1][1]*100)
        # print(car_dict[front_predictions[0][0]], front_predictions[0][1]*100)
        # print(car_dict[front_predictions[1][0]], front_predictions[1][1]*100)

    return (result)

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)

# Get an example and return it's score from the predictor model
@app.route("/", methods=["GET", "POST"])
def score():
    # """
    # When A POST request with json data is made to this uri,
    # Read the example from the json, predict probability and
    # send it with a response
    # """

    image_path = request.form.get("image_path", "")
    if image_path:
        result = prediction(image_path)
    else:
        result = None

    #Set color of first percentage
    if (result):
        if (result[1] >= 66):
            first_color = "#36c340"
        elif(66 > result[1] >= 33):
            first_color = "#d1cb35"
        elif(result[1] < 33):
            first_color = "#cc3c29"
    #Set color of second percentage
        if (result[3] >= 66):
            sec_color = "#36c340"
        elif(66 > result[3] >= 33):
            sec_color = "#d1cb35"
        elif(result[3] < 33):
            sec_color = "#cc3c29"
    #Set variables
        first = result[0]
        first_perc = math.ceil(result[1]*100.0)/100.0
        second = result[2]
        sec_perc = math.ceil(result[3]*100.0)/100.0
    else:
        first = None
        first_perc = None
        first_color = "#000000"
        second = None
        sec_perc = None
        sec_color = "#000000"

    return render_template("index.html", image_path=image_path,
                            first=first, first_perc=first_perc, first_color=first_color,
                            second=second, sec_perc=sec_perc, sec_color=sec_color)

# # Uploading file
# @app.route('/upload')
# def upload_file():
#    return render_template('index.html')
#
# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(secure_filename(f.filename))
#       return 'file uploaded successfully'

#--------- RUN WEB APP SERVER ------------#

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
