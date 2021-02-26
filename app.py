from pycaret.anomaly import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('traffic')

def predict_quality(model, input_df):
    
    predictions_data = predict_model(estimator = model, data =input_df)
    predictions = predictions_data['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('Bear.png')

    st.image(image,use_column_width=False)
    add_selectbox = st.sidebar.selectbox("How would you like to detect anomaly?",("Online","Batch"))
    st.sidebar(image)

    st.title("Traffic Anomaly Detection App")
    st.write('This is a web app to anomaly detection of your data based on\
    several features that you can see in the sidebar. Please adjust the\
    value of each feature. After that, click on the Detect button at the bottom.')
    if add_selectbox == 'Online':
        if st.checkbox('Is that day a holiday?'):
            holiday = 'yes'
        else:
            holiday = 'no'
        temp = st.sidebar.slider(label = 'Temperature', min_value = 243.00,
                          max_value = 373.00 ,
                          value = 274.00,
                          step = 1.00)
        rain_1h	 = st.sidebar.slider(label = 'Rainy Hour', min_value = 0.00,
                          max_value = 6.00 ,
                          value = 1.00,
                          step = 0.5)
                          
        snow_1h	 = st.sidebar.slider(label = 'Snowy Hour',  min_value = 0.00,
                          max_value = 6.00 ,
                          value = 1.00,
                          step = 0.5)                          

        clouds_all = st.sidebar.slider(label = 'Cloud Percentage', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 10.0,
                          step = 1.0)
        wheather_list=['Clouds','Mist','Fog','Rain','Snow','Thunderstorm','Haze','Drizzle','Clear','']
        weather_main = st.selectbox('Select the weather condition',wheather_list)
   
        Rush_Hour = st.sidebar.slider(label = 'Rush Hour', min_value = 0.00,
                          max_value = 6.00 ,
                          value = 0.00,
                          step = 1.0)

        traffic_volume = st.sidebar.slider(label = 'Traffic Volume', min_value = 1000,
                          max_value = 6000 ,
                          value = 150,
                          step = 10)

        features = {'Holiday': holiday,'Temperature': temp, 'Rainy Hour': rain_1h,
            'Snowy Hour': snow_1h	, 'Cloud Percentage': clouds_all,
            'Weather Condition': weather_main, 'Rush Hour': Rush_Hour,
            'Traffic Volume': traffic_volume
            }
        input_df = pd.DataFrame([features])
            
        if st.button("Detect"):
            out = predict(model=model,input_df=input_df)
        st.success(out='There is ' + str(out) +'anomalies detected')
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
if __name__ == '__main__':
    run()
