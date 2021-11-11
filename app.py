#!/usr/bin/env python
# coding: utf-8

# In[4]:


from pycaret.regression import * 
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import os


# In[9]:


class StreamlitApp:
    
    def __init__(self):
        self.model = load_model('model/Final_Huber_Model_10Nov2021') 
        self.save_fn = 'path.csv'     
        
    def predict(self, input_data): 
        return predict_model(self.model, data=input_data)
    
    def store_prediction(self, output_df): 
        if os.path.exists(self.save_fn):
            save_df = pd.read_csv(self.save_fn)
            save_df = save_df.append(output_df, ignore_index=True)
            save_df.to_csv(self.save_fn, index=False)
            
        else: 
            output_df.to_csv(self.save_fn, index=False)  
            
    
    def run(self):
        image = Image.open('../assets/pretty_house.jpg')
        st.image(image, use_column_width=False)
    
    
        add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'Batch')) 
        st.sidebar.info('This app is created to predict house prices' )
        st.sidebar.success('DAT158')
        st.title('House prices prediction')
        
       
        if add_selectbox == 'Online': 
            OverallQual = st.number_input('OverallQual', min_value=1, max_value=10, value=5)
            MSZoning = st.selectbox('MSZoning', ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP','RM'])
            GrLivArea = st.number_input('GrLivArea', min_value=300, max_value=6000, value=1500)
            OverallCond = st.number_input('OverallCond', min_value=1, max_value=9, value=5)            
            GarageType = st.selectbox('GarageType', ['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'])
            LandSlope = st.selectbox('LandSlope', ['Gtl', 'Mod', 'Sev'])
            FullBath = st.number_input('FullBath', min_value=0, max_value=5, value=0)  
            Neighborhood = st.selectbox('Neighborhood', ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 
                                                         'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 
                                                         'MeadowV', 'Mitchel', 'Names','NoRidge', 'NPkVill', 
                                                         'NridgHt', 'NWAmes', 'OldTown','SWISU', 'Sawyer', 
                                                         'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker', 'Up', 'Down'])
            Functional = st.selectbox('Functional', ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'])
            
            
            output=''
            input_dict = {'OverallQual':OverallQual, 'MSZoning':MSZoning, 'GrLivArea':GrLivArea, 'OverallCond':OverallCond, 
                          'GarageType':GarageType, 'LandSlope':LandSlope, 
                          'FullBath':FullBath, 'Neighborhood':Neighborhood, 'Functional':Functional}
            input_df = pd.DataFrame(input_dict, index=[0])
        
            if st.button('Predict'): 
                output = self.predict(input_df)
                self.store_prediction(output)
                
                #output = 'Heart disease' if output['Label'][0] == 1 else 'Normal'
                #output = str(output['Label'])
                
            
            st.success('Predicted output: {}'.format(output))
            
        if add_selectbox == 'Batch': 
            fn = st.file_uploader("Upload csv file for predictions") #st.file_uploader('Upload csv file for predictions, type=["csv"]')
            if fn is not None: 
                input_df = pd.read_csv(fn)
                predictions = self.predict(input_df)
                st.write(predictions)
            
sa = StreamlitApp()
sa.run()


# In[ ]: