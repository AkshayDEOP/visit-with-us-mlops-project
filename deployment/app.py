import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
MODEL_REPO_ID="akshaychougule/visit-with-us-wellness-model"
@st.cache_resource
def load_model(): return joblib.load(hf_hub_download(repo_id=MODEL_REPO_ID, filename="model.joblib"))
model=load_model()
st.title("Visit with Us - Wellness Tourism Package Prediction")
age=st.number_input("Age",18,100,35); type_of_contact=st.selectbox("TypeofContact",["Self Enquiry","Company Invited"]); city=st.selectbox("CityTier",[1,2,3]); duration=st.number_input("DurationOfPitch",0.0,100.0,10.0); occupation=st.selectbox("Occupation",["Salaried","Small Business","Large Business","Free Lancer"]); gender=st.selectbox("Gender",["Male","Female"]); persons=st.number_input("NumberOfPersonVisiting",1,10,3); followups=st.number_input("NumberOfFollowups",0.0,10.0,3.0); product=st.selectbox("ProductPitched",["Basic","Deluxe","Standard","Super Deluxe","King"]); star=st.selectbox("PreferredPropertyStar",[3.0,4.0,5.0]); marital=st.selectbox("MaritalStatus",["Unmarried","Married","Divorced"]); trips=st.number_input("NumberOfTrips",0.0,30.0,2.0); passport=st.selectbox("Passport",[0,1]); score=st.selectbox("PitchSatisfactionScore",[1,2,3,4,5]); own=st.selectbox("OwnCar",[0,1]); children=st.number_input("NumberOfChildrenVisiting",0.0,5.0,1.0); designation=st.selectbox("Designation",["Executive","Manager","Senior Manager","AVP","VP"]); income=st.number_input("MonthlyIncome",0.0,1000000.0,25000.0)
input_df=pd.DataFrame([{"Age":age,"TypeofContact":type_of_contact,"CityTier":city,"DurationOfPitch":duration,"Occupation":occupation,"Gender":gender,"NumberOfPersonVisiting":persons,"NumberOfFollowups":followups,"ProductPitched":product,"PreferredPropertyStar":star,"MaritalStatus":marital,"NumberOfTrips":trips,"Passport":passport,"PitchSatisfactionScore":score,"OwnCar":own,"NumberOfChildrenVisiting":children,"Designation":designation,"MonthlyIncome":income}])
st.dataframe(input_df)
if st.button("Predict"):
    st.success(f"Prediction: {model.predict(input_df)[0]}, Purchase Probability: {model.predict_proba(input_df)[0][1]:.2%}")
