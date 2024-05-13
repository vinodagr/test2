

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.title("My Streamlit App")
st.write("Welcome to campus placement prediction application!")
# Load data
df = pd.read_csv('mca_alumni.csv')

# Split data into input and output variables
X = df.drop(['Campus Placement'], axis=1)
y = df['Campus Placement']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit decision tree model to training data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on testing data
y_pred = model.predict(X_test)

# Calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)



def app():
 with st.form(key='placement-form1'):
    # Get input from user
    tenth = st.text_input('Enter your 10th percentage')
    tenth_board=st.selectbox('Select your 10th board 1-State , 2-CBSE, 3-ICSE', [1, 2,3])
    twelth = st.text_input('Enter your 12th percentage')
    twelth_board=st.selectbox('Select your 12th board 1-State , 2-CBSE, 3-ICSE', [1, 2,3])
    twelth_stream=st.selectbox('Select your 12th Stream 1-Science , 2-Commerce, 3-Arts', [1, 2,3])
    UG = st.text_input('Enter your UG percentage')
    UG_Course=st.selectbox('Select your Under Graduate Course 1-BCS , 2-BCA, 3-B.Sc. Comp.Sci, 4-B.Com, 5-B.Sc., 6-B.Sc. IT, 7-B.Sc. Bioinformatics', [1, 2,3,4,5,6,7])
    PG = st.text_input('Enter your PG percentage')
     
    Student_Category=st.selectbox('Select your Caste category 1-OPEN , 2-NT, 3-OBC , 4-SC, 5-SBC, 6-VJ, 7-ST', [1, 2,3,4,5,6,7])
    Certification = st.selectbox('Have you done any certification 1-Yes , 0-No', [1, 0])
    Extracurricular = st.selectbox('Have you participated in any extracurricular activity 1-Yes , 0-No', [1, 0])
    Backlogs = st.text_input('Do you have/had any backlogs, if yes write no of backlogs otherwise put 0 ')
    # Add button to make prediction
    submit_button = st.form_submit_button(label='Predict Placement')

    if submit_button:
     if not tenth:
      st.error('10th percentage is required!')
      return
     if not tenth.isdigit():
      st.error('Please enter a numeric value for the 10th percentage!')
      return
     if not twelth:
      st.error('12th percentage is required!')
      return
     if not twelth.isdigit():
      st.error('Please enter a numeric value for the 12th percentage!')
     if not UG:
      st.error('UG percentage is required!')
      return
     if not UG.isdigit():
      st.error('Please enter a numeric value for the UG percentage!')
     if not PG:
      st.error('PG percentage is required!')
      return
     if not PG.isdigit():
      st.error('Please enter a numeric value for the 10th percentage!')
      return
     if not Backlogs:
      st.error('No of backlogs required!')
      return
     if not Backlogs.isdigit():
      st.error('Please enter a numeric value for the npo of Backlogs!')
      return
        # Create input dataframe
     input_df = pd.DataFrame({'tenth':[tenth],'tenth_board':[tenth_board],'twelth': [twelth],'twelth_board':[twelth_board],'twelth_stream':[twelth_stream],'UG': [UG],'UG_Course': [UG_Course],'PG': [PG], 'Student_Category': [Student_Category],'Certification': [Certification],'Extracurricular': [Extracurricular],'Backlogs': [Backlogs]})
        
        # One-hot encode categorical variables
        # input_df['Gender'] = input_df['Gender'].map({'M': 1, 'F': 0})
        # input_df['UG_Course'] = input_df['UG_Course'].map({'1': 1, '2': 2,'3': 3})
        # input_df = pd.get_dummies(input_df, columns=['Gender'])

        # Reorder columns to match training data
     input_df = input_df.reindex(columns=X.columns, fill_value=0)

        # Make prediction on input data
     prediction = model.predict(input_df)

        # Print prediction
     if prediction == 1:
         st.write('You are more likely to get placed!')
     else:
         st.write('Sorry, you are less likely to get placed.')


# Run app
if __name__ == '__main__':
    app()


