

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.title("Welcome to campus placement prediction application!")
st.write("Enter your academic details here")
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


passage="""
Persistence and Determination: Remain persistent in their job search efforts, despite facing rejections or challenges.Emphasize the importance of staying determined and not giving up, as success may come after multiple attempts.
Skill Improvement and Up skilling: Continuously work on improving their skills and gaining new experiences through internships, certifications, and online courses. Explore emerging technologies and acquiring relevant skills to stay competitive in the job market.
Networking and Referrals: Highlight the value of networking with professionals in their field of interest, both online (e.g., LinkedIn) and offline (e.g., industry events).Seek referrals from friends, family, or alumni who may have connections in companies they're interested in.
Tailoring Applications and Personalization: Customize resumes and cover letters for each job application, highlighting relevant experiences and skills. Research companies thoroughly and check their applications to match the company's culture and requirements.
Adaptability and Flexibility: Stress the importance of being flexible and open to various job opportunities, even if they are outside their initial preferences. Adapt to changing market conditions and consider relocation if necessary for better job prospects.
Utilization of Online Portals: Utilize online job portals effectively for job search activities, such as creating profiles, setting up job alerts, and applying to relevant positions regularly. Explore job portals specific to their industry or field of interest for additional opportunities.
Challenges of the Job Market: Be prepared for the challenges they may face in the job market, such as high competition, recession, or limited job opportunities in certain locations or industries.Strategies for overcoming these challenges, such as expanding their job search criteria, seeking temporary positions, or considering alternative career paths.
Location and Relocation: Relocation for job opportunities and be open to this option if it aligns with their career goals. Research potential locations for job prospects and considering factors such as cost of living, job market demand, and career growth opportunities.
Initial Rejections and Success over Time:Be prepared for initial rejections in their job search journey and reassure that success may come with persistence and continued effort. Learn from rejections, seek feedback when possible, and use those experiences to improve their future job applications and interviews.
Diversify Your Activities: Explore a variety of activities beyond academics, including extracurriculars, leadership roles, skill development workshops, and community engagement. This will help you develop a well-rounded skill set and enrich your college experience.
Take on Leadership Roles: Seek opportunities to take on leadership and organizational roles in clubs, societies, or student associations. This will not only enhance your leadership skills but also provide valuable experience in team management and event organization.
Invest in Skill Development: Participate in skill development workshops, technical events, and coding competitions to enhance your practical skills. Focus on acquiring both technical and soft skills that are relevant to your field of study or future career aspirations.
Build Your Network: Actively engage in networking opportunities through recruitment drives, departmental activities, and industry events. Networking can help you establish connections with professionals in your field, explore career opportunities, and gain valuable insights into industry trends.
Seek Recognition and Awards: Strive for excellence in your pursuits and actively pursue opportunities to gain recognition for your achievements. Whether it's winning competitions, receiving certifications, or being awarded for your contributions, recognition can boost your confidence and enhance your resume.
Balance Academic and Extracurricular Activities: While academics are important, don't neglect extracurricular activities and personal interests. Balance your time effectively to ensure you have a well-rounded college experience that includes both academic and non-academic pursuits.
Contribute to the Community: Get involved in community engagement initiatives, volunteer programs, or social gatherings that contribute positively to society. Engaging with your community not only helps others but also fosters personal growth and a sense of social responsibility.
Set Personal Goals and Prioritize Self-Development: Reflect on your interests, strengths, and areas for improvement, and set personal goals for self-development. Whether it's learning a new skill, pursuing a passion project, or taking on a leadership role, prioritize activities that align with your personal and professional aspirations.
By following these recommendations, current students can make the most of their college experience, develop valuable skills, expand their networks, and prepare themselves for future success in their academic and professional career.
Regular Profile Maintenance: Make it a habit to regularly update your professional profiles on platforms like LinkedIn, Naukri, and others. Ensure your profiles are complete, up-to-date, and reflect your current skills and experiences.
Engage Actively: Stay actively engaged with relevant content, discussions, and networking events on social media platforms. Participate in industry forums, groups, and online communities to showcase your expertise and expand your network.
Continuous Skill Development: Dedicate time consistently to enhance your skills through certifications, online courses, and practical projects. Identify emerging trends in your field and acquire the necessary skills to stay competitive in the job market.
Networking Persistence: Be proactive in networking by reaching out to professionals in your field, attending industry events, and joining relevant groups. Cultivate meaningful connections and maintain regular communication to expand your professional network.
Maintain a Positive Mindset: Job searching can be challenging, but maintaining a positive attitude and staying resilient in the face of rejections or setbacks is crucial. Stay focused on your goals, celebrate small victories, and keep pushing forward with determination.
Seek Guidance: Don't hesitate to seek guidance from mentors, career counselors, or alumni who can provide valuable insights and advice. Utilize resources available through your university or online platforms to gain guidance on career planning and job search strategies.
Stay Informed: Stay informed about industry trends, job market dynamics, and opportunities in your field. Keep yourself updated on the latest developments, technologies, and job requirements to align your skills and experiences accordingly.
"""


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
     if not (35 <= int(tenth) <= 100):
      st.error('Please enter the correct 10th percentage!')
      return
     if not twelth:
      st.error('12th percentage is required!')
      return
     if not twelth.isdigit():
      st.error('Please enter a numeric value for the 12th percentage!')
      return
     if not (35 <= int(twelth) <= 100):
      st.error('Please enter the correct 12th percentage!')
      return
     if not UG:
      st.error('UG percentage is required!')
      return
     if not UG.isdigit():
      st.error('Please enter a numeric value for the UG percentage!')
      return
     if not (35 <= int(UG) <= 100):
      st.error('Please enter the correct UG percentage!')
      return
     if not PG:
      st.error('PG percentage is required!')
      return
     if not PG.isdigit():
      st.error('Please enter a numeric value for the 10th percentage!')
      return
     if not (35 <= int(PG) <= 100):
      st.error('Please enter the correct 10th percentage!')
      return
     if not Backlogs:
      st.error('No of backlogs required!')
      return
     if not Backlogs.isdigit():
      st.error('Please enter a numeric value for the no of Backlogs!')
      return
     if not (0 <= int(Backlogs) <= 50):
      st.error('Please enter the correct BAcklogs value!')
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
         start = "\033[1m"
         end = "\033[0m"
         st.write('You are more likely to get placed! Still work on following recommendations based on MCA alumni past experiences for you as follows:')

         probability = model.predict_proba(input_df1)  # Replace with your actual method to get probabilities
        
            
         points = passage.strip().split('\n')
    
         # Print each point with a bullet
         for point in points:
          if point.strip():  # Skip empty lines
            st.write(f"- {point.strip()}")
         # Call the function to display the passage
     else:
         st.write('Sorry, you are less likely to get placed. Recommendations based on MCA alumni past experiences for you as follows:')
         points = passage.strip().split('\n')
    
         # Print each point with a bullet
         for point in points:
          if point.strip():  # Skip empty lines
            st.write(f"- {point.strip()}")
         # Call the function to display the passage
         


# Run app
if __name__ == '__main__':
    app()


