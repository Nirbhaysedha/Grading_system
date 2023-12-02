import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(df):
    # Features and target
    X = df.drop('Grade', axis=1)
    y = df['Grade']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model (Linear Regression as an example)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict grades
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mse, r2

def main():
    st.title('Automated Grading System')

    # Example dataset with student submissions and grades
    data = {
        'Correctness': [18, 15, 12, 20, 17],
        'Completeness': [12, 14, 13, 15, 11],
        'Clarity': [8, 9, 7, 10, 6],
        'Creativity': [4, 3, 5, 2, 4],
        'Grade': [85, 75, 70, 90, 80]  # Actual grades
    }

    df = pd.DataFrame(data)

    model, mse, r2 = train_model(df)

    st.write('Mean Squared Error:', mse)
    st.write('R-squared:', r2)

    st.write('Enter Student Submission Scores:')
    correctness = st.slider('Correctness Score', 0, 20, step=1)
    completeness = st.slider('Completeness Score', 0, 15, step=1)
    clarity = st.slider('Clarity Score', 0, 10, step=1)
    creativity = st.slider('Creativity Score', 0, 5, step=1)

    user_input = {
        'Correctness': correctness,
        'Completeness': completeness,
        'Clarity': clarity,
        'Creativity': creativity
    }

    grade_prediction = model.predict(pd.DataFrame([user_input]))
    st.write(f'Predicted Grade: {grade_prediction[0]:.2f}')


if __name__ == '__main__':
    main()
