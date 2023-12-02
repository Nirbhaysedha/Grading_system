import streamlit as st

class GradingSystem:
    def __init__(self, rubric):
        self.rubric = rubric

    def grade_submission(self, submission):
        total_score = 0
        max_score = 0
        for criterion, max_points in self.rubric.items():
            if criterion in submission:
                student_score = submission[criterion]
                total_score += min(student_score, max_points)
                max_score += max_points

        if max_score == 0:
            return 0  # To avoid division by zero
        percentage = (total_score / max_score) * 100
        return percentage

# Example rubric (criterion: maximum points)
assignment_rubric = {
    'Correctness': 20,
    'Completeness': 15,
    'Clarity': 10,
    'Creativity': 5
}

grading_system = GradingSystem(assignment_rubric)

st.title('Automated Grading System')

st.write("Enter Student Submission Scores:")
scores = {}
for criterion, max_points in assignment_rubric.items():
    score = st.slider(f"{criterion} Score (Max: {max_points})", 0, max_points, step=1)
    scores[criterion] = score

grade = grading_system.grade_submission(scores)

st.write(f"Student's Grade: {grade:.2f}%")
