import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

nlp = spacy.load("en_core_web_sm")

data = {
    "task": [
        "Finish the project report by end of the day",
        "Buy groceries for the week",
        "Schedule a meeting with the client",
        "Complete the exercise routine",
        "Call mom to wish her happy birthday",
        "Prepare slides for the presentation",
        "Book flight tickets for vacation",
        "Submit the quarterly financial report",
        "Respond to all pending emails",
        "Clean the house"
    ],
    "category": [
        "Work", "Personal", "Work", "Personal", "Personal", 
        "Work", "Personal", "Work", "Work", "Personal"
    ],
    "priority": [5, 2, 4, 3, 5, 4, 3, 5, 4, 2],
    "deadline": [
        "2024-10-08", "2024-10-12", "2024-10-09", "2024-10-08", 
        "2024-10-08", "2024-10-10", "2024-11-01", "2024-10-15", 
        "2024-10-08", "2024-10-14"
    ]
}

df = pd.DataFrame(data)
df['deadline'] = pd.to_datetime(df['deadline'])

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['task'])

y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Task Categorization Accuracy: {accuracy_score(y_test, y_pred)}")


def prioritize_task(row):
    today = pd.Timestamp(datetime.now())
    days_until_deadline = (row['deadline'] - today).days

    if days_until_deadline <= 0:
        urgency = 5  
    elif days_until_deadline <= 2:
        urgency = 4
    elif days_until_deadline <= 5:
        urgency = 3
    elif days_until_deadline <= 10:
        urgency = 2
    else:
        urgency = 1

    importance = row['priority']

    score = importance * 0.7 + urgency * 0.3
    return score


def get_user_tasks():
    tasks = []
    print("\nEnter your tasks (enter 'done' to finish):")

    while True:
        task = input("Task: ")
        if task.lower() == 'done':
            break

        category = input("Category (Work/Personal): ")
        priority = int(input("Priority (1-5): "))
        deadline = input("Deadline (YYYY-MM-DD): ")

        tasks.append({
            "task": task,
            "category": category,
            "priority": priority,
            "deadline": deadline
        })

    return pd.DataFrame(tasks)

user_df = get_user_tasks()

user_df['deadline'] = pd.to_datetime(user_df['deadline'])

user_df['score'] = user_df.apply(prioritize_task, axis=1)

user_df = user_df.sort_values(by='score', ascending=False)

print("\nPrioritized Tasks:")
print(user_df[['task', 'category', 'priority', 'deadline', 'score']])

def recommend_tasks(df, N=5):
    return df.head(N)

recommended_tasks = recommend_tasks(user_df)
print("\nRecommended Tasks for Today:")
print(recommended_tasks[['task', 'category', 'priority', 'deadline']])
