import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Expanded dataset with more spam emails
data = {
    'email': [
        "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now.",
        "Earn money fast! Work from home with this simple method.",
        "Get cheap medicines now. Special offer only for you.",
        "You have been selected for a cash prize. Click here to claim now!",
        "Urgent! Your bank account is locked. Verify your details immediately.",
        "Limited time offer! Buy 1 get 1 free on all purchases. Claim your reward.",
        "Winner! Your email has been chosen. Get your exclusive prize now.",
        "Investment opportunity: Turn $500 into $5000 in just a week!",
        "Your subscription is about to expire. Renew now to continue service.",
        "Free iPhone giveaway! Click the link to enter the contest today.",
        "Exclusive deal for you! 90% discount on luxury brands. Shop now!",
        "Make thousands from home. No experience required!",
        "Lottery alert! You have won $10,000. Claim immediately.",
        "Dear customer, your Amazon order is delayed. Click the link for refund.",
        "Hot singles in your area looking to chat now!",
        "Your PayPal account has been compromised. Click to secure it.",
        "Urgent security update required! Download the attachment now.",
        "Final reminder! Your loan is pre-approved. Apply today.",
        "Limited stock! Order now before it’s gone.",
        "Your credit score has been updated. Check it now for free.",
        "Your tax refund is ready! Claim your money today.",
        "Act now! This is your last chance to claim your reward.",
        "Your flight has been canceled. Click here to get compensation.",
        "Update your billing details to avoid service interruption.",
        "Win an all-expense-paid trip to Paris! Enter now.",
        "Flash sale! 80% off on all electronics for today only!",
        "Your delivery attempt failed. Reschedule here.",
        "Get rich quick! Learn how to make $1000/day from home.",
        "Special promotion: Get unlimited streaming for free!",
        "Activate your free Netflix account. Limited time only!",
        "Exclusive Bitcoin investment plan – triple your money now!",
        "Your resume has been selected for a high-paying job. Apply now!",
        "Your Social Security Number is at risk. Contact us now!"
    ],
    'label': [1] * 33  # All emails are spam (1)
}

df = pd.DataFrame(data)

# Text preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['email'] = df['email'].apply(clean_text)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, random_state=42)

# Create a spam detection pipeline (TF-IDF + Naïve Bayes classifier)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to check if an email is spam
def check_spam(email_text):
    email_text = clean_text(email_text)
    prediction = model.predict([email_text])
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test the spam detector
test_email = "Congratulations! You have been selected for an exclusive reward. Claim now!"
print("Test Email:", check_spam(test_email))
