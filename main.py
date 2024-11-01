import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from textblob import TextBlob
from datetime import datetime, timedelta
# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the DataFrame
try:
    df = pd.read_csv("Final_Dataset.csv")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Define categories to add
categories_to_add = ['Service Issue', 'Food Quality', 'Atmosphere', 'Value for Money', 'Hygiene', 'Food Options']

# Define the model architecture
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories_to_add))

# Load the model weights
try:
    model.load_weights('tf_model.h5')
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

def predict_multi_label(review_text):
    """Predicts the multi-label categories for a given review text."""
    inputs = tokenizer(review_text, truncation=True, padding=True, max_length=128, return_tensors='tf')
    outputs = model(inputs)
    predictions = tf.nn.sigmoid(outputs.logits).numpy()
    threshold = 0.7
    predicted_labels = [label for i, label in enumerate(categories_to_add) if predictions[0][i] >= threshold]
    if not predicted_labels:
        return [label for i, label in enumerate(categories_to_add) if predictions[0][i] == max(predictions[0])]
    else:
        return predicted_labels

# Function to calculate sentiment score
def get_sentiment(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

# Function to determine severity based on sentiment score
def get_severity(sentiment_score):
    if sentiment_score > 0.1:
        return "Low"
    elif sentiment_score >= -0.1:
        return "Medium"
    else:
        return "High"

# Function to determine urgency based on sentiment score
def get_urgency(sentiment_score):
    return 'Urgent' if sentiment_score < -0.1 else 'Non-Urgent'

def generate_response(categories, severity, urgency):
    base_response = (
        "Hi,\n"
        "We regret the inconveniences you have faced."
    )

    category_sentences = {
        'Service Issue': "We understand that service issues can be frustrating, and we are committed to resolving this.",
        'Food Options': "We appreciate your feedback on our food options, and we are continually working to enhance our menu.",
        'Food Quality': "We take food quality seriously, and we apologize for not meeting your expectations.",
        'Atmosphere': "Creating a pleasant atmosphere is important to us, and we are sorry that we fell short.",
        'Value for Money': "We strive to provide value for money, and we appreciate your input on this matter.",
        'Hygiene': "Hygiene is our top priority, and we are dedicated to maintaining the highest standards."
    }

    category_responses = [category_sentences[category] for category in categories if category in category_sentences]

    if severity == "High":
        severity_response = "We recognize the urgency of this matter."
    elif severity == "Medium":
        severity_response = "We acknowledge the issues you've raised."
    else:
        severity_response = "We appreciate your feedback."

    if urgency == "Urgent":
        urgency_response = "We will prioritize your concern and address it immediately."
    else:
        urgency_response = "We will take your feedback into consideration and work on improvements."

    full_response = f"{base_response} {' '.join(category_responses)} {severity_response}\n{urgency_response}"

   

    full_response += "\n\nWe request you to share your contact information at wecare@restaurant.com. We shall connect with you in no time and assist you with the issue raised. We assure you that this won't happen again next time.\n\nTeam Restaurant"

    return full_response

def get_complaint_info(df, days):
    # Filter for recent complaints based on Review_Days
    recent_complaints = df[df['Review_Days'] <= days]
    total_complaints = recent_complaints.shape[0]
    
    # Define the category columns you want to analyze
    category_columns = ['Service Issue', 'Food Quality', 'Atmosphere', 'Value for Money', 'Hygiene', 'Food Options']
    
    # Create a dictionary to hold counts for each category
    category_counts = {category: 0 for category in category_columns}
    
    # Count occurrences for each category based on boolean values (assumed 1 for complaint, 0 for no complaint)
    for category in category_columns:
        category_counts[category] = recent_complaints[category].sum()
    
    # Prepare the data for returning
    categories = ', '.join(category_counts.keys())
    counts = ', '.join(map(str, category_counts.values()))
    
    return {
        'total_complaints': total_complaints,
        'categories': categories,
        'counts': counts,
        'reviews': recent_complaints
    }
# Sidebar selection for user type
user_type = st.sidebar.radio("Select User Type", ("Customer", "Restaurant Owner"))

# Main content based on user type
st.title("Restaurant Complaint Management System")

if user_type == "Restaurant Owner":
    # Section for Restaurant Owner
    st.header("Complaint Counts Dashboard")
    days = st.number_input("Enter the number of days:", min_value=1, value=30)

    if st.button("Get Complaint Info"):
        if days <= 0:
            st.error("Please enter a positive number of days.")
        else:
            with st.spinner("Fetching complaint information..."):
                result = get_complaint_info(df, days)

            # Display results
            st.write("Total Complaints:", result['total_complaints'])

            # Calculate category counts from specific columns
            category_columns = ['Service Issue', 'Food Quality', 'Atmosphere', 'Value for Money', 'Hygiene', 'Food Options']
            category_counts = {col: result['reviews'][col].sum() for col in category_columns}

            st.write("Category Counts:")
            for category, count in category_counts.items():
                st.write(f"{category}: {count}")

            st.write("Reviews Information:")
            reviews_df = result['reviews']

            # Select relevant columns to display
            selected_columns = ['Review','Severity', 'Urgency', 'Predicted_Category']
            reviews_df = reviews_df[selected_columns]

            # Display the DataFrame
            st.dataframe(reviews_df.reset_index(drop=True))

if user_type == "Customer":
    # Section for Customer
    st.header("Submit Your Complaint")
    
    user_review = st.text_area("Enter your review:")
    submit_button = st.button("Submit Review")
    
    if submit_button and user_review.strip():
        with st.spinner("Processing your review..."):
            sentiment_score = get_sentiment(user_review)
            categories = predict_multi_label(user_review)
            severity = get_severity(sentiment_score)
            urgency = get_urgency(sentiment_score)
    
            # Generate auto-response
            auto_response = generate_response(categories, severity, urgency)
            
            # Display auto-response
            st.write(auto_response)
            
            # Record new entry in the DataFrame
            new_entry = pd.DataFrame({
                'Review': [user_review],
                'Predicted_Category': [', '.join(categories)],
                'Severity': [severity],
                'Urgency': [urgency],
                'Review_Days': [0],  # Set this to 0 for new entries; adjust based on your logic
                'Response_Days': [0],  # Adjust if needed
                # Include additional columns as necessary (e.g., Service Issue, Food Quality, etc.)
                'Service Issue': [int("Service Issue" in categories)],
                'Food Quality': [int("Food Quality" in categories)],
                'Atmosphere': [int("Atmosphere" in categories)],
                'Value for Money': [int("Value for Money" in categories)],
                'Hygiene': [int("Hygiene" in categories)],
                'Food Options': [int("Food Options" in categories)],
            })
            
            # Append and save the new review to the CSV file
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv("Final_Dataset.csv", index=False)
            st.success("Your review has been submitted and saved.")
            
        # # Optional display for confirmation
        # st.write("Review Summary:")
        # st.write(new_entry)
    else:
        st.info("Please enter a review to submit.")