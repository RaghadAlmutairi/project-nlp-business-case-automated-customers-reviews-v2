import streamlit as st
import pandas as pd
import openai
from sklearn.cluster import KMeans
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import joblib

# Set up OpenAI API key for sentiment classification

# Load the pre-trained T5 model and tokenizer for text summarization
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the pre-trained KMeans model (trained earlier)
kmeans = joblib.load('kmeans_model.joblib')

# Load Sentence-Transformer model for generating embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to classify sentiment using GPT-3
def classify_review(review):
    prompt = f"Classify the sentiment of the following review as Positive, Negative, or Neutral:\n\n{review}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5-turbo for chat-based completions
            messages=[{
                "role": "system", 
                "content": "You are a helpful assistant that classifies reviews into positive, neutral, or negative sentiments."
            }, {
                "role": "user", 
                "content": prompt
            }]
        )
        sentiment = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        sentiment = f"Error: {str(e)}"
    
    return sentiment


# Function to perform clustering using the pre-trained KMeans model
def cluster_reviews(data):
    predictions = kmeans.predict(data)
    return predictions


# Function for text summarization using T5 model
def summarize_review(text):
    input_ids = t5_tokenizer.encode(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    output = t5_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)


# Function to create the prompt for GPT-3 to generate category summary
def create_prompt_for_category(category, top_products_info):
    prompt = f"""
    You are a product reviewer and summary generator. 
    Please generate a detailed article for the product category: '{category}'.
    The article should include:
    1. The **top 3 products** in this category along with their **key differences**.
    2. A list of **top complaints** for each of these products.
    3. Identify the **worst product** in the category and explain why it should be avoided.
    4. Make the article concise, professional, and easy to understand.

    Here are the details about the top products in this category:
    {top_products_info}
    """
    return prompt


# Function to call OpenAI GPT-3 and generate summary for each category
def generate_summary_for_category(category, top_products_info):
    prompt = create_prompt_for_category(category, top_products_info)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the GPT chat model
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7  # Control the randomness of the output
    )
    
    return response['choices'][0]['message']['content'].strip()


# Function to process the dataset and generate summaries
def generate_summaries(df):
    summaries = {}
    
    for category in df['categories'].unique():
        category_df = df[df['categories'] == category]
        
        top_products_in_category = category_df.nlargest(3, 'reviews.rating')[['name', 'reviews.rating', 'reviews.text']]
        
        top_products_info = ""
        for _, row in top_products_in_category.iterrows():
            top_products_info += f"Product Name: {row['name']} | Rating: {row['reviews.rating']} | Review: {row['reviews.text'][:150]}...\n"
        
        summary = generate_summary_for_category(category, top_products_info)
        summaries[category] = summary
    
    return summaries


# Streamlit app UI
st.set_page_config(page_title="Product Review Dashboard", layout="wide")

# App title
st.title("📊 Product Review Dashboard")

# Sidebar for file upload
st.sidebar.header("Upload Your Reviews CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file with product reviews", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data preview section
    st.write("### Data Preview:")
    st.dataframe(df.head())

    # Select a category
    selected_category = st.sidebar.selectbox("Select a Product Category", df['categories'].unique())

    # Filter data based on selected category
    category_df = df[df['categories'] == selected_category]

    # Handle missing values
    category_df = category_df.dropna(subset=['reviews.rating'])

    # Display statistics
    st.write(f"### Statistics for '{selected_category}' Category:")
    st.write(f"**Average Rating**: {category_df['reviews.rating'].mean():.2f}")
    st.write(f"**Number of Reviews**: {len(category_df)}")



    # Review Classification and Sentiment Analysis
    st.subheader("Review Classification & Sentiment Analysis")
    for i, row in category_df.head(5).iterrows():
        review = row['reviews.text']
        sentiment = classify_review(review)
        st.write(f"**Review**: {review}")
        st.write(f"**Sentiment**: {sentiment}")
        st.write("---")

    # Clustering products
    st.subheader(f"Product Clustering for '{selected_category}'")
    reviews_data = category_df['reviews.text'].tolist()
    embeddings = embedder.encode(reviews_data)
    clustered = cluster_reviews(embeddings)
    category_df['Cluster'] = clustered
    st.write(category_df[['name', 'Cluster']])

    # Top 3 products
    st.subheader("Top 3 Products in the Category")
    top_products = category_df.nlargest(3, 'reviews.rating')[['name', 'reviews.rating']]
    st.write(top_products)

    # Summarizing reviews
    st.subheader("Review Summarization")
    for i, row in category_df.head(3).iterrows():
        review = row['reviews.text']
        summary = summarize_review(review)
        st.write(f"**Product**: {row['name']}")
        st.write(f"**Review Summary**: {summary}")
        st.write("---")

    # Category Summary Generated by GPT-3
    st.subheader(f"Category Summary Generated by GPT-3")
    category_summary = generate_summaries(df)
    st.write(category_summary[selected_category])
