import pandas as pd
import numpy as np
import re

# Load the dataset
df = pd.read_csv("data/final_data.csv")

# Function to convert item_weight to a consistent unit (kg)
def convert_weight_to_grams(weight):
    # Define conversion patterns for weight
    weight_patterns = {
        'lbs': 453.592,  # Pounds to grams
        'pounds': 453.592,
        'kg': 1000,  # kg to grams
        'kilogram': 1000,
        'g': 1,  # Grams to grams
        'gm': 1,
        'ounces': 28.3495  # Ounces to grams
    }

    match = re.match(r'(\d+(\.\d+)?)\s*(lbs|pounds|kg|kilogram|g|gm|ounces)', str(weight).lower())
    
    if match:
        value = float(match.group(1))
        unit = match.group(3)
        return value * weight_patterns[unit]
    else:
        return np.nan

# Function to clean and convert the final price to numeric values
def clean_final_price(price):
    try:
        cleaned_price = re.sub(r'[^\d\.\-]', '', str(price))
        return float(cleaned_price) if cleaned_price else np.nan
    except ValueError:
        return np.nan

# Function to compute product overview for each attribute
def get_product_overview(df):
    overview = {}

    # Clean and calculate overview for final_price
    df['final_price'] = df['final_price'].apply(clean_final_price)
    price_col = df['final_price']
    overview['final_price'] = {
        'max_price': price_col.max(),
        'min_price': price_col.min(),
        'average_price': price_col.mean(),
        'most_frequent_price': price_col.mode()[0] if not price_col.mode().empty else np.nan
    }

    # Calculate overview for item_weight
    df['item_weight_kg'] = df['item_weight'].apply(convert_weight_to_grams)
    overview['item_weight'] = {
        'max_weight': df['item_weight_kg'].max(),
        'min_weight': df['item_weight_kg'].min(),
        'average_weight': df['item_weight_kg'].mean(),
        'most_frequent_weight': df['item_weight_kg'].mode()[0] if not df['item_weight_kg'].mode().empty else np.nan
    }

    # Calculate overview for rating
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    rating_col = df['rating']
    overview['rating'] = {
        'max_rating': rating_col.max(),
        'min_rating': rating_col.min(),
        'average_rating': rating_col.mean(),
        'most_frequent_rating': rating_col.mode()[0] if not rating_col.mode().empty else np.nan
    }

    # Calculate availability breakdown
    availability_col = df['availability'].str.lower().map({'in stock': 1, 'out of stock': 0}).fillna(0)
    overview['availability'] = {
        'stock_count': availability_col.sum(),
        'out_of_stock_count': len(availability_col) - availability_col.sum(),
        'availability_percent': availability_col.sum() / len(availability_col) * 100
    }

    if 'department' in df.columns:
        df['department'] = df['department'].astype(str).apply(lambda x: re.sub(r'[\u200e\u200f\u202a-\u202e]', '', x))
        unique_departments = df['department'].dropna().unique()
        unique_departments = [dept.strip() for dept in unique_departments if isinstance(dept, str)]
        overview['all_unique_departments'] = list(unique_departments)


    return overview



def llm_context_with_overview():
    # Generate the product overview
    product_overview = get_product_overview(df)

    # Prepare the context for LLM
    llm_context = f"""
    - Final Price:
    - Maximum Price: ${product_overview['final_price']['max_price']:.2f}
    - Minimum Price: ${product_overview['final_price']['min_price']:.2f}
    - Average Price: ${product_overview['final_price']['average_price']:.2f}
    - Most Frequent Price: ${product_overview['final_price']['most_frequent_price']:.2f}

    - Item Weight:
        - Maximum Weight: {product_overview['item_weight']['max_weight']:.2f} grams
        - Minimum Weight: {product_overview['item_weight']['min_weight']:.2f} grams
        - Average Weight: {product_overview['item_weight']['average_weight']:.2f} grams
        - Most Frequent Weight: {product_overview['item_weight']['most_frequent_weight']:.2f} grams

    - Rating:
    - Maximum Rating: {product_overview['rating']['max_rating']}
    - Minimum Rating: {product_overview['rating']['min_rating']}
    - Average Rating: {product_overview['rating']['average_rating']:.2f}
    - Most Frequent Rating: {product_overview['rating']['most_frequent_rating']}

    - Availability:
    - In Stock: {product_overview['availability']['stock_count']} products
    - Out of Stock: {product_overview['availability']['out_of_stock_count']} products
    - Availability Percentage: {product_overview['availability']['availability_percent']:.2f}%

    - All Unique Departments:
    {product_overview.get('all_unique_departments', [])}
    """

    return llm_context
