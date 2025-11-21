import transformers
import torch
import json

from data_loader import load_data

def load_model():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def prompt_model(pipeline, role_description, user_message):
    messages = [
        {"role": "system", "content": role_description},
        {"role": "user", "content": user_message},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]

if __name__ == "__main__":
    samples = load_data()
    
    # Load model once
    pipeline = load_model()
    results = []

    # Test prompt with sentiment analysis role
    role_with_sentiment_analysis = """
    You are an empathetic customer service assistant.

    Your task is to analyze the sentiment of the customer's message and respond appropriately.

    Sentiment categories: very_negative, negative, neutral, positive, very_positive

    Instructions:
    1. Identify the customer's sentiment
    2. Adjust your tone and response based on the sentiment:
    - very_negative/negative: Show extra empathy, apologize, and offer immediate solutions
    - neutral: Be helpful and professional
    - positive/very_positive: Match their enthusiasm and be encouraging

    Format your response exactly as follows:
    Sentiment: <sentiment>
    Response: <your response to the customer>

    Keep your response concise and professional.
    """

    # Test prompt without sentiment analysis role
    role_without_sentiment_analysis = """
    You are a customer service assistant.

    Respond to customer inquiries and provide accurate information.

    Keep your response concise and professional.
    """

    # Prompt model
    for _, row in samples.iterrows():
        id = row['id']
        text = row['text']
        label = row['label']
        label_text = row['label_text']

        response_with = prompt_model(pipeline, role_with_sentiment_analysis, text)
        response_without = prompt_model(pipeline, role_without_sentiment_analysis, text)
        
        print(f"WITH: {response_with}")
        print(f"WITHOUT: {response_without}\n")

        # Save both responses
        results.append({
            "id": id,
            "text": text,
            "label": label,
            "label_text": label_text,
            "response_with_sentiment": response_with,
            "response_without_sentiment": response_without
        })
    
    # Save to JSON file
    with open('output.json', 'w') as f:
        json.dump(results, f, indent=2)
