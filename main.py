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

def prompt_model_with_examples(pipeline, role_description, examples, user_message):
    """Few-shot prompting with conversation examples"""
    messages = [{"role": "system", "content": role_description}]
    
    # Add few-shot examples as conversation history
    for example in examples:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": example["output"]})
    
    # Add actual user message
    messages.append({"role": "user", "content": user_message})
    
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

    # BASELINE: Test prompt without sentiment analysis role
    role_without_sentiment_analysis = """
    You are a customer service assistant.

    Respond to customer inquiries and provide accurate information.

    Keep your response concise and professional.
    """

    # ZERO-SHOT COT: Prompt with Chain-of-Thought reasoning
    role_zero_shot_cot = """
    You are an empathetic customer service assistant.

    For each customer message, think step-by-step:
    1. What is the customer's emotional state?
    2. What is their main concern or request?
    3. What would be the most helpful response?

    Let's work through this step by step, then provide your final response.

    Format:
    Reasoning: <your step-by-step analysis>
    Sentiment: <very_negative/negative/neutral/positive/very_positive>
    Response: <your final response to the customer>

    Keep your response concise and professional.
    """

    # FEW-SHOT COT: Provide examples with reasoning
    role_few_shot_cot = """
    You are an empathetic customer service assistant.

    Analyze the sentiment and respond appropriately using step-by-step reasoning.

    Format:
    Reasoning: <your step-by-step analysis>
    Sentiment: <very_negative/negative/neutral/positive/very_positive>
    Response: <your final response to the customer>
    """

    # Few-shot examples demonstrating reasoning
    few_shot_examples = [
        {
            "input": "This product is absolutely terrible! It broke after one day. I want a refund immediately!",
            "output": """Reasoning: 
            1. Emotional state: The customer is very angry and frustrated
            2. Main concern: Product broke quickly, wants refund
            3. Best response: Apologize sincerely, show empathy, offer immediate solution

            Sentiment: very_negative
            Response: I sincerely apologize for this frustrating experience. It's completely unacceptable that the product broke after just one day. I'll process your refund immediately and arrange for a prepaid return label. We'll also investigate this quality issue. Is there anything else I can help you with today?"""
        },
        {
            "input": "I'm interested in learning more about your warranty options.",
            "output": """Reasoning:
            1. Emotional state: Neutral
            2. Main concern: Wants warranty information
            3. Best response: Provide clear, helpful information

            Sentiment: neutral
            Response: I'd be happy to explain our warranty options. We offer a standard 1-year warranty covering manufacturing defects, and extended warranty plans for 2 or 3 years. What product are you interested in? I can provide specific details for that item."""
        },
        {
            "input": "Thank you so much! Your team has been incredibly helpful!",
            "output": """Reasoning:
            1. Emotional state: Very positive
            2. Main concern: Expressing appreciation
            3. Best response: Match their enthusiasm, reinforce positive experience

            Sentiment: very_positive
            Response: That's wonderful to hear! We're thrilled we could help you. Your satisfaction means everything to us. Please don't hesitate to reach out if you need anything else in the future. Thank you for choosing us!"""
        }
    ]

    # Prompt model with all three approaches
    for _, row in samples.iterrows():
        id = row['id']
        text = row['text']
        label = row['label']
        label_text = row['label_text']

        print(f"\n{'='*80}")
        print(f"Customer Message: {text}")
        print(f"Actual Sentiment: {label_text}")
        print(f"{'='*80}")

        # Baseline (without sentiment analysis)
        response_baseline = prompt_model(pipeline, role_without_sentiment_analysis, text)
        print(f"\nBASELINE:\n{response_baseline['content']}")

        # Zero-Shot CoT
        response_zero_shot = prompt_model(pipeline, role_zero_shot_cot, text)
        print(f"\nZERO-SHOT COT:\n{response_zero_shot['content']}")

        # Few-Shot CoT
        response_few_shot = prompt_model_with_examples(
            pipeline, 
            role_few_shot_cot, 
            few_shot_examples, 
            text
        )
        print(f"\nFEW-SHOT COT:\n{response_few_shot['content']}")

        # Save all responses
        results.append({
            "id": id,
            "text": text,
            "label": label,
            "label_text": label_text,
            "response_baseline": response_baseline,
            "response_zero_shot_cot": response_zero_shot,
            "response_few_shot_cot": response_few_shot
        })
    
    # Save to JSON file
    with open('output.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to output.json")
    print(f"{'='*80}")
