from transformers import pipeline

def detect_ai_generated(text):
    """
    Detects if the text is AI-generated using a pre-trained model.
    """
    if not text or not text.strip():
        return "Error: Empty text provided.", 0.0

    print("Loading model... (this might take a while on the first run)")
    try:
        # Load the pipeline for text classification
        # We use a model specifically trained for AI detection
        pipe = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")
        
        # Get result
        result = pipe(text)
        label = result[0]['label']
        score = result[0]['score']
        
        return label, score
    except Exception as e:
        return f"Error: {str(e)}", 0.0

if __name__ == "__main__":
    print("AI Text Detector")
    print("-" * 20)
    
    # Sample text (You can change this or add input prompt)
    sample_text = input("Enter text to analyze: ")
    
    if sample_text:
        label, score = detect_ai_generated(sample_text)
        print(f"\nResult: {label}")
        print(f"Confidence: {score:.4f}")
    else:
        print("No text entered.")
