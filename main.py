import sys
from data_loader import load_data, clean_text
from model import SpamClassifier

def print_banner():
    print("=" * 50)
    print(" SMS SPAM CLASSIFIER")
    print(" Powered by TF-IDF & Naive Bayes")
    print("=" * 50)

def main():
    print_banner()
    print("\nLoading dataset...")
    
    try:
        df = load_data()
        print(f"Loaded {len(df)} SMS messages.")
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)

    print("Training the model...")
    classifier = SpamClassifier()
    classifier.train(df)
    
    # Print evaluation metrics
    classifier.evaluate(df)
    
    print("Model is ready!\n")
    
    print("Type an SMS message to check if it's SPAM or HAM.")
    print("Type 'exit' or 'quit' to close the program.\n")
    
    while True:
        try:
            user_input = input("Enter message: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            if not user_input:
                continue
                
            # Preprocess the same way training data is preprocessed
            cleaned_input = clean_text(user_input)
            
            # Predict
            prediction = classifier.predict(cleaned_input)
            
            # Display result
            if prediction == 'spam':
                print(f"--> Result: [ SPAM ] 🚨 Be careful!")
            else:
                print(f"--> Result: [ HAM ] ✅ Safe message.")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
