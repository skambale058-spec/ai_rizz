from transformers import pipeline

# Emotion detection model load karein
emotion_detector = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

text = input("Enter text: ")

results = emotion_detector(text)

print("\nDetected Emotions:")
for emotion in results[0]:
    print(f"{emotion['label']}: {emotion['score']:.4f}")
