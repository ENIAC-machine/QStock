# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("seara/rubert-base-cased-russian-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("seara/rubert-base-cased-russian-sentiment")

print(model(tokenizer('Всё хорошо')))
