import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.lower()  # Optional: lowercase
    return text.strip()

with open('data/testdata.txt', 'r', encoding='utf-8') as f:
    content = f.read()

cleaned = clean_text(content)
with open('data/cleaned_text1.txt', 'w', encoding='utf-8') as f:
    f.write(cleaned)