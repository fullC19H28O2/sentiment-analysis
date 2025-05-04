import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt

df = pd.read_csv("yorumlar.csv")
yorumlar = df["yorum"].tolist()

model_adi = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_adi)
model = AutoModelForSequenceClassification.from_pretrained(model_adi)

duygu_modeli = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sonuclar = duygu_modeli(yorumlar)

df["Etiket"] = [s["label"] for s in sonuclar]
df["Skor"] = [s["score"] for s in sonuclar]

etiket_map = {
    "1 star": "Olumsuz",
    "2 stars": "Olumsuz",
    "3 stars": "Nötr",
    "4 stars": "Olumlu",
    "5 stars": "Olumlu"
}
df["Duygu"] = df["Etiket"].map(etiket_map)

print(df)

df["Duygu"].value_counts().plot(kind="bar", title="Duygu Dağılımı")
plt.xlabel("Duygu")
plt.ylabel("Yorum Sayısı")
plt.show()