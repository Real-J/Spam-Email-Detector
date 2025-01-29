# Spam-Email-Detector
---

## 🚀 Features
✅ **Spam Classification** – Identifies whether an email is spam or not.  
✅ **Machine Learning Model** – Uses **TF-IDF Vectorizer** + **Multinomial Naïve Bayes**.  
✅ **Dataset Included** – 30+ spam email samples for training.  
✅ **Text Preprocessing** – Cleans email text (removes punctuation, numbers, and extra spaces).  
✅ **Simple API Function** – Check if an email is spam using `check_spam(email_text)`.  


---

## 🛠 Installation
1️⃣ Clone the repository:
```bash
git clone https://github.com/Real-J/spam-email-detector.git
cd spam-email-detector
```
2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```
3️⃣ Run the script:
```bash
python spam_detector.py
```

---

## 📝 Usage
You can test the spam detector by calling the `check_spam()` function.

```python
from spam_detector import check_spam

email_text = "Congratulations! You won a free iPhone. Click here to claim now!"
result = check_spam(email_text)
print("Email Classification:", result)  # Output: Spam
```

---

## 📊 Model Performance
- **Algorithm Used:** Multinomial Naïve Bayes  
- **Accuracy:** ~90% on sample dataset  
- **Evaluation:** `classification_report()` used for performance metrics  

---

## 🔥 Next Improvements
✔ Use **real-world datasets** (SpamAssassin, Enron Emails)  
✔ Improve model accuracy with **LSTMs or Transformers**  
✔ Deploy as a **Flask/FastAPI Web App**  

---

## 🤝 Contributing
Feel free to contribute! Fork the repo, create a branch, and submit a **Pull Request**.

---

## 🐝 License
This project is licensed under the **MIT License**.  

