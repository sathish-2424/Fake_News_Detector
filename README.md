# 📰 Fake News Detection Using BERT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![BERT](https://img.shields.io/badge/Model-BERT-green.svg)](https://huggingface.co/bert-base-uncased)

An end-to-end NLP pipeline to detect fake news using the BERT transformer model with an interactive web interface.

![Fake News Detection Demo](./screenshot/image1.png)

![Main Interface](./screenshot/image2.png)

## 📌 Project Overview

This project leverages **BERT (Bidirectional Encoder Representations from Transformers)** to classify news articles as **REAL** or **FAKE**. The system provides:

- 🧠 **Deep Learning Classification**: Fine-tuned BERT model for accurate predictions
- 📊 **Confidence Scoring**: Probability-based confidence metrics
- 🌐 **Interactive Web Interface**: User-friendly Streamlit application
- 📁 **Batch Processing**: Analyze multiple articles simultaneously
- 📈 **Real-time Visualization**: Interactive charts and confidence gauges
- 🔍 **Model Explainability**: Understanding prediction reasoning

## 🧠 Technologies Used

| Tool | Purpose | Version |
|------|---------|---------|
| **PyTorch** | Deep learning framework | 2.0.1 |
| **Transformers** | BERT implementation | 4.33.0 |
| **Streamlit** | Web interface | 1.28.0 |
| **scikit-learn** | ML metrics and utilities | 1.3.0 |
| **Plotly** | Interactive visualizations | 5.16.0 |
| **Pandas** | Data manipulation | 2.0.3 |

## 🗂️ Project Structure

```
fake-news-detection-bert/
├── 📄 streamlit_app.py
├── 🧠 fake_news_bert.py
├── 📊 data/
│ ├── fake_or_real_news.csv
│ └── sample_data.csv
├── 🔧 model/
│ └── best_model.pth
├── 📋 requirements.txt
├── 🐳 Dockerfile
├── ⚙️ .streamlit/
│ └── config.toml
└── 📖 README.md
```

## 🚀 Quick Start

### 🔧 1. Clone the Repository
```bash
git clone https://github.com/your-username/fake-news-detection-bert.git
cd fake-news-detection-bert
```

### 📦 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
```

### 📥 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 🏃‍♂️ 4. Run the Application
```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

## 🧪 Training the Model

### Run Basic Training
```bash
python fake_news_bert.py
```

### Configure Parameters
```python
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
```

## 🔍 Making Predictions

### Single Prediction Example
```python
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Your news article text here..."
prediction, confidence = predict_with_confidence(model, text, tokenizer)

print(f"Prediction: {'REAL' if prediction == 1 else 'FAKE'}")
print(f"Confidence: {confidence:.2%}")
```

### Batch Prediction
```python
import pandas as pd

df = pd.read_csv('news_articles.csv')
results = []

for text in df['text']:
pred, conf = predict_with_confidence(model, text, tokenizer)
results.append({'prediction': pred, 'confidence': conf})

pd.DataFrame(results).to_csv('predictions.csv', index=False)
```

## 🌐 Streamlit App Features

| Feature | Description |
|---------|-------------|
| 📝 **Text Analysis** | Input-based real-time classification |
| 📊 **Confidence Gauge** | Visual prediction score |
| 📁 **Batch Upload** | Upload CSV for bulk predictions |
| 📈 **Usage Stats** | Tracks prediction usage |
| 🎯 **Sample Data** | Try out with sample entries |
| 💾 **CSV Export** | Download analyzed results |

## ☁️ Deployment Options

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### Docker
```bash
docker build -t fake-news-app .
docker run -p 8501:8501 fake-news-app
```

## 🧪 Testing

```bash
python -m pytest tests/
```

## 📈 Future Enhancements

- [ ] SHAP/LIME model interpretability
- [ ] Multi-language support
- [ ] API + browser extension
- [ ] Mobile and RESTful API versions

## 🤝 Contributing

1. Fork and clone the repo
2. Create a branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Open a Pull Request

## ⚖️ License

Licensed under [MIT License](LICENSE).

## 🙏 Acknowledgments

- HuggingFace Transformers
- Streamlit.io
- Kaggle dataset
- Open-source NLP community

## 📞 Contact

- Author: sathish kumar
- Email: Sathish9268@gmail.com
- LinkedIn: https://www.linkedin.com/in/sathishkumar32/
- GitHub: [https://github.com/yourusername](https://github.com/sathish-2424)

---
