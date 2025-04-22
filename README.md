# 🧠 Mental Therapy Chatbot
A supportive mental health chatbot that leverages AI to provide empathetic, evidence-based responses. Built using LangChain, FAISS, and Google's Generative AI, it offers a ChatGPT-style interface with adjustable response depth and creativity.

## 🌟 Features
Conversational Interface: Engage in meaningful dialogues about mental health.

Adjustable Response Depth: Tailor the depth of responses to your preference.

Creativity Control: Modify the creativity level of the chatbot's replies.

PDF Integration: Incorporate therapeutic materials from PDFs into the chatbot's knowledge base.

Emotion Detection: Analyze and respond to user emotions effectively.

## 🛠️ Tech Stack
Frontend: Streamlit for an intuitive user interface.

Backend:

LangChain: Framework for building language model applications.

FAISS: Efficient similarity search and clustering of dense vectors.

Google Generative AI: For generating embeddings and conversational responses.

Data Processing:

PyPDF2: Extract text from PDF files.

Pandas: Data manipulation and analysis.

NumPy: Numerical computations.

Environment Management: Python's venv and dotenv for virtual environments and environment variables.

## 🚀 Getting Started
Prerequisites
Python 3.10+

Google Cloud account with access to Generative AI APIs

Installation
Clone the Repository:
```
git clone https://github.com/Manik178/Mental_Therapy_chatbot.git
cd Mental_Therapy_chatbot
```
Set Up Virtual Environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install Dependencies:
```
pip install -r requirements.txt
```
Configure Environment Variables:

Create a .env file in the root directory.

Add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

Prepare PDF Documents:

Place your therapeutic PDF documents in the Lang_approach/src/doc_pdf directory.

Run the Application:
```
streamlit run Lang_approach/src/script.py
'''

## 🧪 Emotion Detection
The repository includes an emotion-detection-final.ipynb notebook that demonstrates how to detect emotions from text inputs. It utilizes machine learning techniques to classify emotions, enhancing the chatbot's ability to respond empathetically.
```
📁 Directory Structure
```
Mental_Therapy_chatbot/
├── Lang_approach/
│   └── src/
│       ├── doc_pdf/               # Directory for PDF documents
│       ├── script.py              # Main Streamlit application
│       └── ...                    # Additional source files
├── emotion-detection-final.ipynb  # Emotion detection notebook
├── train.csv                      # Dataset for training emotion detection model
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```