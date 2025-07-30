# AI Hair Care Recommender (RAG-Based)
A Retrieval-Augmented Generation (RAG) system designed to provide personalized and medically accurate hair care advice using trusted sources, HuggingFace models, LangChain retrievers, and an intuitive Gradio UI.

## ❓ What It Is & Why It Matters
Give users science-backed guidance on common hair concerns (e.g. dryness, breakage, dandruff). Combines trusted web-sourced medical/cosmetic info with LLM-based responses to reduce misinformation and provide trustworthy solutions.

## 🌟 Features
🧠 AI-powered answers tailored to hair issues like dandruff, hair loss, dryness, and breakage.

🔍 Web scraping & context building from trusted dermatological and hair care sites.

🔗 LangChain RAG pipeline with vector search + HuggingFace FLAN-T5 generation.

✨ Automatic grammar correction and cleanup for human-readable, medically safe advice.

🖥️ Interactive Gradio interface for users to ask questions and receive source-backed answers.

## 📊 Tech Stack

LangChain for RAG pipeline (Retriever + Generator)

FLAN-T5 (Large) via HuggingFace Transformers

BeautifulSoup for document scraping

Chroma  for vector store & similarity search

language_tool_python for grammar correction

Gradio for frontend interface

## ⚙️ Installation & Running

git clone https://github.com/1234Godwin/Hair-Care-Recommender-RAG-.git
cd Hair-Care-Recommender-RAG-
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

## 🖥️ Basic Usage Example
Query: How can I stop breakage in relaxed hair?
Response: “Use protein treatments sparingly, moisturize regularly, and avoid frequent heat styling. Ingredients like panthenol or ceramides can help reduce breakage.”
📚 You'll also see source URLs from reliable medical or cosmetic sites.

## 📦 Dependencies & Setup Requirements

Make sure your requirements.txt includes at least:
langchain, langchain-core
transformers, huggingface-hub
beautifulsoup4, requests
chromadb or faiss
language-tool-python
gradio
Ensure you have downloaded and pointed to the local FLAN‑T5 Large model directory.

## 🧩 Extending the Project
Integrate collaborative filtering (user ratings) or content-based similarity for hybrid recommendations.

Add more hair indicators (elasticity, porosity depth) from surveys.

Explore fine‑tuning LLMs or using prompt templates for richer explanations.

Package the RAG model + pipeline into a web or mobile app for user interaction.

## 👤 Author
Author: Chiemelie Onu / https://github.com/1234Godwin

Feel free to file issues or pull requests to improve scraping logic, extend UI features, or swap models.

Enjoy exploring hair-care recommendations powered by data and RAG! 💇‍♀️✨



