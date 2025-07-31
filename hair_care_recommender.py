import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.url import UnstructuredURLLoader
import re
import gradio as gr
import language_tool_python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM



# Base blog page
base_url = "https://skinkraft.com/blogs/articles"

# Send HTTP request
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract all article links
article_links = []
for a_tag in soup.find_all("a", href=True):
    href = a_tag["href"]
    full_url = urljoin(base_url, href)
    if "/blogs/articles/" in full_url and full_url not in article_links:
        article_links.append(full_url)

print(f" Found {len(article_links)} article links.")


# Base path to crawl from
base_url = "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/"

# Fetch and parse page
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

# Extract all <a href> that start with base path
hair_links = []
for a_tag in soup.find_all("a", href=True):
    href = a_tag["href"]
    full_url = urljoin(base_url, href)

    # Keep only links under the same hair care path
    if full_url.startswith(base_url) and full_url not in hair_links:
        hair_links.append(full_url)

print(f" Found {len(hair_links)} hair-related URLs.")



# Starting page
base_page = "https://en.wikipedia.org/wiki/Hair_care"

response = requests.get(base_page)
soup = BeautifulSoup(response.text, "html.parser")

# Collect internal Wikipedia links that include 'hair'
hair_links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.startswith("/wiki/") and "hair" in href.lower():
        full_url = urljoin("https://en.wikipedia.org", href)
        hair_links.append(full_url)

hair_urls = hair_links.copy()
del hair_urls[2:7]
del hair_urls[2]
del hair_urls[12]           # Single item at index 12 (after first deletion, list is now shorter)
del hair_urls[-11]          # Index from the end
del hair_urls[-11]          # Same position after previous removal
del hair_urls[-23]          # Index from the end
del hair_urls[-24]          # Index from the end
del hair_urls[-8:-4]        # Slice near the end
del hair_urls[-1]
del hair_urls[7]
del hair_urls[1]
del hair_urls[5]
del hair_urls[3]
del hair_urls[-8]
del hair_urls[-6]

hair_urls[-5] = 'https://en.wikipedia.org/wiki/Human_hair_growth'
hair_urls.extend(['https://www.verywellhealth.com/wet-dandruff-treatment-5197087',
           'https://www.self.com/story/best-dandruff-shampoos',
            'https://www.allure.com/story/the-science-of-beauty-the-complete-guide-to-scalp-care'])


print(f" Found {len(hair_links)} hair-related URLs on Hair care page.")
print(f" {len(hair_urls)} collated URLs.")


# this combines all the urls retrieved from both websites
all_scraped_urls = hair_links + article_links + hair_urls




def fetch_clean_text_from_url(url):
    try:
        response = requests.get(url, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        return Document(page_content=text, metadata={"source": url})
    except Exception as e:
        print(f" Failed to fetch {url}: {e}")
        return None

# Scrape all documents from URLs
web_docs = [doc for url in all_scraped_urls if (doc := fetch_clean_text_from_url(url))]

print(f" Total successfully scraped documents: {len(web_docs)}")



def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[^A-Za-z0-9.,;:'\"()?!\s-]", "", text)
    return text


urls = [
    
        "https://drive.google.com/uc?export=download&id=1kmrqkdrXcf_0ki7Ut5Ho6pAOFdVsjPE7",
        "https://drive.google.com/uc?export=download&id=1HO7EKseK6TS_UofUbCoJgb9lakXT4sVj",
        "https://drive.google.com/uc?export=download&id=1cADfcpvSJdQqIgChIY_6KsGZkjjZ7VQ6",
        "https://drive.google.com/uc?export=download&id=1WQBSv_jodZsptXXnAhFczOITh1s9MeaK",
        "https://drive.google.com/uc?export=download&id=1Rl_CmIiaib6tIQm8mGbWohdtRwd6137s"
]

loader = UnstructuredURLLoader(
    urls=urls,
    mode="single",
    show_progress_bar=True
)

local_docs = loader.load()

# Clean loaded local_docs
for doc in local_docs:
    doc.page_content = clean_text(doc.page_content)

print(f" Loaded and cleaned {len(local_docs)} documents from {urls}")

# Combine both sources
all_docs = local_docs + web_docs

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = splitter.split_documents(all_docs)
print(f" Total document chunks: {len(all_splits)}")


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return Chroma(
        collection_name="hair_recommender_collection",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

vector_store = load_vector_store()

vector_store.add_documents(all_splits)

print(" Vector store successfully built and saved.")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model


# --- Truncation Helper ---
def truncate_input(text, tokenizer, max_tokens=512):
    # Truncate directly using tokenizer.encode and decode
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# --- HuggingFace Pipeline ---
tokenizer, model = load_model()
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    repetition_penalty=1.2,
)
llm = HuggingFacePipeline(pipeline=generator)


# --- Build RAG Chain with Retriever ---
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 2}),
    return_source_documents=True
)
print(" RetrievalQA pipeline successfully loaded.")

# --- System Prompt ---
system_prompt = (
    "You are an expert hair care assistant. "
    "Given the context from trusted medical or cosmetic sources, "
    "provide a clear, medically accurate, and helpful response. "
    "Avoid repeating the same text, remove unnecessary quotes, and make sure the answer is human-readable. "
    "Do not hallucinate. If the answer is not found in the context, reply: 'I'm not sure based on the current information.'"
)
print('all ran successfully')


# --- Prompt Formatter ---
def format_query(query, context_docs):
    combined_input = (
        system_prompt.strip() + "\n\n"
        + "Context:\n" + context_docs.strip() + "\n\n"
        + "Question: " + query.strip()
    )
    return truncate_input(combined_input, tokenizer)

# --- Answer Cleaning ---
def clean_answer(text):
    text = re.sub(r"\s{2,}", " ", text)  # Remove double spaces
    text = re.sub(r"\(.*?\)", "", text)  # Remove parentheses content
    text = re.sub(r"[\"\'\[\]`]", "", text)  # Strip extra quotation marks and brackets
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # Clean spacing before punctuation
    text = re.sub(r"(?<!\w)\s*[A-Z]{2,}\s*(?!\w)", "", text)  # Remove lone ALL CAPS
    return text.strip().capitalize()


def shorten_context(docs, max_chars=2000):
    combined = ""
    for doc in docs:
        if len(combined) + len(doc.page_content) > max_chars:
            break
        combined += doc.page_content + "\n\n"
    return combined.strip()



tool = language_tool_python.LanguageToolPublicAPI('en-US')

def correct_grammar(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

def fix_common_contractions(text):
    text = re.sub(r"\bIm\b", "I'm", text)
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bdont\b", "don't", text)
    text = re.sub(r"\bdoesnt\b", "doesn't", text)
    # Add more contractions as needed
    return text


def ask_rag(query):
    retrieved_docs = rag_chain.retriever.invoke(query)

    # Shorten the context to avoid long input
    context = shorten_context(retrieved_docs, max_chars=2000)

    # Format and truncate final input
    formatted_query = format_query(query, context)

    # Generate answer
    result = rag_chain.invoke(formatted_query)

    # Postprocess answer
    answer = clean_answer(result["result"])
    answer = fix_common_contractions(answer)         # <-- Add this
    answer = correct_grammar(answer)                 # <-- And this

    # Print
    print("🧠 Answer:", answer)
    print("\n📚 Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "No source"))


tool = language_tool_python.LanguageToolPublicAPI('en-US')

# -- Helper Functions --
def fix_common_contractions(text):
    text = re.sub(r"\bIm\b", "I'm", text)
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bdont\b", "don't", text)
    text = re.sub(r"\bdoesnt\b", "doesn't", text)
    # Add more as needed
    return text

def correct_grammar(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

def clean_answer(text):
    text = fix_common_contractions(text)
    text = correct_grammar(text)
    return text.strip()

# Optional: shorten long context
def shorten_context(docs, max_chars=2000):
    content = ""
    for doc in docs:
        if len(content) + len(doc.page_content) > max_chars:
            break
        content += doc.page_content + "\n\n"
    return content.strip()

# Format user query + system prompt
def format_query(query, context):
    return f"""You are a helpful hair care assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}

Helpful answer:"""

# Main query function
def answer_query(query):
    retrieved_docs = rag_chain.retriever.invoke(query)
    context = shorten_context(retrieved_docs)
    prompt = format_query(query, context)

    result = rag_chain.invoke(prompt)
    answer = clean_answer(result["result"])

    sources = "\n".join([doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])])
    return answer, sources

# -- Gradio UI --
with gr.Blocks(title="AI HAIR CARE RECOMMENDER", css="""
    body {
        background: linear-gradient(to right, #fff8f0, #fef6ec);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }
    .gr-button-primary {
        background-color: #6a5acd !important;
        color: white !important;
        font-size: 16px !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
    }
    .gr-button-primary:hover {
        background-color: #5a4bb5 !important;
    }
    .gr-button-secondary {
        background-color: #f08080 !important;
        color: white !important;
        font-size: 16px !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
    }
    .gr-button-secondary:hover {
        background-color: #e66767 !important;
    }
    textarea {
        font-size: 16px !important;
        padding: 12px;
        border-radius: 6px;
    }
""") as demo:

    gr.Markdown(
        """
        <div style='text-align: center; font-size: 32px; font-weight: bold; color: #6a5acd;'>AI HAIR CARE RECOMMENDER</div>
        <div style='text-align: center; font-size: 18px; margin-bottom: 20px; color: #444;'>
            Ask personalized questions about dandruff, hair loss, dryness, oily scalp, breakage, or growth. Get AI-powered recommendations with sources.
        </div>
        """
    )

    with gr.Row():
        input = gr.Textbox(label='Question', lines=5, placeholder="Ask your hair care question here...")
        output = gr.Textbox(label="AI's Recommendation", lines=6,placeholder="AI hair care assisstant's response")

    with gr.Row():
        sources_output = gr.Textbox(label="Sources", lines=4)

    with gr.Row():
        generate_btn = gr.Button("Generate Answer", elem_classes="gr-button-primary")
        clear_btn = gr.Button("Clear", elem_classes="gr-button-secondary")

    generate_btn.click(fn=answer_query, inputs=input, outputs=[output, sources_output])
    clear_btn.click(fn=lambda: ("", "", ""), outputs=[input, output, sources_output])

    gr.Markdown("#### Try asking:")
    gr.Examples(
        examples=[
            "How do I treat an oily scalp?",
            "Best practices for growing thick natural hair?",
            "How to stop breakage in relaxed hair?"
        ],
        inputs=input,
        label="Example Questions"
    )

if __name__ == "__main__":
    demo.launch()



