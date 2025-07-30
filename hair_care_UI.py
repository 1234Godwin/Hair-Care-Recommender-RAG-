import gradio as gr
import re
import language_tool_python
from . import rag_chain


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

demo.launch(share=True)
