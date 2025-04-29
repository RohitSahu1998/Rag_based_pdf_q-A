# Imports
import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# Build vector store
def build_vector_store(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
    return FAISS.from_documents(docs, embeddings)

# Create prompt template
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided document context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['context', 'question']
)

# Format docs into single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build main question-answering chain
def build_main_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain

# Streamlit App
def main():
    st.title("📄 PDF Question Answering App")

    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        with st.spinner('Reading and processing PDF...'):
            text = extract_text_from_pdf(pdf_file)
            docs = split_text(text)
            vector_store = build_vector_store(docs)
            main_chain = build_main_chain(vector_store)

        st.success("PDF uploaded and processed! You can now ask questions.")

        query = st.text_input("Ask a question about the PDF:")

        if query:
            with st.spinner('Generating answer...'):
                response = main_chain.invoke(query)
                st.write(response)

if __name__ == "__main__":
    main()
