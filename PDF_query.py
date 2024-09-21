import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def extract_pdf_text(pdf_files):
    combined_text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            combined_text += page.extract_text()
    return combined_text


def split_text_into_chunks(document_text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_fragments = splitter.split_text(document_text)
    return text_fragments


def create_vector_database(text_fragments):
    embedding_model = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(texts=text_fragments, embedding=embedding_model)
    return vector_db


def setup_conversation_chain(vector_db):
    language_model = ChatOpenAI()
    memory_buffer = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm=language_model,
        retriever=vector_db.as_retriever(),
        memory=memory_buffer
    )
    return convo_chain


def process_user_query(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.conversation_history = response['chat_history']

    for index, chat_message in enumerate(st.session_state.conversation_history):
        if index % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", chat_message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", chat_message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatbot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = None

    st.header("Chat with your PDFs :books:")
    user_input = st.text_input("Ask something about your uploaded PDFs:")
    if user_input:
        process_user_query(user_input)

    with st.sidebar:
        st.subheader("Upload your PDF files")
        uploaded_pdfs = st.file_uploader(
            "Upload PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing your files"):
                document_text = extract_pdf_text(uploaded_pdfs)
                text_chunks = split_text_into_chunks(document_text)
                vector_db = create_vector_database(text_chunks)
                st.session_state.conversation = setup_conversation_chain(vector_db)


if __name__ == '__main__':
    main()
