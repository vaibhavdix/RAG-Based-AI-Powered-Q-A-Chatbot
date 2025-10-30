import streamlit as st
import faiss # type: ignore
import os
import numpy as np
from config import EMBEDDING_MODEL,REPO_ID
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint # type: ignore
from dotenv import load_dotenv


load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

def process_data(input_type, input_data):
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        web_documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = [str(doc.page_content) for doc in text_splitter.split_documents(web_documents)]

    else:
        raise ValueError("Unsupported input type")


    model_name = EMBEDDING_MODEL
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_texts(texts)  
    return vector_store

def question_answer(vectorstore, query):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=REPO_ID,
            token=HUGGINGFACE_API_KEY,
            temperature=0.6,
            max_new_tokens=1000
        )
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        result = qa({"query": query})
        return result.get("result", "No answer found. The model might not have understood the question.")
    except Exception as e:
        return f"An error occurred: {str(e)}"


def main():
    st.title("Chat with URLs")
    st.write("Please provide links to ask questions based on the URL content.")

    input_data = None
    number_of_links = st.number_input("Number of Links", min_value=1, max_value=10, step=1)

    if number_of_links > 0:
        input_data = [st.text_input(f"Link {i+1}") for i in range(number_of_links)]
        input_data = [link for link in input_data if link.strip()]  

    if st.button("Process Input"):
        if not input_data:
            st.error("Please provide at least one valid link.")
        else:
            try:
                vectorstore = process_data("Link", input_data)
                st.session_state["vectorstore"] = vectorstore
                st.success("Input processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the input: {str(e)}")


    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question:")
        if st.button("Submit"):
            if query.strip():
                answer = question_answer(st.session_state["vectorstore"], query)
                st.markdown(f"*Answer:* {answer}")
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].append((query, answer))
            else:
                st.error("Please enter a valid question.")

        if "history" in st.session_state and st.session_state["history"]:
            with st.expander("Question-Answer History"):
                for i, (q, a) in enumerate(st.session_state["history"], 1):
                    st.markdown(f"*Q{i}:* {q}")
                    st.markdown(f"*A{i}:* {a}")


if __name__ == "__main__":
    main()