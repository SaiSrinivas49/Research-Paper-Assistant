
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper Assistant")
st.caption("Ask questions about research papers using Retrieval-Augmented Generation.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

with st.sidebar:
    st.header("Upload a paper")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
    )

    if st.button("Process PDF", disabled=uploaded_file is None):
        with st.spinner("Extracting text and building the search index..."):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

                st.session_state.uploaded_filename = result["filename"]
                st.session_state.chat_history = []

                st.success(
                    f"Indexed {result['chunks']} chunks successfully!"
                )
                st.rerun()

            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

    if st.session_state.uploaded_filename:
        st.success(f"Active paper: {st.session_state.uploaded_filename}")

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# Display previous messages.
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):
        st.markdown(item["answer"])

        if item["sources"]:
            with st.expander("View retrieved sources"):
                for source in item["sources"]:
                    st.write(
                        f"- {source['file']} — Page {source['page']}"
                    )

if not st.session_state.uploaded_filename:
    st.info("Upload and process a research paper to start asking questions.")

question = st.chat_input(
    "Ask something about your paper...",
    disabled=not bool(st.session_state.uploaded_filename),
)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the paper and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("View retrieved sources"):
                        for source in result["sources"]:
                            st.write(
                                f"- {source['file']} — Page {source['page']}"
                            )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })

            except requests.RequestException as exc:
                st.error(f"Could not get an answer: {exc}")
