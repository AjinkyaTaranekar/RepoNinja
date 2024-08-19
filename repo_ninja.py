import os
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from llama_index.readers.github import GithubClient, GithubRepositoryReader

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a code analysis assistant with deep understanding of software development, capable of interpreting complex codebases.

Given the following context, provide a precise and detailed answer to the question. Ensure your response is clear, concise, and directly addresses the question.

Context:
{context}

---

Question: {question}

Answer:
"""


class RepoNinja:
    def __init__(
        self,
        owner: str,
        repo: str,
        branch: str,
        directories: str,
        model_name: str = "llama3.1",
    ):
        start_time = datetime.now()
        self.model_name = model_name
        github_token = os.environ.get("GITHUB_TOKEN")
        github_client = GithubClient(github_token=github_token, verbose=True)

        documents = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=True,
            verbose=False,
            filter_directories=(
                directories.split(","),
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            filter_file_extensions=(
                [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".svg",
                    ".ico",
                    ".json",
                    ".ipynb",
                ],
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
        ).load_data(branch=branch)

        print(f"🐙 Github Repo {owner}/{repo} loaded.")
        chunks = self.split_documents(
            [
                Document(page_content=doc.text, metadata=doc.metadata)
                for doc in documents
            ]
        )
        self.add_to_chroma(chunks)
        end_time = datetime.now()
        print(f"⏰ Time taken to load repo in Chroma: {end_time - start_time}")

    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def get_embedding_function(self):
        embeddings = OllamaEmbeddings(model=self.model_name)
        return embeddings

    def add_to_chroma(self, chunks: list[Document]):
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.get_embedding_function(),
        )

        chunks_with_ids = self.calculate_chunk_ids(chunks)

        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"📄 Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = [
            chunk
            for chunk in chunks_with_ids
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            print("✅ New documents added")
        else:
            print("✅ No new documents to add")

    def calculate_chunk_ids(self, chunks):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("file_path")
            page = chunk.metadata.get("page", 0)
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

        return chunks

    def answer_query(self, query_text: str):
        embedding_function = self.get_embedding_function()
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=embedding_function
        )

        results = db.similarity_search_with_score(query_text, k=5)

        num_chunks = min(10, len(results))  # Start with the top 10 chunks
        context_texts = [doc.page_content for doc, _score in results[:num_chunks]]

        context_text = "\n\n---\n\n".join(context_texts)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model=self.model_name)
        response_text = model.invoke(prompt)

        if "I don't know" in response_text or len(response_text.strip()) < 10:
            print("🤔 Initial response not confident, refining context...")
            num_chunks += 5  # Increase context size and re-query
            context_texts = [doc.page_content for doc, _score in results[:num_chunks]]
            context_text = "\n\n---\n\n".join(context_texts)

            prompt = prompt_template.format(context=context_text, question=query_text)
            response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results[:num_chunks]]
        formatted_response = {"responses": response_text, "sources": sources}
        return formatted_response
