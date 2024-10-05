from typing import List

from langchain_core.documents import Document

def create_langchain_documents(doc_contents: List[str]) -> List[Document]:
    """Create a list of Document objects from a list of strings.

    Args:
    doc_contents (List[str]): List of strings, where each string is the content of a document.

    Returns:
    List[Document]: List of Document instances.
    """
    documents = []
    for content in doc_contents:
        # Creating a Document instance for each string in the list
        document = Document(page_content=content)
        documents.append(document)
    return documents
