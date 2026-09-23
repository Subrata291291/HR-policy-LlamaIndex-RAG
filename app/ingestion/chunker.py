from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


CHUNK_SIZE = 200
CHUNK_OVERLAP = 40


def create_nodes(documents: list[Document]):
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    return nodes