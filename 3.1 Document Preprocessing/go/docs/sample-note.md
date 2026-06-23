# RAG Sample Note

Retrieval-Augmented Generation (RAG) combines a **retriever** with a
**generator**. The retriever finds relevant documents; the generator (an LLM)
writes an answer grounded in them.

- Step 1: preprocess documents into clean text
- Step 2: chunk the text
- Step 3: embed each chunk
- Step 4: store vectors and search them
