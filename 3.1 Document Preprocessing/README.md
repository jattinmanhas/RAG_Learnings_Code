# 3.1 Document Preprocessing

The first stage of a RAG pipeline: turn messy, multi-format files into clean
plain **text**. You drop files (`.pdf`, `.docx`, `.md`, `.txt`) into a `docs/`
folder and the program extracts the text from each one.

The exact same pipeline is implemented three ways so you can compare languages:

| Folder    | Language       | PDF lib            | DOCX approach                          |
| --------- | -------------- | ------------------ | -------------------------------------- |
| `nodejs/` | TypeScript     | `unpdf` (pdf.js)   | `mammoth`                              |
| `python/` | Python         | `pypdf`            | `python-docx`                          |
| `go/`     | Go             | `ledongthuc/pdf`   | unzipped by hand with the std library |

Each `docs/` folder is pre-loaded with one sample of every format
(`.pdf`, `.docx`, `.md`, `.txt`) so you can run immediately.

All three follow the identical shape:

```
docs/ -> discover files -> dispatch by extension -> extract text -> collect
```

The key idea: every file format gets its own **extractor**, but they all share
the same signature (`path -> text`). A small **dispatch table** routes each file
to the right one by its extension. Adding a new format = add one extractor + one
table entry.

## Running each one

**Node.js**
```bash
cd nodejs
npm install
npm start
```

**Python**
```bash
cd python
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python main.py
```

**Go**
```bash
cd go
go run .
```

Put some test files in the respective `docs/` folder first, then run. Each
program prints how many characters it extracted per file plus a short preview.

## What's next in a real RAG pipeline

This stage stops at clean text. After this you would:

1. **Chunk** the text into small overlapping pieces.
2. **Embed** each chunk into a vector (see `../2.1 Embeddings`).
3. **Store** the vectors in a vector DB (see `../2.2 Vector Search`).
