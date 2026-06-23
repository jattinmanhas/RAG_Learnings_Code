/**
 * ============================================================================
 *  DOCUMENT PREPROCESSING PIPELINE  (Node.js / TypeScript)
 * ============================================================================
 *
 *  GOAL
 *  ----
 *  In a RAG (Retrieval-Augmented Generation) system, before you can embed or
 *  search anything, you first need RAW TEXT. But knowledge lives in many file
 *  formats: PDFs, Word documents, Markdown notes, plain text...
 *
 *  This pipeline does ONE job well:
 *      "Look inside the ./docs folder, and for every file found, pull out its
 *       plain text — no matter what format it is."
 *
 *  THE MENTAL MODEL
 *  ----------------
 *  Think of it as a factory line:
 *
 *      docs/ folder  ─▶  [discover files]  ─▶  [pick the right extractor
 *                                                based on file extension]
 *                    ─▶  [extract text]    ─▶  [collect results]
 *
 *  Each file TYPE needs a different "extractor", because a PDF is binary, a
 *  .docx is actually a ZIP archive of XML files, and a .md is just text.
 *  We hide those differences behind a common shape: every extractor takes a
 *  file path and returns a string of text. That uniformity is the whole trick.
 * ============================================================================
 */

// Node's built-in modules — no install needed.
// `fs/promises` gives us async file-system functions that return Promises,
// so we can use async/await instead of callbacks.
import { promises as fs } from "fs";
// `path` helps us build/parse file paths in an OS-independent way
// (so it works the same on macOS, Linux, and Windows).
import path from "path";

// Third-party libraries (installed via `npm install`). Each one is a
// specialist that knows how to decode one tricky binary format:
//   unpdf   -> reads PDF files and returns their text (a maintained wrapper
//              around Mozilla's pdf.js engine)
//   mammoth -> reads .docx (Microsoft Word) files and returns text
// We DON'T need a library for .md / .txt because those are already plain text.
import { extractText as extractPdfText, getDocumentProxy } from "unpdf";
import mammoth from "mammoth";

// ----------------------------------------------------------------------------
// A small "type" describing what one processed document looks like.
// This is just a labeled bag of data so the rest of the code is readable.
// ----------------------------------------------------------------------------
interface ExtractedDocument {
  fileName: string; // e.g. "report.pdf"
  fileType: string; // the extension, e.g. ".pdf"
  text: string; // the extracted plain text
  charCount: number; // how many characters we got (a quick sanity check)
}

// The folder we will scan. `path.join` joins pieces with the correct slash.
// __dirname is "the folder this file lives in"; ".." steps up out of /src
// into the project root, then into /docs.
const DOCS_DIR = path.join(__dirname, "..", "docs");

// ============================================================================
//  THE EXTRACTORS — one function per file format.
//  Each has the SAME signature: (filePath) => Promise<string>.
//  That sameness is what lets us treat them interchangeably later.
// ============================================================================

/**
 * PDF — Portable Document Format.
 * PDFs are binary, so we read the file as RAW BYTES (a Buffer), then hand
 * those bytes to pdf-parse, which walks the PDF's internal structure and
 * stitches the text back together.
 */
async function extractPdf(filePath: string): Promise<string> {
  // fs.readFile WITHOUT an encoding gives us a Buffer = raw bytes.
  const dataBuffer = await fs.readFile(filePath);
  // unpdf wants a Uint8Array view of those bytes. We hand them to pdf.js,
  // which parses the PDF's internal structure...
  const pdf = await getDocumentProxy(new Uint8Array(dataBuffer));
  // ...then pull the text out. `mergePages: true` concatenates every page
  // into one string instead of an array of per-page strings.
  const { text } = await extractPdfText(pdf, { mergePages: true });
  return text;
}

/**
 * DOCX — Microsoft Word.
 * Fun fact: a .docx file is secretly a ZIP archive containing XML files.
 * mammoth unzips it, finds the document body, and converts it to clean text,
 * throwing away Word's formatting noise.
 */
async function extractDocx(filePath: string): Promise<string> {
  // mammoth can take a path directly. `extractRawText` skips styling and
  // just returns the words.
  const result = await mammoth.extractRawText({ path: filePath });
  return result.value;
}

/**
 * Markdown (.md) and plain text (.txt).
 * These are ALREADY text on disk, so "extraction" is just reading the file
 * as a UTF-8 string. We keep the Markdown symbols (#, *, -) as-is; for RAG
 * they're usually harmless and can even help preserve document structure.
 */
async function extractPlainText(filePath: string): Promise<string> {
  // Passing "utf-8" tells readFile to decode bytes into a JS string for us.
  return fs.readFile(filePath, "utf-8");
}

// ============================================================================
//  THE DISPATCHER — "given a file, which extractor should I use?"
//
//  This is the heart of a multi-format pipeline. We look at the file
//  EXTENSION and route to the matching extractor. Adding a new format later
//  = add one `case`. (This pattern is called a "dispatch table" / "router".)
// ============================================================================
async function extractText(filePath: string): Promise<string> {
  // path.extname("a/b/report.PDF") -> ".PDF". We lowercase it so ".PDF",
  // ".Pdf", and ".pdf" all match the same branch.
  const ext = path.extname(filePath).toLowerCase();

  switch (ext) {
    case ".pdf":
      return extractPdf(filePath);
    case ".docx":
      return extractDocx(filePath);
    case ".md":
    case ".txt":
      return extractPlainText(filePath);
    default:
      // We THROW instead of silently returning "" so that unsupported files
      // are loud and visible — you want to KNOW a file was skipped.
      throw new Error(`Unsupported file type: "${ext}"`);
  }
}

// ============================================================================
//  THE PIPELINE — tie it all together.
// ============================================================================
async function processAllDocuments(): Promise<ExtractedDocument[]> {
  // 1) DISCOVER: list every entry inside the docs folder.
  const entries = await fs.readdir(DOCS_DIR);

  // We'll collect successful results here.
  const results: ExtractedDocument[] = [];

  // 2) LOOP over each file name.
  for (const fileName of entries) {
    const filePath = path.join(DOCS_DIR, fileName);

    // Skip sub-directories and hidden files (like .DS_Store on macOS).
    const stat = await fs.stat(filePath);
    if (stat.isDirectory() || fileName.startsWith(".")) continue;

    // 3) EXTRACT — wrapped in try/catch so ONE bad file doesn't crash the
    //    whole run. This is the difference between a toy and a real pipeline:
    //    real folders contain surprises, and you must degrade gracefully.
    try {
      const text = await extractText(filePath);
      results.push({
        fileName,
        fileType: path.extname(fileName).toLowerCase(),
        text,
        charCount: text.length,
      });
      console.log(`✅ ${fileName} — extracted ${text.length} characters`);
    } catch (err) {
      // Report the problem but keep going to the next file.
      console.warn(`⚠️  Skipped ${fileName}: ${(err as Error).message}`);
    }
  }

  return results;
}

// ----------------------------------------------------------------------------
//  ENTRY POINT
//  In a full RAG system, the next steps after this would be:
//      text -> CHUNK into small pieces -> EMBED each chunk -> STORE in a
//      vector database. This file stops at clean text, which is the
//      foundation everything else builds on.
// ----------------------------------------------------------------------------
async function main() {
  console.log(`📂 Scanning: ${DOCS_DIR}\n`);
  const docs = await processAllDocuments();

  console.log(`\n📊 Done. Processed ${docs.length} document(s).`);

  // Print a tiny preview of each so you can SEE the text was extracted.
  for (const doc of docs) {
    const preview = doc.text.trim().slice(0, 120).replace(/\s+/g, " ");
    console.log(`\n--- ${doc.fileName} (${doc.charCount} chars) ---`);
    console.log(preview + (doc.text.length > 120 ? " ..." : ""));
  }
}

// Kick everything off and make sure any unexpected error is printed clearly.
main().catch((err) => {
  console.error("Pipeline failed:", err);
  process.exit(1);
});
