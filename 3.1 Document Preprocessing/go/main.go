// ============================================================================
//  DOCUMENT PREPROCESSING PIPELINE  (Go)
// ============================================================================
//
//  GOAL
//  ----
//  In a RAG (Retrieval-Augmented Generation) system, everything begins with
//  RAW TEXT. But your knowledge lives in many formats: PDFs, Word docs,
//  Markdown notes, plain text...
//
//  This program does ONE job: scan the ./docs folder and, for EVERY file it
//  finds, extract the plain text regardless of its format.
//
//  THE MENTAL MODEL
//  ----------------
//      docs/ folder -> discover files -> choose the right extractor
//                                        (based on file extension)
//                   -> extract text   -> collect results
//
//  Each FORMAT is stored very differently on disk:
//    - .txt / .md  -> already plain text, just read the bytes
//    - .pdf        -> binary; a library walks its internal structure
//    - .docx       -> secretly a ZIP archive of XML files; we unzip it
//                     ourselves using Go's standard library (educational!)
//  We hide those differences behind one idea: every extractor is a function
//  that takes a file path and returns a string. That uniformity is the trick.
// ============================================================================

package main

import (
	"archive/zip"       // to open .docx files, which are really ZIP archives
	"encoding/xml"      // to parse the XML inside a .docx
	"fmt"               // formatted printing
	"io"                // reading streams of bytes
	"os"                // file system access
	"path/filepath"     // OS-independent path handling
	"strings"           // string helpers (Join, ToLower, Fields...)

	"github.com/ledongthuc/pdf" // third-party: extracts text from PDF files
)

// ExtractedDocument is a small record describing one processed file.
// In Go we define a struct to give our data named, typed fields.
type ExtractedDocument struct {
	FileName  string // e.g. "report.pdf"
	FileType  string // the extension, e.g. ".pdf"
	Text      string // the extracted plain text
	CharCount int    // quick sanity check on how much we got
}

// docsDir is the folder we scan, relative to where you run the program.
const docsDir = "docs"

// ===========================================================================
//  THE EXTRACTORS — one function per format.
//  They all share the SAME signature: (path string) (string, error).
//  Returning an error is the idiomatic Go way to say "this might fail".
// ===========================================================================

// extractPlainText handles .md and .txt: they're ALREADY text on disk, so
// "extraction" is just reading the file's bytes and converting to a string.
func extractPlainText(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// extractPDF handles .pdf (binary). The pdf library opens the file, then we
// loop over its pages and append each page's plain text.
func extractPDF(path string) (string, error) {
	// pdf.Open returns an *os.File (which we must close) and a Reader.
	f, reader, err := pdf.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close() // `defer` runs this when the function returns — guaranteed cleanup.

	var builder strings.Builder // efficient way to assemble a big string piece by piece.
	totalPages := reader.NumPage()
	for pageIndex := 1; pageIndex <= totalPages; pageIndex++ {
		page := reader.Page(pageIndex)
		if page.V.IsNull() {
			continue // skip empty/invalid pages
		}
		// GetPlainText pulls the readable text out of the page's content.
		content, err := page.GetPlainText(nil)
		if err != nil {
			return "", err
		}
		builder.WriteString(content)
	}
	return builder.String(), nil
}

// extractDOCX handles .docx. A .docx is a ZIP archive; the actual text lives
// in a file inside it called "word/document.xml". We:
//   1) open the ZIP,
//   2) find word/document.xml,
//   3) parse the XML and pull out every <w:t> ("text") element.
// Doing this by hand (instead of a library) shows you what a .docx REALLY is.
func extractDOCX(path string) (string, error) {
	// Open the .docx as a ZIP archive.
	archive, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer archive.Close()

	// Find the document body inside the archive.
	var docXML io.ReadCloser
	for _, file := range archive.File {
		if file.Name == "word/document.xml" {
			docXML, err = file.Open()
			if err != nil {
				return "", err
			}
			break
		}
	}
	if docXML == nil {
		return "", fmt.Errorf("word/document.xml not found inside %s", path)
	}
	defer docXML.Close()

	// We stream through the XML token by token. Every run of visible text in
	// Word is wrapped in a <w:t> element, so we collect the characters that
	// appear inside those elements.
	decoder := xml.NewDecoder(docXML)
	var builder strings.Builder
	insideTextElement := false

	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break // reached the end of the XML
		}
		if err != nil {
			return "", err
		}

		switch element := token.(type) {
		case xml.StartElement:
			// "t" is the local name of Word's <w:t> text element.
			if element.Name.Local == "t" {
				insideTextElement = true
			}
		case xml.EndElement:
			if element.Name.Local == "t" {
				insideTextElement = false
			}
		case xml.CharData:
			// CharData is the raw text between tags. Keep it only when we're
			// currently inside a <w:t> element.
			if insideTextElement {
				builder.Write(element)
			}
		}
	}
	return builder.String(), nil
}

// ===========================================================================
//  THE DISPATCHER — "given a file, which extractor do I use?"
//  We map each extension to its extractor function. In Go, functions are
//  values, so we can store them in a map. This is a "dispatch table":
//  adding a new format later = add ONE entry here.
// ===========================================================================
var extractors = map[string]func(string) (string, error){
	".pdf":  extractPDF,
	".docx": extractDOCX,
	".md":   extractPlainText,
	".txt":  extractPlainText,
}

// extractText looks at the extension and runs the matching extractor.
func extractText(path string) (string, error) {
	// filepath.Ext("a/b/report.PDF") -> ".PDF"; ToLower normalizes the case
	// so ".PDF"/".Pdf"/".pdf" all route to the same extractor.
	ext := strings.ToLower(filepath.Ext(path))

	extractor, ok := extractors[ext]
	if !ok {
		// Return an error loudly instead of silently producing "" — you want
		// to KNOW when a file was skipped for being an unsupported type.
		return "", fmt.Errorf("unsupported file type: %q", ext)
	}
	return extractor(path)
}

// ===========================================================================
//  THE PIPELINE — tie it all together.
// ===========================================================================
func processAllDocuments() ([]ExtractedDocument, error) {
	// 1) DISCOVER: list everything inside the docs folder.
	entries, err := os.ReadDir(docsDir)
	if err != nil {
		return nil, err
	}

	var results []ExtractedDocument

	// 2) LOOP over each entry.
	for _, entry := range entries {
		name := entry.Name()

		// Skip sub-directories and hidden files (like macOS's .DS_Store).
		if entry.IsDir() || strings.HasPrefix(name, ".") {
			continue
		}

		fullPath := filepath.Join(docsDir, name)

		// 3) EXTRACT. If ONE file fails we log it and keep going, so a single
		//    bad file never crashes the whole run. Real folders are messy;
		//    a production pipeline degrades gracefully.
		text, err := extractText(fullPath)
		if err != nil {
			fmt.Printf("⚠️  Skipped %s: %v\n", name, err)
			continue
		}

		results = append(results, ExtractedDocument{
			FileName:  name,
			FileType:  strings.ToLower(filepath.Ext(name)),
			Text:      text,
			CharCount: len(text),
		})
		fmt.Printf("✅ %s — extracted %d characters\n", name, len(text))
	}

	return results, nil
}

func main() {
	fmt.Printf("📂 Scanning: %s\n\n", docsDir)

	docs, err := processAllDocuments()
	if err != nil {
		fmt.Printf("Pipeline failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n📊 Done. Processed %d document(s).\n", len(docs))

	// Print a short preview so you can SEE the text really came out.
	for _, doc := range docs {
		// strings.Fields splits on any whitespace; Join with single spaces
		// collapses newlines/tabs into a clean one-line preview.
		oneLine := strings.Join(strings.Fields(doc.Text), " ")
		preview := oneLine
		if len(preview) > 120 {
			preview = preview[:120] + " ..."
		}
		fmt.Printf("\n--- %s (%d chars) ---\n%s\n", doc.FileName, doc.CharCount, preview)
	}

	// NEXT STEPS in a full RAG system (not done here):
	//   text -> CHUNK into small pieces -> EMBED each chunk -> STORE in a
	//   vector database. This program stops at clean text: the foundation.
}
