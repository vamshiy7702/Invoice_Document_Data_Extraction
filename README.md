## 🧾 InvoiceIQ — AI Invoice Extraction with QR Validation

InvoiceIQ is a GenAI-powered application that automates the extraction and validation of GST invoice data from PDFs. It combines computer vision, QR-based verification, and multimodal LLM inference to build a robust document processing pipeline.

The system converts invoices into images, decodes embedded QR data, and extracts structured fields using a vision-capable LLM. A validation layer cross-checks critical identifiers like IRN and GSTIN against QR ground truth using fuzzy matching, improving reliability and reducing errors.

The application is built using Streamlit and provides a clean user interface with support for exporting extracted data in JSON and CSV formats. This project demonstrates practical implementation of LLM-based document intelligence systems in real-world scenarios.
