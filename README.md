# Document Digitizer with Summarization

This project is a web application that digitizes images of documents by extracting text and providing a concise summary. Built with **FastAPI** for the backend and **React** for the frontend, the app uses **OCR (Optical Character Recognition)** to convert images to text and **NLP** for summarizing the extracted content.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [License](#license)

---

## Features

- **Image to Text Conversion**: Extracts text from uploaded images.
- **Text Summarization**: Provides a concise summary of the extracted text.
- **Error Handling**: User-friendly error messages for unsupported files or empty inputs.
- **Responsive Design**: Optimized for desktop and mobile devices.

---

## Tech Stack

- **Frontend**: React, Tailwind CSS
- **Backend**: FastAPI, Python, Pytesseract, Hugging Face Transformers (Summarization Model)
- **Deployment**: Netlify (Frontend), Heroku (Backend)

---

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/document-digitizer.git
   cd document-digitizer