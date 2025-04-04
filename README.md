PPA Explainer - AI Chatbot 🤖

Overview
The PPA Explainer is an AI-powered chatbot designed to provide insights into Power Purchase Agreements (PPAs). Using OpenAI's GPT model, ChromaDB, and Streamlit, this project enables users to explore and understand PPA-related topics efficiently.

Features
Interactive Chatbot: Engages in a conversational manner to explain PPAs.
Data-Driven Responses: Uses a 503-page dataset (4 PDF files) with 2881 processed text chunks.
Fact-Based & Transparent: Summarizes key points and provides source references.
Streamlit Interface: A user-friendly web-based UI for interaction.

Technology Stack
Python (Core logic)
Streamlit (Frontend interface)
OpenAI GPT-4 (Language model)
ChromaDB (Vector database for retrieving relevant PPA content)
LangChain (Text processing & embedding)

How It Works
Data Preparation
Extract text from 4 PPA PDFs.
Split text into overlapping chunks while preserving paragraph context.
Store vector embeddings in ChromaDB.
User Interaction
Users ask questions through the Streamlit app.
The chatbot retrieves relevant sections from ChromaDB.
OpenAI's GPT-4 generates a structured response.

Response Format

Brief Overview

Key Points (bullet format)

Source References
