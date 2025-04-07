# Code Review AI - RAG Backend

A FastAPI backend service that provides Retrieval-Augmented Generation (RAG) capabilities for code repositories. This service uses LlamaIndex for document processing, ChromaDB for vector storage, and OpenAI's GPT-4 for generating responses.

## Features

- 🚀 FastAPI backend with async support
- 📚 RAG implementation using LlamaIndex
- 💾 Local vector storage with ChromaDB
- 🔍 Code-aware indexing and querying
- 🤖 OpenAI GPT-4 integration
- 📁 Support for multiple code file types
- 🗂️ Intelligent directory filtering (ignores node_modules, etc.)

## Prerequisites

- Python 3.9+
- OpenAI API key
- Git (for version control)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/code-review-app-backend.git
   cd code-review-app-backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### 1. Prepare Your Code for Indexing

Place the code you want to index in the `data/` directory. The system supports various file types including:
- Python (.py)
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- Java (.java)
- And many more (see `scripts/index_data.py` for full list)

### 2. Index Your Code

Run the indexing script:
```bash
python scripts/index_data.py
```

This will:
- Process all supported files in the data directory
- Create embeddings using OpenAI's API
- Store the vectors in ChromaDB
- Save the index in the `indexes/` directory

### 3. Start the Server

Run the FastAPI server:
```bash
python run.py
```

The server will start at http://localhost:8001

### 4. Using the API

#### Query Endpoint

POST `/query`
```bash
curl -X POST "http://localhost:8001/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "How does the authentication system work?"}'
```

Response format:
```json
{
  "answer": "Generated response about the authentication system...",
  "sources": [
    {
      "filename": "auth.py",
      "filepath": "/path/to/auth.py",
      "file_size": 1234,
      "created_at": 1712345678.0,
      "modified_at": 1712345678.0
    }
  ]
}
```

## Project Structure

```
.
├── data/               # Directory for source files to be indexed
├── indexes/            # Directory for persisted indexes
├── scripts/
│   └── index_data.py  # Script to preprocess and index files
├── src/
│   ├── main.py        # FastAPI application
│   └── utils.py       # Utility functions
├── .env               # Environment variables
├── .env.example       # Example environment variables
├── requirements.txt   # Python dependencies
└── run.py            # Server startup script
```

## Development

### VS Code Configuration

The project includes VS Code settings for optimal development experience:
- Python linting and formatting
- Git integration
- Custom theme and editor settings

### Adding New File Types

To add support for new file types:
1. Update the `CODE_EXTENSIONS` list in `scripts/index_data.py`
2. Re-run the indexing script

### Customizing the RAG System

You can modify various aspects of the system:
- Chunk size and overlap in `scripts/index_data.py`
- Vector store settings in the ChromaDB configuration
- Query parameters in the FastAPI endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LlamaIndex](https://github.com/jerryjliu/llama_index) for the RAG implementation
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenAI](https://openai.com/) for the language model 