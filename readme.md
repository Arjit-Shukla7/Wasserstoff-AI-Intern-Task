# Document Research & Theme Identification Chatbot

An AI-powered chatbot that analyzes large document collections to identify common themes and provide detailed, cited responses to user queries. Built for the Wasserstoff AI internship task.

## 🚀 Features

- **Multi-format Document Support**: PDF, TXT, DOC, DOCX, and scanned images
- **OCR Processing**: Automatic text extraction from scanned documents
- **Semantic Search**: Vector-based document search using embeddings
- **Theme Identification**: Automatic identification of common themes across documents
- **Detailed Citations**: Precise citations with page and paragraph references
- **Interactive Web Interface**: User-friendly chat interface
- **RESTful API**: Complete API for integration with other systems

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Vector DB     │
│   (HTML/JS)     │◄──►│   Backend       │◄──►│   (Qdrant)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AI Services   │
                       │ (OpenAI/Groq)   │
                       └─────────────────┘
```

## 📁 Project Structure

```
chatbot_theme_identifier/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── documents.py      # Document upload/management endpoints
│   │   │   └── query.py          # Query processing endpoints
│   │   ├── core/
│   │   │   ├── config.py         # Configuration settings
│   │   │   └── database.py       # Database initialization
│   │   ├── models/
│   │   │   ├── models.py         # SQLAlchemy models
│   │   │   └── schemas.py        # Pydantic schemas
│   │   ├── services/
│   │   │   ├── ai_service.py     # AI/LLM integration
│   │   │   ├── document_service.py # Document processing
│   │   │   └── vector_service.py # Vector operations
│   │   ├── main.py               # FastAPI application
│   │   └── config.py             # App configuration
│   ├── data/                     # Data storage
│   ├── static/                   # Static files
│   ├── Dockerfile               # Docker configuration
│   └── requirements.txt         # Python dependencies
├── docs/                        # Documentation
├── tests/                       # Test files
├── demo/                        # Demo materials
├── docker-compose.yml           # Docker Compose configuration
└── README.md                    # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (recommended)
- API keys for AI services (OpenAI, Groq, or Google)

### Method 1: Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot_theme_identifier
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Method 2: Local Development

1. **Install system dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils
   
   # macOS
   brew install tesseract poppler
   ```

2. **Set up Python environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start Qdrant (Vector Database)**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant:latest
   ```

4. **Set environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GROQ_API_KEY="your-groq-key"  # Optional
   export GOOGLE_API_KEY="your-google-key"  # Optional
   ```

5. **Run the application**
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `GROQ_API_KEY` | Groq API key | No | - |
| `GOOGLE_API_KEY` | Google API key | No | - |
| `DATABASE_URL` | Database connection string | No | `sqlite:///./data/chatbot.db` |
| `QDRANT_HOST` | Qdrant host | No | `localhost` |
| `QDRANT_PORT` | Qdrant port | No | `6333` |

### API Configuration

The application supports multiple AI providers:
- **OpenAI GPT-4/GPT-3.5**: Primary recommendation
- **Groq (Llama)**: Free alternative with good performance
- **Google Gemini**: Alternative option

## 📚 Usage

### 1. Document Upload

Upload documents through the web interface or API:

```python
import requests

files = [('files', open('document1.pdf', 'rb')), 
         ('files', open('document2.pdf', 'rb'))]
response = requests.post('http://localhost:8000/api/documents/upload', files=files)
```

### 2. Query Documents

Ask questions about your document collection:

```python
query_data = {"query": "What are the main compliance issues mentioned?"}
response = requests.post('http://localhost:8000/api/query/research', json=query_data)
```

### 3. Expected Response Format

The system returns structured responses with:

```json
{
  "success": true,
  "individual_responses": [
    {
      "document_id": "DOC001",
      "answer": "The order states that the fine was imposed under section 15 of the SEBI Act.",
      "citation": "Page 4, Para 2"
    }
  ],
  "themes": [
    {
      "title": "Regulatory Non-Compliance",
      "description": "Documents highlight regulatory non-compliance with SEBI Act and LODR.",
      "documents": ["DOC001", "DOC002"]
    }
  ]
}
```

## 🔍 API Endpoints

### Documents

- `POST /api/documents/upload` - Upload multiple documents
- `GET /api/documents/list` - List all uploaded documents
- `DELETE /api/documents/{doc_id}` - Delete a specific document
- `GET /api/documents/{doc_id}` - Get document details

### Query

- `POST /api/query/research` - Perform theme-based research query
- `POST /api/query/simple` - Simple document search
- `GET /api/query/history` - Get query history

### System

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## 🧪 Testing

### Run Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Coverage report
python -m pytest --cov=app tests/
```

### Test Document Sets

For testing, you can use various document types:
- Legal documents (court orders, contracts)
- Technical reports
- Business documents
- Medical research papers
- Policy documents

## 🚀 Deployment

### Free Hosting Options

1. **Render**
   ```bash
   # Deploy directly from GitHub
   # Set environment variables in Render dashboard
   ```

2. **Railway**
   ```bash
   railway login
   railway init
   railway add # Add Qdrant service
   railway deploy
   ```

3. **Hugging Face Spaces**
   ```bash
   # Create Space with Docker
   # Upload code and set secrets
   ```

### Production Considerations

- Use PostgreSQL instead of SQLite for production
- Implement proper authentication
- Set up monitoring and logging
- Configure CORS properly
- Use environment-specific configurations
- Implement rate limiting

## 🔧 Troubleshooting

### Common Issues

1. **OCR Not Working**
   ```bash
   # Install Tesseract
   sudo apt-get install tesseract-ocr tesseract-ocr-eng
   ```

2. **Vector Database Connection**
   ```bash
   # Check Qdrant is running
   curl http://localhost:6333/health
   ```

3. **API Key Issues**
   ```bash
   # Verify environment variables
   echo $OPENAI_API_KEY
   ```

4. **Memory Issues with Large Documents**
   - Increase Docker memory limits
   - Process documents in batches
   - Use document chunking

### Performance Optimization

- Use async processing for large document sets
- Implement caching for frequently accessed documents
- Optimize vector database indices
- Use document preprocessing pipelines

## 📊 Evaluation Metrics

The system is evaluated on:

### Functionality (40%)
- Document upload and processing
- Theme identification accuracy
- Citation precision
- Query response quality

### Code Quality (30%)
- Modular design
- Error handling
- Documentation
- Version control usage

### System Design (20%)
- Scalability considerations
- Database design
- API structure
- Deployment readiness

### User Experience (10%)
- Interface usability
- Response time
- Error messages
- Documentation clarity

## 📋 Technical Details

### Document Processing Pipeline

1. **Upload** → File validation and storage
2. **OCR** → Text extraction from scanned documents
3. **Chunking** → Split documents into processable segments
4. **Embedding** → Generate vector embeddings
5. **Storage** → Store in vector database with metadata

### Theme Identification Algorithm

1. **Query Processing** → Extract relevant chunks per document
2. **Response Generation** → Generate individual document responses
3. **Theme Analysis** → Identify common patterns across responses
4. **Synthesis** → Create final themed response with citations

### Vector Search Strategy

- **Embedding Model**: OpenAI text-embedding-ada-002
- **Similarity Metric**: Cosine similarity
- **Chunk Size**: 1000 tokens with 200 token overlap
- **Top-K Retrieval**: 5 most relevant chunks per document

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request
