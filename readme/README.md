# 📚 UPSC MCQ PDF/Image Processor

A production-ready Streamlit application that uses Google's Gemini AI to extract, parse, and classify multiple-choice questions from PDF or image files according to strict UPSC (Union Public Service Commission) topic hierarchy.

## 🌟 Features

### Core Functionality

- **LLM-Powered Extraction**: Uses Gemini AI for text extraction and parsing (no traditional OCR)
- **Intelligent Question Parsing**: Automatically extracts questions, options (A-D), answers, and explanations
- **UPSC Hierarchy Classification**: Classifies questions into predefined topics, subtopics, and sub-subtopics
- **Large File Support**: Handles PDFs with 200+ pages efficiently
- **Batch Processing**: Intelligent batching for optimal performance
- **Multi-Format Support**: Accepts PDF, PNG, JPG, and JPEG files

### Output Features

- **Structured JSON Output**: Clean, validated JSON with complete question metadata
- **Excel Export**: Download results as formatted Excel spreadsheet
- **Data Validation**: Pydantic-based validation ensures data quality
- **Preview Dashboard**: Visual analytics and preview of processed questions
- **Error Tracking**: Detailed error reporting for validation issues

## 🎯 Question Structure

Each processed question follows this structure:

```json
{
  "id": 1,
  "question": "Who was the real founder of Turk rule in Bihar?",
  "options": {
    "A": "Qutb-ud-din Aibak",
    "B": "Bakhtiyar Khilji",
    "C": "Iltutmish",
    "D": "Balban"
  },
  "answer": "B",
  "topic": "History",
  "subtopic1": "Medieval India",
  "subtopic2": "Arab & Turkish Invasions",
  "difficulty": "Medium",
  "explanation": "Bakhtiyar Khilji, a general of Qutb-ud-din Aibak...",
  "citations": [],
  "tags": []
}
```

## 📋 UPSC Topic Hierarchy

The application strictly follows the UPSC syllabus hierarchy:

### 1. History

- **Ancient India**: Sources, Prehistoric Cultures, Indus Valley, Vedic Age, etc.
- **Medieval India**: Early Medieval Kingdoms, Delhi Sultanate, Mughal Empire, etc.
- **Modern India**: British Rule, Freedom Struggle, Constitutional Developments, etc.

### 2. Geography

- **Physical Geography**: Earth & Solar System, Landforms, Climate, etc.
- **Indian Geography**: Physical Features, Drainage, Climate, Agriculture, etc.
- **World Geography**: Climate Zones, Rivers, Natural Disasters, etc.

### 3. Polity

- **Indian Constitution**: Preamble, Fundamental Rights, Directive Principles, etc.
- **Central Government**: President, PM, Parliament, Supreme Court, etc.
- **State Government**: Governor, CM, State Legislature, etc.
- **Judiciary**: Supreme Court, High Courts, Judicial Review, etc.
- **Local Governance**: Panchayati Raj, Municipalities, etc.

### 4. Economy

- **Indian Economy**: Economic Reforms, Fiscal Policy, Monetary Policy, etc.
- **Economic Planning**: Planning Commission, Five Year Plans, Budget, etc.
- **Agriculture & Industry**: Policies, Marketing, Irrigation, etc.
- **Banking & Finance**: RBI, Commercial Banks, Stock Exchange, etc.
- **Government Schemes**: MGNREGA, PMGSY, Ayushman Bharat, etc.

### 5. Environment & Ecology

- **Ecology & Ecosystems**: Basics, Food Chains, Energy Flow, etc.
- **Biodiversity**: Hotspots, Conservation, Invasive Species, etc.
- **Environmental Pollution**: Air, Water, Soil, Plastic, etc.
- **Climate Change**: Greenhouse Effect, Indicators, Agreements, etc.

### 6. Science & Technology

- **General Science**: Physics, Chemistry, Biology, Astronomy, etc.
- **Space Technology**: ISRO, Satellite Tech, Space Missions, etc.
- **Defence Technology**: Missiles, Nuclear Tech, Combat Aircraft, etc.
- **IT & Computers**: AI, Cybersecurity, Digital India, Blockchain, etc.
- **Biotechnology**: Genetic Engineering, Vaccines, CRISPR, etc.

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Google Gemini API keys
- pip (Python package manager)

### Step 1: Clone or Download

```bash
cd /Users/shaneel/Projects/Daira/Streamlit-Upsc-pdf
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

**Note**: The application has multiple API keys hardcoded for rotation to handle rate limits. For production, move these to environment variables.

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload File

- Click "Browse files" or drag and drop
- Supported formats: PDF, PNG, JPG, JPEG
- Maximum recommended size: 50MB per file

### 2. Configure Processing (Optional)

Use the sidebar to adjust:

- **Max pages**: Limit pages for testing (0 = process all)
- **Extraction batch size**: Pages to process at once (default: 5)
- **Classification batch size**: Questions to classify at once (default: 20)

### 3. Start Processing

- Click "🚀 Start Processing"
- Monitor progress through status messages
- Processing steps:
  1. Extract content from file
  2. Extract text with Gemini AI
  3. Parse questions
  4. Classify with UPSC hierarchy
  5. Validate questions

### 4. Review Results

- View preview of first 5 questions
- Check topic and difficulty distributions
- Review validation errors (if any)

### 5. Download Results

- **JSON Format**: For further processing or API integration
- **Excel Format**: For spreadsheet analysis or manual review

## 🏗️ Architecture

### Processing Pipeline

```
PDF/Image Input
    ↓
[PyMuPDF] Extract pages as images
    ↓
[Gemini Vision API] Extract text from images
    ↓
[Gemini LLM] Parse questions from text
    ↓
[Gemini LLM] Classify questions (topic hierarchy)
    ↓
[Pydantic] Validate data structure
    ↓
Excel/JSON Output
```

### Key Components

1. **File Processing Layer**

   - PyMuPDF: PDF to image conversion
   - PIL: Image handling
   - Gemini Vision: Text extraction

2. **AI Processing Layer**

   - Gemini LLM: Question extraction
   - Gemini LLM: Classification and enrichment
   - Custom prompts: Ensure strict hierarchy compliance

3. **Validation Layer**

   - Pydantic models: Schema validation
   - Custom validators: Answer and difficulty validation
   - Hierarchy validator: Topic/subtopic verification

4. **Output Layer**
   - Pandas: Data manipulation
   - OpenPyXL: Excel generation
   - JSON serialization

## ⚙️ Configuration

### Rate Limiting

The application implements intelligent rate limiting:

- Multiple API keys with automatic rotation
- Multiple model configurations (Flash, Pro)
- Configurable batch sizes
- Automatic retry with backoff

### Model Configuration

```python
MODELS = [
    {"name": "gemini-2.0-flash-exp", "rpm": 15, "rpd": 1500},
    {"name": "gemini-1.5-flash", "rpm": 15, "rpd": 1500},
    {"name": "gemini-1.5-pro", "rpm": 2, "rpd": 50},
]
```

### Batch Sizes

- **Text Extraction**: 5 pages per batch (adjustable)
- **Question Parsing**: 10 text chunks per batch
- **Classification**: 20 questions per batch (adjustable)

## 🎨 UI Features

### Main Interface

- Clean, intuitive file upload
- Real-time processing status
- Progress bars for long operations
- Success/error notifications

### Sidebar Controls

- Processing configuration
- Real-time statistics
- Session metrics

### Results Dashboard

- Question preview table
- Topic distribution chart
- Difficulty distribution chart
- Subtopic distribution chart
- Download buttons (JSON & Excel)
- Validation error viewer

## 🐛 Error Handling

### Automatic Recovery

- API key rotation on rate limit errors
- Retry logic for transient failures
- Graceful degradation (continues on partial failures)

### Error Reporting

- Validation errors tracked separately
- Detailed error messages
- Questions with errors available for review

## 📊 Performance

### Optimization Strategies

1. **Batch Processing**: Reduces API calls
2. **Parallel Processing**: Where applicable
3. **Smart Caching**: Session state management
4. **Rate Limiting**: Prevents API throttling

### Expected Performance

- Small files (1-10 pages): 1-3 minutes
- Medium files (11-50 pages): 5-15 minutes
- Large files (51-200 pages): 15-60 minutes

_Performance varies based on API response times and rate limits_

## 🚢 Deployment

### Streamlit Cloud

1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Add API keys to Secrets
4. Deploy

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:

```bash
docker build -t upsc-mcq-processor .
docker run -p 8501:8501 upsc-mcq-processor
```

### Environment Variables for Production

```env
GOOGLE_API_KEY_1=your_key_1
GOOGLE_API_KEY_2=your_key_2
GOOGLE_API_KEY_3=your_key_3
# ... add more keys as needed
```

## 🔒 Security Considerations

### Production Checklist

- [ ] Move API keys to environment variables
- [ ] Add authentication layer
- [ ] Implement file size limits
- [ ] Add virus scanning for uploads
- [ ] Enable HTTPS
- [ ] Set up logging and monitoring
- [ ] Add rate limiting per user
- [ ] Implement file cleanup after processing

## 🧪 Testing

### Test with Sample Data

Create a simple PDF with MCQs:

```
1. What is the capital of India?
A) Mumbai
B) New Delhi
C) Kolkata
D) Chennai
Answer: B

2. Who was the first Prime Minister of India?
A) Mahatma Gandhi
B) Jawaharlal Nehru
C) Sardar Patel
D) Rajendra Prasad
Answer: B
```

### Validation Checks

- Questions have valid IDs
- All options A-D are present
- Answer is one of A, B, C, D
- Topic hierarchy is valid
- Difficulty is Easy/Medium/Hard
- No empty required fields

## 📝 API Usage

The processed questions can be integrated into your application:

```python
import json

# Load processed questions
with open('upsc_mcq_20251003_120000.json', 'r') as f:
    questions = json.load(f)

# Filter by topic
history_questions = [q for q in questions if q['topic'] == 'History']

# Filter by difficulty
hard_questions = [q for q in questions if q['difficulty'] == 'Hard']

# Get specific subtopic
medieval_questions = [
    q for q in questions
    if q['subtopic1'] == 'Medieval India'
]
```

## 🤝 Contributing

### Areas for Contribution

- Additional validation rules
- More output formats (CSV, database)
- Improved error handling
- Performance optimizations
- UI/UX enhancements
- Additional language support

## 📄 License

This project is intended for educational purposes for UPSC preparation.

## 👥 Support

For issues or questions:

- Check the error messages in the application
- Review this documentation
- Check validation errors in the error viewer

## 🔄 Version History

### Version 1.0.0 (October 2025)

- Initial release
- PDF and image support
- Gemini-powered extraction
- UPSC hierarchy classification
- Excel and JSON export
- Validation and error tracking

## 📚 Additional Resources

- [UPSC Official Website](https://upsc.gov.in/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---
