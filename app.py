import streamlit as st
import json
import time
import re
import io
import base64
import os
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# Core libraries
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ValidationError
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# PDF/Image processing
import fitz  # PyMuPDF
from PIL import Image

# Load environment variables
load_dotenv()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class QuestionInput(BaseModel):
    """Input model for MCQ questions"""
    id: Optional[int] = None
    question: str
    options: Dict[str, str] = Field(default_factory=dict)
    answer: str
    topic: str = ""
    subtopic1: str = ""
    subtopic2: str = ""
    difficulty: str = ""
    explanation: str = ""
    citations: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True


class QuestionOutput(BaseModel):
    """Output model for processed MCQ questions"""
    id: int
    question: str
    options: Dict[str, str]
    answer: str
    topic: str
    subtopic1: str
    subtopic2: str
    difficulty: str
    explanation: str
    citations: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v):
        if v not in ["A", "B", "C", "D"]:
            raise ValueError("Answer must be A, B, C, or D")
        return v
    
    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v):
        if v not in ["Easy", "Medium", "Hard"]:
            raise ValueError("Difficulty must be Easy, Medium, or Hard")
        return v

# =============================================================================
# UPSC HIERARCHY
# =============================================================================

ALLOWED_HIERARCHY = {
    "History": {
        "Ancient India": [
            "Sources of Ancient Indian History", "Prehistoric Cultures",
            "Indus Valley Civilization", "Vedic Age",
            "Mahajanapadas & Rise of Magadha", "Religious Movements",
            "Mauryan Empire", "Post-Mauryan Period", "Gupta Empire",
            "Harsha and Early Medieval Kingdoms", "Science, Technology, Art & Culture"
        ],
        "Medieval India": [
            "Early Medieval Kingdoms", "Arab & Turkish Invasions",
            "Delhi Sultanate", "Bhakti & Sufi Movements",
            "Vijayanagar & Bahmani Kingdoms", "Mughal Empire",
            "Mughal Art & Architecture", "Rise of Marathas",
            "Medieval Economy and Society"
        ],
        "Modern India": [
            "Advent of Europeans", "Establishment of British Rule in India",
            "British Administrative Policies", "Economic Impact of British Rule",
            "Socio-Religious Reform Movements", "Revolt of 1857",
            "Growth of Nationalism", "Extremist Phase (1905–1917)",
            "Gandhian Era (1917–1947)", "Revolutionary Movements",
            "Leftist & Peasant Movements", "Indian National Army & Subhas Bose",
            "Constitutional Developments", "Towards Independence & Partition",
            "Princely States & Integration (Post-1947)",
            "British Legacy in Indian Administration"
        ]
    },
    "Geography": {
        "Physical Geography": [
            "Earth and Solar System", "Landforms", "Climatic Regions",
            "Biodiversity", "Water Resources", "Atmospheric Circulation",
            "Soils", "Natural Resources", "Disasters"
        ],
        "Indian Geography": [
            "Physical Features of India", "Drainage Systems", "Climate of India",
            "Agriculture", "Mineral Resources", "Forest and Wildlife",
            "Population Distribution"
        ],
        "World Geography": [
            "World Climate Zones", "Rivers and Lakes", "Population Distribution",
            "Political Divisions", "Natural Disasters", "Environmental Issues"
        ]
    },
    "Polity": {
        "Indian Constitution": [
            "Preamble", "Fundamental Rights", "Directive Principles",
            "Amendment Process", "Federal Structure"
        ],
        "Central Government": [
            "President", "Prime Minister", "Council of Ministers",
            "Parliament", "Supreme Court"
        ],
        "State Government": [
            "Governor", "Chief Minister", "State Legislature", "State Judiciary"
        ],
        "Judiciary": [
            "Supreme Court", "High Courts", "Judicial Review", "Judicial Activism"
        ],
        "Local Governance": [
            "Panchayati Raj", "Municipalities", "Urban & Rural Local Bodies", "Finances"
        ],
        "Constitutional & Statutory Bodies": [
            "Election Commission", "UPSC", "CAG",
            "National Human Rights Commission", "Central Vigilance Commission"
        ],
        "Governance & Policies": [
            "E-Governance", "Public Grievances", "RTI", "Civil Services"
        ]
    },
    "Economy": {
        "Indian Economy": [
            "Economic Reforms", "Fiscal Policy", "Monetary Policy",
            "Growth & Development", "Indian Agriculture"
        ],
        "Economic Planning & Budgeting": [
            "Planning Commission", "Five Year Plans", "Union Budget",
            "Economic Reforms", "Public Finance"
        ],
        "Agriculture & Industry": [
            "Agriculture Policies", "Industrial Policies",
            "Agricultural Marketing", "Irrigation"
        ],
        "Banking & Financial Systems": [
            "Reserve Bank of India", "Commercial Banks",
            "Financial Markets", "Stock Exchange"
        ],
        "Government Schemes": [
            "Mahatma Gandhi National Rural Employment Guarantee Act",
            "PMGSY", "Ayushman Bharat", "Jan Dhan Yojana"
        ],
        "Social Development": [
            "Healthcare", "Education Policies", "Gender Equality",
            "Social Welfare Schemes"
        ]
    },
    "Environment & Ecology": {
        "Ecology & Ecosystems": [
            "Ecology Basics", "Food Chains", "Energy Flow",
            "Biomes", "Ecological Succession"
        ],
        "Biodiversity": [
            "Hotspots", "Invasive Species", "Conservation Strategies",
            "Biodiversity Loss"
        ],
        "Environmental Pollution": [
            "Air Pollution", "Water Pollution", "Soil Pollution", "Plastic Pollution"
        ],
        "Climate Change": [
            "Greenhouse Effect", "Climate Change Indicators",
            "Impact of Climate Change", "International Climate Agreements"
        ],
        "Conservation Efforts": [
            "Protected Areas", "Wildlife Protection", "Afforestation",
            "Wildlife Corridors"
        ],
        "Environmental Laws & Treaties": [
            "Environmental Protection Act", "Forest Conservation Act",
            "CITES", "Paris Agreement"
        ]
    },
    "Science & Technology": {
        "General Science": [
            "Physics", "Chemistry", "Biology", "Astronomy", "Ecology"
        ],
        "Space Technology": [
            "ISRO Programs", "Satellite Technology", "Space Missions",
            "Space Agencies Worldwide"
        ],
        "Defence Technology": [
            "Missiles", "Nuclear Technology", "Combat Aircraft", "Naval Technology"
        ],
        "IT & Computers": [
            "Artificial Intelligence", "Cybersecurity", "Digital India", "Blockchain"
        ],
        "Biotechnology & Health": [
            "Genetic Engineering", "Vaccines", "Stem Cells", "CRISPR", "Health Tech"
        ],
        "Emerging Technologies": [
            "Quantum Computing", "5G", "Nanotechnology", "Space Tourism"
        ]
    }
}

# =============================================================================
# GEMINI API CONFIGURATION
# =============================================================================

def get_secret(key: str) -> str:
    return st.secrets.get(key) or os.getenv(key)
# Multiple API keys for rate limiting
API_KEYS = [
    key for key in [
        get_secret("GOOGLE_API_KEY_1"),
        get_secret("GOOGLE_API_KEY_2"),
        get_secret("GOOGLE_API_KEY_3"),
        get_secret("GOOGLE_API_KEY_4"),
        get_secret("GOOGLE_API_KEY_5"),
        get_secret("GOOGLE_API_KEY_6"),
        get_secret("GOOGLE_API_KEY_7")
    ] if key
]

# Model configurations with rate limits
MODELS = [
    {"name": "gemini-2.0-flash-exp", "rpm": 15, "rpd": 1500},
    {"name": "gemini-1.5-flash", "rpm": 15, "rpd": 1500},
    {"name": "gemini-1.5-pro", "rpm": 2, "rpd": 50},
]

# Global state for API rotation
if 'current_api_key_index' not in st.session_state:
    st.session_state.current_api_key_index = 0
if 'current_model_index' not in st.session_state:
    st.session_state.current_model_index = 0
if 'request_timestamps' not in st.session_state:
    st.session_state.request_timestamps = []


def get_current_api_key() -> str:
    """Get the current API key"""
    return API_KEYS[st.session_state.current_api_key_index]


def rotate_api_key():
    """Rotate to next API key"""
    st.session_state.current_api_key_index = (
        st.session_state.current_api_key_index + 1
    ) % len(API_KEYS)


def get_llm() -> ChatGoogleGenerativeAI:
    """Get configured LLM instance with rate limiting"""
    model = MODELS[st.session_state.current_model_index]
    return ChatGoogleGenerativeAI(
        model=model["name"],
        temperature=0,
        max_retries=2,
        google_api_key=get_current_api_key()
    )


def configure_genai():
    """Configure Gemini API for direct use"""
    genai.configure(api_key=get_current_api_key())
    return genai.GenerativeModel(MODELS[st.session_state.current_model_index]["name"])


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def get_hierarchy_string() -> str:
    """Generate formatted hierarchy string for prompts"""
    hierarchy_lines = []
    for topic, subtopics in ALLOWED_HIERARCHY.items():
        hierarchy_lines.append(f"\nTopic: {topic}")
        for subtopic1, subtopic2_list in subtopics.items():
            hierarchy_lines.append(f"  Subtopic1: {subtopic1}")
            for subtopic2 in subtopic2_list:
                hierarchy_lines.append(f"    Subtopic2: {subtopic2}")
    return "\n".join(hierarchy_lines)


EXTRACTION_PROMPT = """You are an AI assistant extracting MCQ questions from a document for UPSC Civil Services Examination preparation.

**TASK**: Extract ALL multiple-choice questions from the provided content.

For each question, extract:
- Question text
- Options (A, B, C, D) - exactly as written
- Correct answer (if provided)
- Explanation (if provided)
- Tags (if provided)

**OUTPUT FORMAT**: Return ONLY a valid JSON array. No markdown, no code blocks, no extra text.

Example format:
[
  {
    "id": 1,
    "question": "Question text here?",
    "options": {
      "A": "Option A text",
      "B": "Option B text",
      "C": "Option C text",
      "D": "Option D text"
    },
    "answer": "B",
    "explanation": "Explanation if available",
    "tags": []
  }
]

**CRITICAL RULES**:
- Response MUST start with [ and end with ]
- NO markdown formatting, NO ```json blocks
- Double quotes for all keys and values
- If answer is not provided, use empty string ""
- If explanation is not provided, use empty string ""
- Extract ALL questions, even if information is incomplete
- Maintain exact question and option text
- Preserve any numbering or formatting in questions

Now extract all questions from the following content:
"""


CLASSIFICATION_PROMPT = """You are an AI assistant helping students prepare for the UPSC Civil Services Examination.
You will be given a list of multiple-choice questions in JSON format.
For each question, perform the following:
1. Provide a clear and concise explanation for the correct answer.
2. Classify the difficulty level as one of: Easy, Medium, or Hard.
3. Identify the most relevant topic, subtopic1, and subtopic2 STRICTLY using the hierarchy below:
- `subtopic1` MUST be chosen from the allowed Subtopic1 list of the selected topic.
- `subtopic2` MUST be chosen from the allowed Subtopic2 list under the chosen subtopic1.
- `subtopic1` and `subtopic2` CANNOT be identical.
- NEVER invent new subtopic names. Only use the provided ones.
If the provided answer is incorrect or missing, correct it and explain why.
If the answer is already a valid choice (A, B, C, D) and correct, do not change it.
Return ONLY the updated list of questions in valid JSON format with added fields: explanation, difficulty, topic, subtopic1, and subtopic2.

CRITICAL JSON RULES:
- Response MUST start with [ and end with ]
- No markdown, no code blocks
- Double quotes for all keys/values
- No null or empty answer field — always one of A, B, C, D
- Keep original `tags` unchanged
- Ensure subtopic2 is logically a child of subtopic1
- Do not use subtopic1 name as subtopic2
- You MUST always provide non-empty values for topic, subtopic1, and subtopic2.
- If you cannot find an exact match, choose the closest possible match from the hierarchy.
- If you return any empty value for topic, subtopic1, or subtopic2, your response will be considered INVALID.
- Do NOT skip or leave them blank under any circumstance.
- STRICTLY cross-verify for Economy and Agriculture topics.

AVAILABLE TOPICS AND SUBTOPICS:
{hierarchy}

Now process these questions:
"""

# =============================================================================
# PDF/IMAGE PROCESSING FUNCTIONS
# =============================================================================

def extract_images_from_pdf(pdf_path: str, max_pages: int = None) -> List[Image.Image]:
    """Extract images from PDF pages using PyMuPDF"""
    images = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            # Render page as image at high resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        doc.close()
        return images
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
        return []


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_text_with_gemini(images: List[Image.Image], batch_size: int = 5) -> str:
    """Extract text from images using Gemini Vision API"""
    extracted_text = []
    
    model = configure_genai()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        progress = (i + len(batch)) / len(images)
        progress_bar.progress(progress)
        status_text.text(f"Extracting text from pages {i+1} to {i+len(batch)}...")
        
        try:
            # Process each image in batch
            for img in batch:
                # Convert PIL Image to bytes for Gemini API
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                
                response = model.generate_content([
                    "Extract all text from this image exactly as it appears. Include questions, options, answers, and explanations.",
                    {"mime_type": "image/png", "data": img_bytes}
                ])
                extracted_text.append(response.text)
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            st.warning(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            rotate_api_key()
            time.sleep(2)
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return "\n\n".join(extracted_text)


def extract_questions_with_gemini(content: str, batch_size: int = 10) -> List[Dict]:
    """Extract structured questions from text using Gemini"""
    llm = get_llm()
    
    # Split content into chunks for processing
    content_chunks = content.split("\n\n")
    all_questions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(content_chunks), batch_size):
        batch = "\n\n".join(content_chunks[i:i + batch_size])
        progress = min((i + batch_size) / len(content_chunks), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Extracting questions... ({len(all_questions)} found)")
        
        try:
            prompt = EXTRACTION_PROMPT + "\n\n" + batch
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            
            # Parse JSON response
            questions = safe_json_parse(response.content)
            if questions:
                all_questions.extend(questions)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            st.warning(f"Error extracting questions from batch: {str(e)}")
            rotate_api_key()
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return all_questions


def classify_questions_with_gemini(questions: List[Dict], batch_size: int = 20) -> List[Dict]:
    """Classify questions with topic hierarchy using Gemini"""
    llm = get_llm()
    classified_questions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    hierarchy_str = get_hierarchy_string()
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        progress = (i + len(batch)) / len(questions)
        progress_bar.progress(progress)
        status_text.text(f"Classifying questions {i+1} to {i+len(batch)}...")
        
        try:
            # Add IDs to questions if not present
            for idx, q in enumerate(batch, start=i+1):
                if 'id' not in q or not q['id']:
                    q['id'] = idx
            
            prompt = CLASSIFICATION_PROMPT.format(hierarchy=hierarchy_str)
            prompt += "\n\n" + json.dumps(batch, indent=2)
            
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            
            # Parse JSON response
            classified_batch = safe_json_parse(response.content)
            if classified_batch:
                classified_questions.extend(classified_batch)
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            st.warning(f"Error classifying batch {i//batch_size + 1}: {str(e)}")
            rotate_api_key()
            # Add unclassified questions as-is
            classified_questions.extend(batch)
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return classified_questions


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_json_parse(response_text: str) -> List[Dict]:
    """Safely parse JSON from LLM response"""
    try:
        # Remove markdown code blocks if present
        response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'^```\s*$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()
        
        # Find JSON array
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx+1]
            return json.loads(json_str)
        
        return []
    except Exception as e:
        st.error(f"JSON parsing error: {str(e)}")
        return []


def validate_question(question: Dict) -> tuple[bool, str]:
    """Validate a single question against schema"""
    try:
        QuestionOutput(**question)
        return True, ""
    except ValidationError as e:
        return False, str(e)


def questions_to_excel(questions: List[Dict]) -> bytes:
    """Convert questions list to Excel bytes"""
    df = pd.DataFrame(questions)
    
    # Reorder columns
    column_order = [
        'id', 'question', 'options', 'answer', 'topic', 'subtopic1', 
        'subtopic2', 'difficulty', 'explanation', 'citations', 'tags'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Convert dict/list columns to strings for Excel
    if 'options' in df.columns:
        df['options'] = df['options'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    if 'citations' in df.columns:
        df['citations'] = df['citations'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='MCQ Questions')
        
        # Auto-adjust column widths
        worksheet = writer.sheets['MCQ Questions']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
    
    return output.getvalue()


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="UPSC MCQ Processor",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 UPSC MCQ PDF/Image Processor")
    st.markdown("""
    Upload a PDF or image file containing multiple-choice questions.  
    The app uses **Gemini AI** to extract, parse, and classify questions according to UPSC hierarchy.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Processing Options")
        max_pages = st.number_input(
            "Max pages to process (0 = all)",
            min_value=0,
            max_value=500,
            value=0,
            help="Limit the number of pages to process for testing"
        )
        
        extraction_batch_size = st.slider(
            "Text extraction batch size",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of pages to process at once"
        )
        
        classification_batch_size = st.slider(
            "Classification batch size",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of questions to classify at once"
        )
        
        st.divider()
        
        st.subheader("📊 Statistics")
        if 'stats' in st.session_state:
            stats = st.session_state.stats
            st.metric("Total Questions", stats.get('total', 0))
            st.metric("Successfully Processed", stats.get('processed', 0))
            st.metric("Validation Errors", stats.get('errors', 0))
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF or Image",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Upload a file containing MCQ questions"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Display file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📄 File size: {file_size:.2f} MB")
        
        # Process button
        if st.button("🚀 Start Processing", type="primary"):
            process_file(
                uploaded_file,
                max_pages if max_pages > 0 else None,
                extraction_batch_size,
                classification_batch_size
            )
    
    # Display results if available
    if 'processed_questions' in st.session_state:
        display_results()


def process_file(uploaded_file, max_pages, extraction_batch_size, classification_batch_size):
    """Main processing pipeline"""
    
    with st.spinner("Processing file..."):
        start_time = time.time()
        
        # Step 1: Extract images
        st.info("📥 Step 1: Extracting content from file...")
        
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if uploaded_file.type == "application/pdf":
            images = extract_images_from_pdf(temp_path, max_pages)
        else:
            images = [Image.open(temp_path)]
        
        if not images:
            st.error("❌ Failed to extract content from file")
            return
        
        st.success(f"✅ Extracted {len(images)} page(s)")
        
        # Step 2: Extract text using Gemini
        st.info("🔍 Step 2: Extracting text with Gemini AI...")
        extracted_text = extract_text_with_gemini(images, extraction_batch_size)
        
        if not extracted_text:
            st.error("❌ Failed to extract text")
            return
        
        st.success(f"✅ Extracted {len(extracted_text)} characters of text")
        
        # Step 3: Extract questions
        st.info("📝 Step 3: Parsing questions...")
        raw_questions = extract_questions_with_gemini(extracted_text, 10)
        
        if not raw_questions:
            st.error("❌ No questions found in the content")
            return
        
        st.success(f"✅ Found {len(raw_questions)} questions")
        
        # Step 4: Classify questions
        st.info("🏷️ Step 4: Classifying questions with UPSC hierarchy...")
        classified_questions = classify_questions_with_gemini(
            raw_questions,
            classification_batch_size
        )
        
        # Step 5: Validate and clean
        st.info("✔️ Step 5: Validating questions...")
        valid_questions = []
        errors = []
        
        for q in classified_questions:
            is_valid, error_msg = validate_question(q)
            if is_valid:
                valid_questions.append(q)
            else:
                errors.append({"question_id": q.get('id', 'unknown'), "error": error_msg})
        
        # Store results
        st.session_state.processed_questions = valid_questions
        st.session_state.errors = errors
        st.session_state.stats = {
            'total': len(classified_questions),
            'processed': len(valid_questions),
            'errors': len(errors)
        }
        
        processing_time = time.time() - start_time
        
        st.success(f"""
        ✅ **Processing Complete!**
        - Total questions: {len(classified_questions)}
        - Valid questions: {len(valid_questions)}
        - Validation errors: {len(errors)}
        - Processing time: {processing_time:.2f} seconds
        """)


def display_results():
    """Display processed results and download options"""
    
    st.header("📊 Results")
    
    questions = st.session_state.processed_questions
    
    # Preview
    st.subheader("👀 Preview (First 5 Questions)")
    preview_df = pd.DataFrame(questions[:5])
    st.dataframe(preview_df, use_container_width=True)
    
    # Topic distribution
    st.subheader("📈 Topic Distribution")
    topic_counts = pd.Series([q['topic'] for q in questions]).value_counts()
    st.bar_chart(topic_counts)
    
    # Difficulty distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Difficulty Distribution")
        difficulty_counts = pd.Series([q.get('difficulty', 'Unknown') for q in questions]).value_counts()
        st.bar_chart(difficulty_counts)
    
    with col2:
        st.subheader("🎯 Subtopic1 Distribution")
        subtopic1_counts = pd.Series([q.get('subtopic1', 'Unknown') for q in questions]).value_counts()
        st.bar_chart(subtopic1_counts.head(10))
    
    # Download options
    st.divider()
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download as JSON
        json_str = json.dumps(questions, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"upsc_mcq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Download as Excel
        excel_bytes = questions_to_excel(questions)
        st.download_button(
            label="📥 Download Excel",
            data=excel_bytes,
            file_name=f"upsc_mcq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Show errors if any
    if st.session_state.errors:
        with st.expander("⚠️ View Validation Errors"):
            st.json(st.session_state.errors)


if __name__ == "__main__":
    main()
