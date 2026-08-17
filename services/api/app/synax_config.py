# services/api/app/synax_config.py
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from neo4j import GraphDatabase
from openai import AsyncOpenAI
from google import genai
from supermemory import AsyncSupermemory

load_dotenv()

# Dataset and Output Directories
DATA_DIR = "./synax_dataset"
ENTITY_OUTPUT_DIR = os.path.join(DATA_DIR, "entities")
COREF_LINKED_ENTITY_OUTPUT_DIR = os.path.join(DATA_DIR, "coref_linked_entities")
RELATIONSHIP_OUTPUT_DIR = os.path.join(DATA_DIR, "relationships")
KNOWLEDGE_GRAPH_MANIFEST_DIR = os.path.join(DATA_DIR, "knowledge_graph_manifests")
EMBEDDING_OUTPUT_DIR = os.path.join(DATA_DIR, "embeddings")

# Configuration Constants
WIKI_LANG = ["en"]
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_DIM = 1024
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBEDDING_SIMILARITY_THRESHOLD = 0.85
NUM_FAISS_SHARDS = 12
FAISS_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=16)
FAISS_SHARD_DIR = os.path.join(DATA_DIR, "faiss_shards")
UPDATE_INTERVAL_MINUTES = int(os.getenv("UPDATE_INTERVAL_MINUTES", "120"))
INGESTION_ENABLED_KEY = "synax:ingestion:enabled"
REDIS_URL = "redis://localhost:6379/0"
WIKIDATA_CACHE_PATH = os.path.join(DATA_DIR, "wikidata_labels.json")
BATCH_WRITE_SIZE = 4000

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    raise RuntimeError(
        "NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD are required."
    )

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required.")

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required.")

# Supermemory Configuration
SUPERMEMORY_API_KEY = os.getenv("SUPERMEMORY_API_KEY")

if not SUPERMEMORY_API_KEY:
    raise RuntimeError("SUPERMEMORY_API_KEY is required.")

SUPERMEMORY_TIMEOUT = 30
SUPERMEMORY_MAX_RETRIES = 3

# Paystack Configuration
PAYSTACK_BASE_URL = "https://api.paystack.co"
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")

if not PAYSTACK_SECRET_KEY:
    raise RuntimeError("PAYSTACK_SECRET_KEY is required.")

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Initialization
embedder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))
embedder.eval()

reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
reranker = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL).to(device)
reranker.eval()

# Database & API Clients
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
)
gpt_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
supermemory_client = AsyncSupermemory(
    api_key=SUPERMEMORY_API_KEY,
    timeout=SUPERMEMORY_TIMEOUT,
    max_retries=SUPERMEMORY_MAX_RETRIES,
)