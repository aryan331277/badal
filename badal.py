import os
import yaml
import logging
import asyncio
from google.colab import files
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load college info
try:
    with open('college_info.yaml', 'r') as f:
        college_info = yaml.safe_load(f)
    print("Found college_info.yaml!")
except FileNotFoundError:
    print("Please upload your college_info.yaml file")
    uploaded = files.upload()
    with open('college_info.yaml', 'r') as f:
        college_info = yaml.safe_load(f)
    print("Successfully loaded college info!")

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
HF_TOKEN = os.environ["HF_TOKEN"]

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=0.7,
    max_new_tokens=256
)

def setup_vector_db():
    # Create persistent storage directory
    persist_dir = "chroma_db"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    texts = []
    metadatas = []

    # Process the college info into chunks
    for category, details in college_info.items():
        for facility, info in details.items():
            text = f"Category: {category}\nFacility: {facility}\nInformation: {info}"
            texts.append(text)
            metadatas.append({"category": category, "facility": facility})

    # Create or load the vector store with persistent storage
    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    # Persist the database to disk
    db.persist()
    logger.info("Created/loaded vector database with persistent storage")
    return db

# Initialize the vector DB with error handling
try:
    vector_db = setup_vector_db()
except Exception as e:
    logger.error(f"Failed to initialize vector DB: {e}")
    # Try to recover by deleting and recreating the DB
    if os.path.exists("chroma_db"):
        import shutil
        shutil.rmtree("chroma_db")
    vector_db = setup_vector_db()

def generate_response(question, context_texts):
    # Common college-related questions pattern
    college_keywords = [
        "library", "mess", "cafeteria", "laundry", "hostel", "wifi",
        "medical", "sports", "gym", "lab", "faculty", "professor",
        "timing", "time", "schedule", "location", "where is", "how to",
        "college", "campus", "facility", "facilities"
    ]

    # Handle greetings
    greetings = ["hello", "hi", "hey", "greetings"]
    if any(greet in question.lower() for greet in greetings):
        return "👋 Hello! I'm your dedicated college assistant. Ask me about:\n\n" + \
               "\n".join(["• " + kw.capitalize() for kw in college_keywords[:10]]) + \
               "\n\nOr type /help for more options."

    # Handle thanks/goodbye
    if any(word in question.lower() for word in ["thank", "thanks", "bye", "goodbye"]):
        return "You're welcome! Always happy to help with college matters."

    # Check if question is college-related
    if not any(keyword in question.lower() for keyword in college_keywords):
        return "I specialize in college-related queries. Please ask about:\n" + \
               "\n".join(["• " + kw for kw in college_keywords[:8]]) + \
               "\n\nFor other questions, please contact college administration."

    if not context_texts:
        return "I couldn't find specific information about that. Try asking:\n" + \
               "• 'What are the library timings?'\n" + \
               "• 'How does the laundry system work?'\n" + \
               "• 'Where is the medical center located?'"

    try:
        # Process and prioritize information
        info_by_facility = {}
        for doc in context_texts:
            parts = doc.page_content.split('\n')
            if len(parts) >= 3:
                facility = parts[1].replace('Facility: ', '').replace('_', ' ').title()
                info = parts[2].replace('Information: ', '')
                if facility not in info_by_facility:
                    info_by_facility[facility] = info

        # Generate comprehensive response
        context_str = "\n\n".join(
            f"**{facility}**\n{info}"
            for facility, info in info_by_facility.items()
        )

        prompt = f"""As a knowledgeable college assistant, provide a detailed answer to the student's question using ONLY the following information:

{context_str}

Question: {question}

Answer in 2-3 clear sentences, being as helpful as possible. If unsure about any detail, suggest contacting the relevant department:"""

        response = llm.invoke(prompt).strip()

        # Validate response quality
        if not response or len(response.split()) < 8:
            raise ValueError("Insufficient response")

        return response

    except Exception as e:
        logger.error(f"Response error: {e}")
        # Return well-formatted complete information
        return "Here's the complete information I have:\n\n" + \
               "\n\n".join(
                   f"**{facility}**\n{info}"
                   for facility, info in info_by_facility.items()
               ) + \
               "\n\nFor more details, please visit the concerned facility."

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """📚 *College Assistant Help* 📚

I can provide information about:
- Library services and timings
- Mess/cafeteria menus and schedules
- Hostel rules and facilities
- Laundry system operation
- WiFi access and troubleshooting
- Medical center services
- Sports facilities availability
- Lab schedules and resources

*Sample Questions:*
• What are today's mess timings?
• How do I access the WiFi?
• Where is the medical center?
• What sports facilities are available?
• What are the library working hours?

For specific department contacts, ask about:
- Academic office
- Administration
- Finance department
- Student welfare"""
    await update.message.reply_text(help_text, parse_mode='Markdown')
# Colab-compatible main function
def run_bot():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running! Press Ctrl+C to stop.")
    application.run_polling()

# Special Colab handling
try:
    import nest_asyncio
    nest_asyncio.apply()
    run_bot()
except ImportError:
    print("Running in non-Colab environment")
    run_bot()
except Exception as e:
    print(f"Error: {e}")
    print("Bot stopped. Please check your tokens and try again.")
