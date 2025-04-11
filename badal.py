import os
import yaml
import logging
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings # Use updated import pathfrom telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load college info - Expect the file to be in the same directory
try:
    with open('college_info.yaml', 'r') as f:
        college_info = yaml.safe_load(f)
    logger.info("Successfully loaded college_info.yaml")
# REMOVED: Colab-specific file upload fallback
except FileNotFoundError:
    logger.critical("FATAL ERROR: 'college_info.yaml' not found in the project directory.")
    logger.critical("Please ensure college_info.yaml is included in your deployment.")
    exit(1) # Exit if the essential config file is missing
except yaml.YAMLError as e:
    logger.critical(f"FATAL ERROR: Could not parse 'college_info.yaml': {e}")
    exit(1)
except Exception as e:
    logger.critical(f"FATAL ERROR: An unexpected error occurred loading 'college_info.yaml': {e}")
    exit(1)


# --- Environment Variable Loading ---
# Ensure these are set in your Render environment variables settings
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not TELEGRAM_TOKEN:
    logger.critical("FATAL ERROR: TELEGRAM_TOKEN environment variable not set.")
    exit(1)
if not HF_TOKEN:
    logger.critical("FATAL ERROR: HF_TOKEN environment variable not set.")
    exit(1)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# --- Langchain Setup ---
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        temperature=0.7,
        max_new_tokens=256
    )
except Exception as e:
    logger.critical(f"FATAL ERROR: Failed to initialize Langchain components: {e}")
    exit(1)

# --- Vector DB Setup ---
def setup_vector_db():
    persist_dir = "chroma_db_render" # Use a different name if needed, ensure permissions
    # Render provides ephemeral storage by default. For persistence across deploys/restarts,
    # you might need Render Disks (paid feature) or recreate the DB on each start.
    # This implementation recreates the DB on each start if persist_dir doesn't exist or load fails.

    if os.path.exists(persist_dir):
         logger.info(f"Attempting to load existing vector DB from {persist_dir}")
         try:
             # Try loading existing DB
             db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
             logger.info("Successfully loaded vector database.")
             return db
         except Exception as e:
             logger.warning(f"Could not load existing DB from {persist_dir}: {e}. Recreating...")
             # If loading fails, remove the potentially corrupt directory
             import shutil
             shutil.rmtree(persist_dir)

    # If directory doesn't exist or loading failed, create it
    logger.info(f"Creating new vector DB in {persist_dir}")
    if not os.path.exists(persist_dir):
        try:
            os.makedirs(persist_dir)
        except OSError as e:
            logger.error(f"Failed to create directory {persist_dir}: {e}")
            raise # Reraise the exception as this is critical

    texts = []
    metadatas = []

    for category, details in college_info.items():
        for facility, info in details.items():
            text = f"Category: {category}\nFacility: {facility}\nInformation: {info}"
            texts.append(text)
            metadatas.append({"category": category, "facility": facility})

    if not texts:
        logger.warning("No texts found in college_info.yaml to add to the vector store.")
        # Depending on your logic, you might want to exit or handle this case
        # For now, we allow creating an empty DB if needed.

    try:
        db = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        # Persist the database to disk
        # db.persist() # persist() might not be needed if using persist_directory on creation/load
        logger.info("Created and persisted vector database.")
        return db
    except Exception as e:
        logger.error(f"Failed to create vector database: {e}")
        raise # Reraise the exception

# Initialize the vector DB with error handling during startup
try:
    vector_db = setup_vector_db()
except Exception as e:
    logger.critical(f"FATAL ERROR: Could not initialize vector database during startup: {e}")
    exit(1)


# --- Response Generation Logic ---
def generate_response(question, context_texts):
    # Common college-related questions pattern
    college_keywords = [
        "library", "mess", "cafeteria", "laundry", "hostel", "wifi",
        "medical", "sports", "gym", "lab", "faculty", "professor",
        "timing", "time", "schedule", "location", "where is", "how to",
        "college", "campus", "facility", "facilities", "academic", "administration",
        "finance", "student welfare" # Added keywords from help
    ]

    # Handle greetings
    greetings = ["hello", "hi", "hey", "greetings", "/start"]
    if any(greet in question.lower() for greet in greetings):
        return "👋 Hello! I'm your dedicated college assistant. Ask me about various facilities or type /help for more options."

    # Handle thanks/goodbye
    if any(word in question.lower() for word in ["thank", "thanks", "bye", "goodbye"]):
        return "You're welcome! Always happy to help with college matters. 😊"

    # Check if question is college-related (basic check)
    if not any(keyword in question.lower() for keyword in college_keywords):
        return "I specialize in college-related queries (like library, mess, hostel, etc.). For other topics, please contact the college administration directly."

    if not context_texts:
        # If vector search returned nothing relevant
        logger.warning(f"No relevant context found for question: {question}")
        return ("I couldn't find specific information matching your question in my current knowledge base. "
                "Could you please rephrase or ask about a specific facility like 'library timings' or 'hostel rules'? "
                "You can also type /help.")

    try:
        # Process and prioritize information from context
        info_by_facility = {}
        for doc in context_texts:
            # Assuming doc is a Langchain Document object with page_content
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            parts = content.split('\n')
            if len(parts) >= 3:
                # Extract category, facility, and info robustly
                category = parts[0].replace('Category: ', '').strip()
                facility = parts[1].replace('Facility: ', '').replace('_', ' ').title().strip()
                info = parts[2].replace('Information: ', '').strip()
                # Use category+facility as key to avoid duplicates if different categories have same facility name
                key = f"{category} - {facility}"
                if key not in info_by_facility:
                    info_by_facility[key] = info

        if not info_by_facility:
             logger.warning(f"Context texts were found but couldn't be parsed into facility info for question: {question}")
             # Fallback if parsing failed but context exists
             context_str = "\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in context_texts])

        else:
            # Generate context string from parsed info
            context_str = "\n\n".join(
                # Reconstruct format similar to original text for LLM
                f"Facility: {key.split(' - ')[1]}\nInformation: {info}"
                for key, info in info_by_facility.items()
            )

        prompt = f"""You are a helpful college assistant bot. Answer the student's question accurately and concisely using ONLY the provided context information. Do not make up information. If the context doesn't contain the answer, say you don't have that specific detail.

Context Information:
---
{context_str}
---

Student's Question: {question}

Answer:"""

        response = llm.invoke(prompt).strip()

        # Basic validation
        if not response or response.lower() == "i don't have that specific detail.":
             logger.info(f"LLM indicated no specific answer found for: {question}")
             # Provide a more helpful fallback than just "I don't know"
             return ("I found some related information, but not the specific detail you asked for. "
                     "Perhaps try asking about a different aspect, or contact the relevant college department directly for precise information.")

        return response

    except Exception as e:
        logger.error(f"Error generating response for question '{question}': {e}")
        # Fallback response in case of LLM or processing error
        return ("I encountered an issue while processing your request. "
                "Please try asking in a different way, or contact administration if the problem persists.")

# --- Telegram Bot Handlers ---

# PLACEHOLDER: Implement your actual /start command logic
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    welcome_text = (
        f"👋 Hello {user.mention_html()}! I'm your college assistant bot.\n\n"
        "I can help with info about library, mess, hostels, WiFi, medical services, sports, labs, and more.\n\n"
        "Ask me a question like:\n"
        "  • 'What are the library hours?'\n"
        "  • 'Where is the cafeteria?'\n\n"
        "Type /help to see more commands and examples."
    )
    await update.message.reply_html(welcome_text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a help message when the /help command is issued."""
    help_text = """
📚 *College Assistant Help* 📚

I can provide information based on the `college_info.yaml` file about various facilities and services.

*How to Ask:*
Just type your question naturally! For example:
• What are today's mess timings?
• How do I access the campus WiFi?
• Where is the medical center located?
• What sports facilities are available?
• Tell me about the library services.
• Hostel rules for visitors?

*Available Commands:*
/start - Welcome message
/help - Shows this help message

*Topics I might know about:*
(Depends on your `college_info.yaml`)
- Library services & timings
- Mess/cafeteria menus & schedules
- Hostel rules & facilities
- Laundry system operation
- WiFi access & troubleshooting
- Medical center services & location
- Sports facilities availability
- Lab schedules & resources
- Academic office info
- Administration contacts
- Finance department info
- Student welfare details

If I can't answer, the information might not be in my knowledge base. Please contact the relevant department directly in such cases.
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')


# PLACEHOLDER: Implement your actual message handling logic
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles non-command text messages."""
    question = update.message.text
    logger.info(f"Received message: {question}")

    try:
        # Perform similarity search
        # Adjust 'k' based on how many relevant chunks you want
        search_results = await asyncio.to_thread(vector_db.similarity_search, question, k=3)

        if not search_results:
            logger.info(f"No similar documents found for: {question}")
            response = generate_response(question, []) # Let generate_response handle empty context
        else:
            logger.info(f"Found {len(search_results)} relevant documents for: {question}")
            # Pass the Document objects directly if generate_response handles them,
            # otherwise extract page_content: context_texts = [doc.page_content for doc in search_results]
            response = generate_response(question, search_results) # Pass Document objects

        await update.message.reply_text(response)

    except Exception as e:
        logger.error(f"Error handling message '{question}': {e}")
        await update.message.reply_text("Sorry, I encountered an error trying to process your message. Please try again later.")


# --- Main Bot Execution ---
def run_bot():
    """Sets up and runs the Telegram bot."""
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot handlers configured. Starting polling...")
    # Start the Bot
    application.run_polling()
    logger.info("Bot stopped.")

# Standard Python entry point
if __name__ == "__main__":
    try:
        run_bot()
    except Exception as e:
        logger.critical(f"Bot failed to run: {e}", exc_info=True) # Log traceback
        print(f"Critical Error: Bot failed to run. Check logs. Error: {e}")

# REMOVED: Colab-specific nest_asyncio handling
