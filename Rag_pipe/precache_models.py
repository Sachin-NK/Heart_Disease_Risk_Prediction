from embedding import EmbeddingModel
from model import QwenLLM
from loguru import logger

def main():
    logger.info("Pre-caching models for offline use...")
    
    # Cache embedding model
    try:
        logger.info("Caching BGE-M3 embedding model...")
        EmbeddingModel(offline=False)
        logger.success("Embedding model cached successfully")
    except Exception as e:
        logger.error(f"Error caching embedding model: {e}")
    
    # Cache LLM
    try:
        logger.info("Caching Qwen-0.5B LLM...")
        QwenLLM(offline=False)
        logger.success("LLM cached successfully")
    except Exception as e:
        logger.error(f"Error caching LLM: {e}")
    
    logger.success("Model caching complete. You can now use --offline mode")

if __name__ == "__main__":
    main()