import asyncio
from rag import RAGPipeline
import argparse
import time
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="GRASS GIS RAG Chatbot")
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument("--async", action="store_true", dest="async_mode", help="Use async query")
    args = parser.parse_args()

    rag = RAGPipeline(offline=args.offline)
    
    if args.async_mode:
        logger.info("Running async query...")
        start = time.time()
        response = asyncio.run(rag.aquery(args.question))
        latency = time.time() - start
        logger.info(f"Async total latency: {latency:.3f}s")
    else:
        logger.info("Running sync query...")
        response = rag.query(args.question)
    
    answer, results, metrics = response
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    main()