import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

load_dotenv()
my_api_key = os.getenv("OPENAI_API_KEY")

if not my_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Put it in .env or export it.")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=my_api_key)

# Initialize ChromaDB Vector Store
vector_store = Chroma(
    collection_name="CS401",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # Optional: persist to disk
)

def read_python_file(filepath: str) -> str:
    """Read content from a Python file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def list_function_files(folder_path: str = "Functions") -> List[str]:
    """List all Python files in the Functions folder"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found!")
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
    return files

def initialize_database(folder_path: str = "Functions", clear_existing: bool = False):
    """
    Initialize ChromaDB with all Python functions from the Functions folder
    
    Args:
        folder_path: Path to the folder containing Python function files
        clear_existing: If True, clear existing data before adding new
    """
    global vector_store
    
    # Clear existing data if requested
    if clear_existing:
        print("Clearing existing ChromaDB collection...")
        try:
            vector_store.delete_collection()
            print("✓ Existing collection cleared")
        except Exception as e:
            print(f"Note: Could not clear collection (may not exist): {e}")
        
        # Reinitialize after clearing
        vector_store = Chroma(
            collection_name="CS401",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
    
    # Get all function files
    function_files = list_function_files(folder_path)
    
    if not function_files:
        print(f"No Python files found in '{folder_path}/'")
        return
    
    print(f"\nFound {len(function_files)} Python files in '{folder_path}/'")
    print("=" * 60)
    
    # Prepare documents
    documents = []
    
    for i, filename in enumerate(function_files, 1):
        filepath = os.path.join(folder_path, filename)
        
        try:
            # Read file content
            code_content = read_python_file(filepath)
            
            # Create document with metadata
            doc = Document(
                page_content=code_content,
                metadata={
                    "source": filename,
                    "filepath": filepath,
                    "type": "original_function",
                    "index": i
                }
            )
            
            documents.append(doc)
            print(f"{i}. {filename} ({len(code_content)} chars)")
            
        except Exception as e:
            print(f"✗ Error reading {filename}: {e}")
    
    # Add documents to ChromaDB
    print("\n" + "=" * 60)
    print(f"Adding {len(documents)} documents to ChromaDB...")
    
    try:
        vector_store.add_documents(documents)
        print(f"✓ Successfully added {len(documents)} functions to the database!")
        
    except Exception as e:
        print(f"✗ Error adding documents to ChromaDB: {e}")
        return
    
    # Verify the data was added
    print("\n" + "=" * 60)
    print("Verifying database contents...")
    
    try:
        # Try a simple retrieval
        test_results = vector_store.similarity_search("loop iteration", k=3)
        print(f"✓ Database verification successful!")
        print(f"  Total documents in DB: {len(documents)}")
        print(f"  Test query returned: {len(test_results)} results")
        
        if test_results:
            print(f"\n  Sample result:")
            print(f"    Source: {test_results[0].metadata.get('source', 'Unknown')}")
            print(f"    Content preview: {test_results[0].page_content[:100]}...")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")

def check_database_status():
    """Check current status of the database"""
    print("\nDatabase Status Check")
    print("=" * 60)
    
    try:
        # Get collection info
        collection = vector_store._collection
        count = collection.count()
        
        print(f"Collection Name: CS401")
        print(f"Total Documents: {count}")
        
        if count > 0:
            # Get a sample
            sample = vector_store.similarity_search("function", k=1)
            if sample:
                print(f"\nSample Document:")
                print(f"  Source: {sample[0].metadata.get('source', 'Unknown')}")
                print(f"  Content: {sample[0].page_content[:150]}...")
        else:
            print("\n⚠ Database is empty! Run initialization first.")
            
    except Exception as e:
        print(f"Error checking database: {e}")
        print("\n⚠ Database may not be initialized yet.")

def add_single_function(filepath: str, metadata: dict = None):
    """
    Add a single Python function file to the database
    
    Args:
        filepath: Path to the Python file
        metadata: Optional metadata dictionary
    """
    try:
        code_content = read_python_file(filepath)
        
        if metadata is None:
            metadata = {
                "source": os.path.basename(filepath),
                "filepath": filepath,
                "type": "original_function"
            }
        
        doc = Document(page_content=code_content, metadata=metadata)
        vector_store.add_documents([doc])
        
        print(f"✓ Added {os.path.basename(filepath)} to database")
        
    except Exception as e:
        print(f"✗ Error adding {filepath}: {e}")

def search_similar_functions(query: str, k: int = 5):
    """
    Search for similar functions in the database
    
    Args:
        query: Query string (can be code snippet or description)
        k: Number of results to return
    """
    print(f"\nSearching for: '{query}'")
    print("=" * 60)
    
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        
        if not results:
            print("No results found")
            return
        
        for i, (doc, score) in enumerate(results, 1):
            similarity = 1 - score  # Convert distance to similarity
            print(f"\n{i}. Similarity: {similarity:.3f}")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Type: {doc.metadata.get('type', 'Unknown')}")
            print(f"   Preview: {doc.page_content[:100]}...")
            
    except Exception as e:
        print(f"Error searching: {e}")

if __name__ == "__main__":
    import sys
    
    print("ChromaDB Initialization Tool")
    print("=" * 60)
    
    # Check if database exists and has data
    check_database_status()
    
    # Ask user what to do
    print("\nOptions:")
    print("1. Initialize database (add all functions)")
    print("2. Re-initialize database (clear and add all functions)")
    print("3. Test search")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        folder = input("Enter folder path (default: Functions): ").strip() or "Functions"
        initialize_database(folder_path=folder, clear_existing=False)
        
    elif choice == "2":
        folder = input("Enter folder path (default: Functions): ").strip() or "Functions"
        confirm = input("⚠ This will DELETE all existing data. Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            initialize_database(folder_path=folder, clear_existing=True)
        else:
            print("Cancelled.")
            
    elif choice == "3":
        query = input("Enter search query: ").strip()
        k = int(input("Number of results (default: 5): ").strip() or "5")
        search_similar_functions(query, k=k)
        
    elif choice == "4":
        print("Exiting...")
        
    else:
        print("Invalid choice!")
    
    print("\n" + "=" * 60)
    print("Done!")