"""
Script to view Qdrant vector database collections and data
Usage: python view_qdrant.py
"""
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def view_collections(client):
    """View all collections in Qdrant"""
    
    print("=" * 70)
    print("📚 QDRANT COLLECTIONS")
    print("=" * 70)
    
    try:
        collections = client.get_collections()
        
        if not collections.collections:
            print("(No collections found)")
            print("\n💡 To create a collection, run:")
            print("   python ingest_data.py")
            return
        
        print(f"\nFound {len(collections.collections)} collection(s):\n")
        
        for col in collections.collections:
            print(f"📦 {col.name}")
            
            # Get detailed info
            try:
                info = client.get_collection(col.name)
                print(f"   Vectors count:  {info.vectors_count:,}")
                print(f"   Points count:   {info.points_count:,}")
                print(f"   Status:         {info.status}")
                
                # Get config
                if hasattr(info.config, 'params'):
                    params = info.config.params
                    if hasattr(params, 'vectors'):
                        vector_config = params.vectors
                        if hasattr(vector_config, 'size'):
                            print(f"   Vector size:    {vector_config.size}")
                        if hasattr(vector_config, 'distance'):
                            print(f"   Distance:       {vector_config.distance}")
                
                print()
            except Exception as e:
                print(f"   Error getting details: {e}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def view_collection_details(client, collection_name):
    """View detailed information about a specific collection"""
    
    print("=" * 70)
    print(f"📊 COLLECTION DETAILS: {collection_name}")
    print("=" * 70)
    
    try:
        # Get collection info
        info = client.get_collection(collection_name)
        
        print(f"\n📈 Statistics:")
        print(f"   Vectors count:  {info.vectors_count:,}")
        print(f"   Points count:   {info.points_count:,}")
        print(f"   Status:         {info.status}")
        
        # Configuration
        print(f"\n⚙️  Configuration:")
        if hasattr(info.config, 'params'):
            params = info.config.params
            if hasattr(params, 'vectors'):
                vector_config = params.vectors
                if hasattr(vector_config, 'size'):
                    print(f"   Vector size:    {vector_config.size}")
                if hasattr(vector_config, 'distance'):
                    print(f"   Distance:       {vector_config.distance}")
        
        # Sample points
        print(f"\n📝 Sample Points (first 5):")
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        if not points:
            print("   (No points found)")
        else:
            for i, point in enumerate(points, 1):
                print(f"\n   [{i}] Point ID: {point.id}")
                if point.payload:
                    for key, value in point.payload.items():
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"       {key}: {value}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Collection '{collection_name}' may not exist")
        print("   Run 'python view_qdrant.py' to see all collections")


def search_similar(client, collection_name, query_text, k=5):
    """Search for similar vectors using text query"""
    
    print("=" * 70)
    print(f"🔍 SEARCH: {query_text}")
    print("=" * 70)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load embedding model
        print("\n⏳ Loading embedding model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Generate query vector
        print("⏳ Generating query embedding...")
        query_vector = model.encode(query_text).tolist()
        
        # Search
        print(f"⏳ Searching in collection '{collection_name}'...\n")
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )
        
        if not results:
            print("(No results found)")
        else:
            print(f"Found {len(results)} result(s):\n")
            
            for i, result in enumerate(results, 1):
                print(f"[{i}] Score: {result.score:.4f}")
                print(f"    Point ID: {result.id}")
                
                if result.payload:
                    for key, value in result.payload.items():
                        if isinstance(value, str):
                            if len(value) > 200:
                                value = value[:200] + "..."
                            print(f"    {key}: {value}")
                        else:
                            print(f"    {key}: {value}")
                
                print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


def delete_collection(client, collection_name):
    """Delete a collection"""
    
    print(f"⚠️  WARNING: This will delete collection '{collection_name}' and all its data!")
    confirm = input("   Continue? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Cancelled")
        return
    
    try:
        client.delete_collection(collection_name)
        print(f"✅ Deleted collection '{collection_name}'")
    except Exception as e:
        print(f"❌ Error: {e}")


def check_health(client):
    """Check Qdrant health"""
    
    print("=" * 70)
    print("🏥 QDRANT HEALTH CHECK")
    print("=" * 70)
    
    try:
        # Try to get collections (simple health check)
        collections = client.get_collections()
        print("\n✅ Qdrant is healthy!")
        print(f"   Collections: {len(collections.collections)}")
        
        # Get version if available
        try:
            import requests
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            response = requests.get(f"{qdrant_url}/")
            if response.status_code == 200:
                data = response.json()
                if 'version' in data:
                    print(f"   Version: {data['version']}")
        except:
            pass
        
        print()
        
    except Exception as e:
        print(f"\n❌ Qdrant is not healthy!")
        print(f"   Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if Qdrant is running:")
        print("      docker ps | grep qdrant")
        print("   2. Start Qdrant:")
        print("      docker-compose up -d qdrant")
        print("   3. Check logs:")
        print("      docker-compose logs qdrant")
        print()


def main():
    """Main function"""
    
    # Get Qdrant URL
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    print("=" * 70)
    print("🔷 QDRANT DATABASE VIEWER")
    print("=" * 70)
    print(f"Qdrant URL: {qdrant_url}\n")
    
    # Connect to Qdrant
    try:
        client = QdrantClient(url=qdrant_url)
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant at {qdrant_url}")
        print(f"   Error: {e}")
        print("\n💡 Make sure Qdrant is running:")
        print("   docker-compose up -d qdrant")
        return
    
    # Parse command
    if len(sys.argv) == 1:
        # No arguments - show all collections
        view_collections(client)
        check_health(client)
        
    elif len(sys.argv) == 2:
        command = sys.argv[1]
        
        if command == "health":
            check_health(client)
        else:
            # Assume it's a collection name
            view_collection_details(client, command)
    
    elif len(sys.argv) >= 3:
        command = sys.argv[1]
        
        if command == "search":
            collection_name = sys.argv[2]
            query = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
            
            if not query:
                print("Usage: python view_qdrant.py search <collection_name> <query>")
                return
            
            search_similar(client, collection_name, query)
        
        elif command == "delete":
            collection_name = sys.argv[2]
            delete_collection(client, collection_name)
        
        else:
            print("Unknown command")
            print_usage()
    
    else:
        print_usage()


def print_usage():
    """Print usage information"""
    print("\nUsage:")
    print("  python view_qdrant.py                              # View all collections")
    print("  python view_qdrant.py health                       # Check Qdrant health")
    print("  python view_qdrant.py <collection_name>            # View collection details")
    print("  python view_qdrant.py search <collection> <query>  # Search similar vectors")
    print("  python view_qdrant.py delete <collection_name>     # Delete collection")
    print("\nExamples:")
    print("  python view_qdrant.py")
    print("  python view_qdrant.py math_philosophy")
    print("  python view_qdrant.py search math_philosophy 'What is mathematics?'")
    print("  python view_qdrant.py delete old_collection")


if __name__ == "__main__":
    main()
