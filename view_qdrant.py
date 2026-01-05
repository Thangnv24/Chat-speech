import os
import sys
from qdrant_client import QdrantClient


def view_all(client):
    collections = client.get_collections()
    
    if not collections.collections:
        print("No collections found")
        return
    
    print(f"\nCollections: {len(collections.collections)}\n")
    
    for col in collections.collections:
        info = client.get_collection(col.name)
        print(f"{col.name}")
        print(f"  Points: {info.points_count:,}")
        print(f"  Status: {info.status}")
        
        if hasattr(info.config, 'params') and hasattr(info.config.params, 'vectors'):
            vec = info.config.params.vectors
            if hasattr(vec, 'size'):
                print(f"  Vector size: {vec.size}")
        print()


def view_detail(client, name):
    info = client.get_collection(name)
    
    print(f"\nCollection: {name}")
    print(f"Points: {info.points_count:,}")
    print(f"Status: {info.status}")
    
    if hasattr(info.config, 'params') and hasattr(info.config.params, 'vectors'):
        vec = info.config.params.vectors
        if hasattr(vec, 'size'):
            print(f"Vector size: {vec.size}")
        if hasattr(vec, 'distance'):
            print(f"Distance: {vec.distance}")
    
    print(f"\nSample points (first 5):")
    points, _ = client.scroll(
        collection_name=name,
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    
    if not points:
        print("No points")
        return
    
    for i, point in enumerate(points, 1):
        print(f"\n[{i}] ID: {point.id}")
        if point.payload:
            for key, val in point.payload.items():
                if isinstance(val, str) and len(val) > 100:
                    val = val[:100] + "..."
                print(f"  {key}: {val}")


def main():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    try:
        client = QdrantClient(url=url)
    except Exception as e:
        print(f"Cannot connect to {url}: {e}")
        return
    
    if len(sys.argv) == 1:
        view_all(client)
    
    # python view_qdrant.py math_philosophy
    elif len(sys.argv) == 2:
        view_detail(client, sys.argv[1])
    else:
        print("Usage: python view_qdrant.py [collection_name]")


if __name__ == "__main__":
    main()