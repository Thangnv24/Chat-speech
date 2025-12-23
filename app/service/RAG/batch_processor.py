import concurrent.futures
from typing import List, Callable, TypeVar, Any
from itertools import islice

# Process parallel with ThreadPoolExecutor for speeding up embedding
# batch size 32, worker = 4
T = TypeVar('T')
R = TypeVar('R')

class BatchProcessor:
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def _batches(self, items: List[T]) -> List[List[T]]:
        iterator = iter(items)
        while batch := list(islice(iterator, self.batch_size)):
            yield batch
    
    def process_parallel(self, items: List[T], process_fn: Callable[[List[T]], List[R]]) -> List[R]:
        batches = list(self._batches(items))
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(process_fn, batch): batch for batch in batches}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_result = future.result()
                    results.extend(batch_result)
                except Exception as e:
                    raise Exception(f"Batch processing failed: {e}")
        
        return results
    
    def embed_documents_parallel(self, embedder, texts: List[str]) -> List[List[float]]:
        def embed_batch(batch: List[str]) -> List[List[float]]:
            return embedder.embed_documents(batch)
        
        return self.process_parallel(texts, embed_batch)