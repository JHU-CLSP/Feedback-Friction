import redis
import json
from datetime import datetime
import hashlib

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._print_status()
    
    def _print_status(self):
        """Print current cache status"""
        try:
            info = self.redis.info()
            print("\n=== Redis Cache Status ===")
            print(f"Connected to: {info['tcp_port']}")
            print(f"Keys: {self.redis.dbsize()}")
            print(f"Memory: {info['used_memory_human']}")
            print("=========================\n")
        except redis.ConnectionError:
            print("Could not connect to Redis!")

    def _generate_key(self, prompt, model):
        """Generate a stable key for prompt+model combination"""
        combo = f"{prompt}:{model}"
        return f"llm:{hashlib.md5(combo.encode()).hexdigest()}"

    def store(self, prompt, response, **kwargs):
        """ 
        Create or update a record in one step:
        - If key doesn't exist, create it.
        - If key does exist, update/overwrite it.
        """
        model = kwargs.get('model', 'default')
        key = self._generate_key(prompt, model)

        # Detect if we are creating a new record or updating an existing one
        new_record = not self.redis.exists(key)

        data = {
            'prompt': prompt,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        pipe = self.redis.pipeline()

        # Create or overwrite the hash
        pipe.hset(key, mapping=data)

        # (Re-)Index by model
        pipe.sadd(f"idx:model:{model}", key)

        # (Re-)Index by temperature if provided
        if 'temperature' in kwargs:
            # Remove old score (if any), then add the new one
            pipe.zrem("idx:temp", key)
            pipe.zadd("idx:temp", {key: float(kwargs['temperature'])})

        # Add to global index
        pipe.sadd("idx:all", key)

        # Word-index the prompt (if you still need it)
        words = set(prompt.lower().split())
        for word in words:
            pipe.sadd(f"idx:word:{word}", key)

        pipe.execute()

        if new_record:
            print(f"[STORE] Created new entry for key: {key}")
        else:
            print(f"[STORE] Updated existing entry for key: {key}")

        return data

    
    def create(self, prompt, response, **kwargs):
        """
        Create a new LLM record only if the key does NOT exist.
        If the key already exists, do nothing.
        """
        model = kwargs.get('model', 'default')
        
        # Generate Redis key
        key = self._generate_key(prompt, model)

        # Check if key already exists
        if self.redis.exists(key):
            print(f"[CREATE] Key already exists. Skipping. (key={key})")
            return None

        # Create a new record
        data = {
            'prompt': prompt,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        pipe = self.redis.pipeline()
        
        # Create the hash
        pipe.hset(key, mapping=data)
        
        # Index by model
        pipe.sadd(f"idx:model:{model}", key)
        
        # Index by temperature if provided
        if 'temperature' in kwargs:
            pipe.zadd(f"idx:temp", {key: float(kwargs['temperature'])})
            
        # Add to global index
        pipe.sadd("idx:all", key)
        
        # Index by words (optional)
        words = set(prompt.lower().split())
        for word in words:
            pipe.sadd(f"idx:word:{word}", key)
        
        pipe.execute()
        
        print(f"[CREATE] Created new key: {key}")
        return data

    def get(self, prompt, model='default'):
        """Get entry by exact prompt and model"""
        key = self._generate_key(prompt, model)
        data = self.redis.hgetall(key)
        return data if data else None

    def search(self, prompt=None, model=None, temperature=None, **kwargs):
        """
        Exact-match search for:
          - prompt (must match entire prompt exactly, if provided)
          - model (exact match, if provided)
          - temperature (exact match, if provided)
          - additional kwargs fields must also match exactly

        Returns a list of matches, sorted by descending timestamp.
        """
        # Collect all keys that begin with "llm:"
        all_keys = set(self.redis.keys("llm:*"))
        if not all_keys:
            return []

        matching_keys = all_keys.copy()

        # 1) Filter by model (exact match)
        if model:
            model_keys = self.redis.smembers(f"idx:model:{model}")
            matching_keys &= model_keys
            if not matching_keys:
                return []

        # 2) Filter by temperature (exact match)
        if temperature is not None:
            # Redis ZSET range by score returns members with that exact score
            temp_keys = set(self.redis.zrangebyscore("idx:temp", temperature, temperature))
            matching_keys &= temp_keys
            if not matching_keys:
                return []

        # Gather data
        pipe = self.redis.pipeline()
        for key in matching_keys:
            pipe.hgetall(key)
        results_data = pipe.execute()

        # print(results_data)
        # 3) Filter by prompt (exact match) and additional kwargs
        results = []
        for data in results_data:
            if not data:
                continue

            # If prompt was given, we require an exact match
            if prompt is not None and data.get('prompt') != prompt:
                continue

            # Additional metadata filters (exact match)
            matched_all = True
            for k, v in kwargs.items():
                if str(data.get(k)) != str(v):
                    matched_all = False
                    break

            if matched_all:
                results.append(data)

        # Sort by descending timestamp
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)


    def export(self, filepath="cache_export.json", model=None, dataset=None, type=None):
        """
        Export data to a JSON file. Optionally filter by model
        and sort by descending timestamp.
        """
        data = []

        # If a model is provided, get only keys for that model.
        # Otherwise, get all llm:* keys.
        if model:
            all_keys = self.redis.smembers(f"idx:model:{model}")
        else:
            all_keys = self.redis.keys("llm:*")

        for key in all_keys:
            entry = self.redis.hgetall(key)
            if entry:
                # You could also check if entry['model'] == model if you want
                # a secondary verification. But if you're using idx:model:{model}
                # set membership, that's probably enough.
                if 'dataset' in entry and entry['dataset'] == dataset and 'type' in entry and entry['type'] == type:
                    data.append(entry)

        # Sort by timestamp, descending
        # We'll handle the case if 'timestamp' doesn't exist by providing a fallback.
        data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath


    def clear(self):
        """Clear all data"""
        self.redis.flushdb()


def one_process(cache):
    # Attempt to create a new record
    cache.create(
        prompt="What's the weather?",
        response="It's sunny!",
        model="gpt-4",
        temperature=0.7
    )
    
    # Attempting to create the same prompt+model again should skip
    cache.create(
        prompt="What's the weather?",
        response="Should never store",
        model="gpt-4",
        temperature=0.7
    )
    
    # Instead, we explicitly update an existing key
    cache.store(
        prompt="What's the weather?",
        response="It's now cloudy!",
        model="gpt-4",
        temperature=0.6
    )
    
    # Exact-match search by prompt, model, and temperature
    results = cache.search(
        prompt="What's the weather?",
        model="gpt-4",
        temperature=0.6
    )
    print(f"Results for exact-match search: {len(results)}")
    for r in results:
        print(r)


# Example usage
if __name__ == "__main__":
    cache = RedisCache()
    # one_process(cache)
    # cache.export("cache_export.json", model='meta-llama/Llama-3.1-70B-Instruct', dataset='gsm8k')
    cache.export("cache_export.json", model='meta-llama/Llama-3.1-70B-Instruct', dataset='gsm8k', type="feedback")
    
