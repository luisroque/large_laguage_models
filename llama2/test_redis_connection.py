import redis

try:
    # Connect to the Redis server running on localhost (assuming default port 6379)
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # Test connection
    response = r.ping()
    if response:
        print("Redis server is running and accessible.")
    else:
        print("Redis server is not responding.")
except redis.exceptions.ConnectionError:
    print("Failed to connect to Redis server.")
