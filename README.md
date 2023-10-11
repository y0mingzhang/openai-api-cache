# openai-api-cache

## What is this

A cache wrapper for OpenAI's API to avoid duplicate requests.

## Dependency

This code uses [Redis](https://redis.io) as the backend cache server.
[Install redis from source](https://redis.io/docs/getting-started/installation/install-redis-from-source/).

Alternatively, run the following to build Redis locally. 

```bash
wget https://download.redis.io/redis-stable.tar.gz
tar -xzvf redis-stable.tar.gz
cd redis-stable
make
```

After setting up Redis, enable a local Redis server and keep note of which port
it runs on!

## Install

`pip install git+https://github.com/Y0mingZhang/openai-api-cache.git`

## Example Usage

**OpenAI, chat mode**
```python
import openai
from openai_api_cache import OpenAIAPICache
openai.api_key_path = "PATH TO A FILE WITH YOUR OPENAI API KEY"

redis_port = 6379
cache = OpenAIAPICache(mode="chat", port=redis_port)

output = cache.generate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=1.0,
        max_tokens=5,
)

```

**OpenAI, completion mode**
```python
import openai
from openai_api_cache import OpenAIAPICache
openai.api_key_path = "PATH TO A FILE WITH YOUR OPENAI API KEY"

redis_port = 6379
cache = OpenAIAPICache(mode="completion", port=redis_port)

output = cache.generate(
        model="text-davinci-003",
        prompt="What exactly is ChatGPT??",
        temperature=1.0,
        max_tokens=20,
)

```

## License
MIT