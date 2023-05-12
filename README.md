# openai-api-cache

## What is this

A cache wrapper for OpenAI's API to avoid duplicate requests.

## Dependency

This code uses [Redis](https://redis.io) as the backend cache server.
It will probably take ~5 minutes to [install from source](https://redis.io/docs/getting-started/installation/install-redis-from-source/).

After setting up Redis, enable a local Redis server and keep note of which port
it runs on!

## Install

`pip install git+https://github.com/Y0mingZhang/openai-api-cache.git`

## Example Usage

**Completion mode**
```python
api_key = open("key.txt").read().strip()
api = OpenAIAPICache(api_key, port=6379)

resp = api.generate(
    model="text-davinci-002",
    prompt="testing..",
    max_tokens=50
)

```

**Chat mode**
```python
api_key = open("key.txt").read().strip()
api = OpenAIAPICache(api_key, port=6379)

resp = api.generate(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

```
