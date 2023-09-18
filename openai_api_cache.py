""" A redis-based cache wrapper for GPT-3/Jurassic-1 API to avoid duplicate requests """
import hashlib
import collections
import pickle
import logging
import time
from abc import ABC
from typing import Any

import redis
import openai

logger = logging.getLogger(name="OpenAIAPICache")


def deterministic_hash(data) -> int:
    try:
        data_str = str(data).encode("utf-8")
    except:
        raise Exception(f"Unable to convert type {type(data)} to string.")
    return int(hashlib.sha512(data_str).hexdigest(), 16)


class FrozenDict:
    """frozen, hashable mapping"""

    def __init__(self, mapping):
        self.data = {}
        for key, value in mapping.items():
            if not isinstance(key, collections.abc.Hashable):
                raise Exception(f"{type(key)} is not hashable")
            if not isinstance(value, collections.abc.Hashable):
                if isinstance(value, collections.abc.Mapping):
                    value = FrozenDict(value)
                elif isinstance(value, collections.abc.Sequence):
                    value = tuple(value)
                else:
                    raise Exception(f"{type(value)} is not hashable")
            self.data[key] = value

    def __hash__(self):
        ordered_keys = sorted(self.data.keys(), key=deterministic_hash)
        return deterministic_hash(tuple((k, self.data[k]) for k in ordered_keys))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        raise Exception("FrozenDict is immutable")

    def __repr__(self):
        return repr(self.data)

    def __eq__(self, other):
        return self.data == other.data


class RateLimiter:
    def __init__(self, window: float, max_rate: int):
        self.window = window
        self.max_rate = max_rate
        self.backoff_time = 1.0

    def backoff(self):
        logger.debug(f"Backing off for {self.backoff_time} seconds")
        time.sleep(self.backoff_time)
        self.backoff_time *= 2
        logger.debug(f"Setting backoff time to {self.backoff_time} seconds")

    def add_event(self):
        self.backoff_time = 1.0
        time.sleep(self.window / self.max_rate)


class APICache(ABC):
    """Abstract base class for GPT-3/Jurassic cache wrappers.

    Should not be instantiated on its own."""

    def __init__(self, port: int):
        logger.info(f"Connecting to Redis DB on port {port}")
        self.r = redis.Redis(host="localhost", port=port)

        # max 60 requests per 60 seconds
        self.rate_limiter = RateLimiter(60.0, 60)
        self.costs = []

    def generate(self, overwrite_cache: bool = False, **kwargs):
        """Makes an API request if not found in cache, and returns the response.

        Args:
            overwrite_cache: If true, ignore and overwrite existing cache.
              Useful when sampling multiple times.
            **kwargs: Generation specific arguments passed to the API.

        Returns:
            A JSON-like API response.
        """
        query = FrozenDict(kwargs)
        hashval = hash(query)
        cache = self.r.hget(hashval, "data")
        if overwrite_cache:
            logger.debug("Overwriting cache")
        elif cache is not None:
            query_cached, resp_cached = pickle.loads(cache)
            if query_cached == query:
                logger.debug(f"Matched cache for query")
                return resp_cached
            logger.debug(
                f"Hash matches for query and cache, but contents are not equal. "
                + "Overwriting cache."
            )
        else:
            logger.debug(f"Matching hash not found for query")

        self.rate_limiter.add_event()
        logger.debug(f"Request Completion from {self.service} API...")

        while 1:
            try:
                resp = self.api_call(**kwargs)
                break
            except (openai.error.RateLimitError, openai.error.APIError):
                logger.warning("Getting an error from openai API, backing off...")
                self.rate_limiter.backoff()

        data = pickle.dumps((query, resp))
        logger.debug(f"Writing query and resp to Redis")
        self.r.hset(hashval, "data", data)
        self.compute_cost(resp)
        return resp

    def compute_cost(self, resp):
        ...

    def session_total_cost(self) -> float:
        return sum(self.costs)


class OpenAIAPICache(APICache):
    """A cache wrapper for OpenAI's Chat and Completion API calls.

    Typical usage example:

      api = OpenAIAPICache(open("key.txt").read().strip(), 6379)
      resp = api.generate(model="text-davinci-002", prompt="This is a test", temperature=0.0)
    """

    def __init__(self, port: int = 6379, mode: str = "completion"):
        """Initializes an OpenAI Cache Object.

        Args:
            port: Port of the Redis backend.
            mode: "completion" or "chat", determines which API to call
        """
        self.mode = mode
        if mode == "completion":
            self.api_call = openai.Completion.create
        elif mode == "chat":
            self.api_call = openai.ChatCompletion.create
        super().__init__(port)

    service = "OpenAI"
