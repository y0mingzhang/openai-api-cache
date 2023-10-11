#!/usr/bin/env python

from distutils.core import setup

setup(
    name="openai-api-cache",
    version="1.0.0",
    description="Redis-based Cache for OpenAI API calls.",
    author="Yiming Zhang",
    author_email="yimingz3@cs.cmu.edu",
    py_modules=["openai_api_cache"],
    install_requires=["openai", "cohere", "redis"],
)
