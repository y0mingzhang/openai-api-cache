#!/usr/bin/env python

from distutils.core import setup

setup(
    name="openai-api-cache",
    version="0.2",
    description="Redis-based Cache for OpenAI API calls.",
    author="Yiming Zhang",
    author_email="yimingz0@uchicago.edu",
    py_modules=["openai_api_cache"],
    install_requires=["requests", "openai", "redis"],
)
