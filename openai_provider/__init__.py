"""OpenAI-compatible LLMProvider for KiroCrew.

Drop-in replacement for AcpProvider — routes to any OpenAI-compatible endpoint
while emitting the same AcpEvent stream KiroCrew expects.

Usage::

    from openai_provider import install
    install()   # call before kirocrew gateway starts
"""

from .provider import OpenAIProvider
from .install import install

__all__ = ["OpenAIProvider", "install"]
