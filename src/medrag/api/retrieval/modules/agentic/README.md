# agentic

Status: **planned, not started.**

Agentic retrieval flows (LangChain / LangGraph / LlamaIndex based agents that
plan retrieval steps, call tools, loop). Design and interface TBD.

Extra: `pip install -e ".[agentic]"`

Note: this extra will likely be the heaviest one dependency-wise — keep it
isolated so modules that don't need an agent framework never have to install
one.
