"""chatbot -- orchestration layer that wires retrieval's (mkd-retriever)
modules into a single conversation flow: classify intent -> deterministic
routing -> run flow -> call the answering model.

Deliberately separate from `retrieval`: that package rejects orchestration by
design (retrieval/ARCHITECTURE.md #6) -- this component is the "consuming
project" it expects. See README.md for the full picture.
"""

__version__ = "0.1.0"
