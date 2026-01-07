# Overview

Control and customize agent execution at every step

Middleware provides a way to more tightly control what happens inside the agent. Middleware is useful for the following:

*   Tracking agent behavior with logging, analytics, and debugging.
*   Transforming prompts, tool selection, and output formatting.
*   Adding retries, fallbacks, and early termination logic.
*   Applying rate limits, guardrails, and PII detection.

Add middleware by passing them to `create_agent`:
