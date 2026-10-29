## Tool-Dispatch Reasoning Instructions
You have access to the following registered tools (JSON Schema):
{tools_catalog}

Evaluate whether any tool above must be executed to answer the User Query (`{user_query}`):
1. **If a tool IS required** and all its required parameters can be extracted from the User Query:
   - Do NOT write your final response yet.
   - Reply **strictly and exclusively** with a single valid JSON object requesting the tool execution:
     `{"tool_calls": [{"name": "<tool_name>", "arguments": {"<param_name>": <extracted_value>}}]}`
2. **If NO tool is required** (e.g., the query is conceptual, unrelated to the tools, or missing required parameters):
   - Do **NOT** output `tool_calls` JSON.
   - Proceed immediately and reply directly in the format requested by your main task above.
