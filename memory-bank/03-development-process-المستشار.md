# Development Process & Protocols: "المستشار"
## 1. Iterative Development
 - The project follows an iterative approach, building upon foundational components and progressively adding advanced features.

 ## 2. Supervisory Interaction Protocol with AI Coding Assistant
 - **Investigate First, Instruct Later:** Always require the AI Agent to investigate existing code/state before providing coding instructions.
 - **Extreme Specificity:** Provide exact file names, class/method names, and conceptual logic.
 - **Context and Rationale:** Explain why changes are needed.
 - **Review Agent's Plans/Drafts:** If the Agent proposes significant structures or plans, review them before implementation.
 - **One Task at a Time:** Focus the Agent on single, well-defined tasks.
 - **Iterative Verification:** After the Agent implements a task, provide specific verification instructions to confirm correctness.

 ## 3. Leveraging Existing Code from Previous Pipeline (Refined from User Feedback)
 - **Context:** The Project Lead possesses a working Python-based RAG pipeline that predates the "المستشار" LangGraph project. For many planned functionalities (e.g., synthesis, direct lookup, specific tool integrations), equivalent or similar logic may already exist in this prior pipeline.
 - **Workflow Adjustment for AI Coding Assistant (Cline):**
   - Before planning or implementing new complex nodes or functionalities from scratch, the AI Coding Assistant (Cline) **MUST proactively inquire** if relevant, adaptable code exists from the Project Lead's previous pipeline.
   - If such code is available, the Project Lead will provide it.
   - The AI will then prioritize adapting this existing, proven code to the LangGraph framework, rather than redeveloping it.
 - **Benefits:**
   - Reduces development time.
   - Minimizes the introduction of new bugs.
   - Leverages already refined logic and prompts.
   - Ensures alignment with battle-tested approaches.
 - **Precedent:** This approach was successfully demonstrated during Task 13, where the `UnderstandQueryStep` code was provided by the Project Lead and adapted by the AI.

 ## 4. Testing Strategies and Considerations
 - **Direct Lookup Testing in `tests/integration_tests/test_almustashar_agent.py`**:
   - Currently, the test script *simulates* the MCP (Model Context Protocol) server's response for direct Supabase lookups. This involves a two-pass graph invocation:
     1. The first pass generates arguments for the MCP `execute_sql` tool.
     2. The test script then uses `graph.update_state()` to inject a *simulated* JSON response (mimicking what the actual MCP tool would return) into `AgentState.direct_lookup_mcp_response`.
     3. The second pass of the graph processes this simulated response.
   - **Reason for Simulation:** The `supabase-mcp-server` used in the Vibe Coding development environment is configured with `"transportType": "stdio"`. This makes direct HTTP calls from the Python test script to this specific MCP server instance problematic without a known, concurrently running HTTP endpoint for that server. The Vibe agent (Cline) can interact with `stdio` MCP servers via its `use_mcp_tool`, but a standard Python script cannot easily do the same.
   - **Future Improvement Goal:** For more comprehensive end-to-end testing from the script, the ideal scenario would be for the test script to retrieve *actual* documents from Supabase. This would require:
     - The `supabase-mcp-server` (or a dedicated test instance) to expose a stable HTTP endpoint that the Python test script can call.
     - Or, an alternative integration testing strategy that can bridge the Python test script with the `stdio`-based MCP server if direct HTTP is not feasible.
   - **Production Note:** In a deployed production environment, the "المستشار" application would typically use a direct Supabase client library (e.g., `supabase-py`) for database interactions, rather than relying on an MCP server for this purpose. The current MCP-based direct lookup is a development/testing setup.
