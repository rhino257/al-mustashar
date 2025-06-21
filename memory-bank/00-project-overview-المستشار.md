# Project Overview: "المستشار"
## 1. Core Project Goal
 To develop, refine, and deploy "المستشار," a sophisticated, stateful Retrieval Augmented Generation (RAG) agent specializing in Yemeni law (primarily in Arabic). The overarching aim is to build the most intelligent legal AI agent for Yemen, capable of performing deep legal analysis, interacting like a skilled lawyer, and providing demonstrably accurate answers with zero hallucination.

 ## 2. Strategic Foundation
 The "LangGraph RAG Research Agent Template" is the initial technical foundation. The project involves migrating and significantly enhancing an existing backend RAG pipeline into a new LangGraph-based agent architecture, evolving towards an Agentic Neuro-Symbolic RAG with Multi-Layered Verification & Learning.

 ## 3. Key High-Level Functionalities to Build
 - Advanced Retrieval Engine (ARE)
 - Yemen Legal Knowledge Graph (YLKG)
 - Hybrid Reasoning Module (HRM)
 - Multi-Layered Verification & Confidence Module (MVCM)
 - Advanced Synthesis Module (ASM)
 - Data & Learning Infrastructure (DLI)

 ## 4. Target Audience & Language
 - Primary users: Legal professionals, researchers, and citizens in Yemen.
 - Primary language: Arabic.

 ## 5. Technology Stack (Initial & Planned)
 - LangGraph
 - Python
 - LLMs (Initial: Gemini via custom provider; *To be benchmarked and confirmed: Jais, LlamAr, ALKAFI variants*)
 - Embedding Models (Initial: OpenAI via custom provider; *To be benchmarked and confirmed for Arabic legal text*)
 - Vector Database: Pinecone
 - Metadata/Relational Database: Supabase (for initial direct lookups, will evolve for DLI)
 - Graph Database: (To be selected, e.g., Neo4j, ArangoDB for YLKG)
 - FastAPI (for API deployment)
