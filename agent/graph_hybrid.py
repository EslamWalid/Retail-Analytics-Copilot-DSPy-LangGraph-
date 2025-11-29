from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import List, Optional, Any
import sqlite3
from pydantic import BaseModel,Field
from typing import List, Optional, Any, Dict
import operator
from typing import Annotated
from agent.dspy_signatures import RouterSignature
from agent.dspy_signatures import NL2SQLSignature
from agent.dspy_signatures import SynthesizerSignature
from agent.rag.retrieval import SimpleRetriever
from agent.tools.sqlite_tool import SQLiteTool

retries_golbal = 0
retrieval_tool = SimpleRetriever(docs_dir=r"E:\Projects\copilot_DSPy\docs", top_k=3)
SQL = SQLiteTool("northwind.db")
router_model = RouterSignature()
nl2sql_model = NL2SQLSignature()
synth_model = SynthesizerSignature()


class RAGState(BaseModel):
    chunks: List[Dict] = []  # {chunk_id, text, score}
    def __add__(self, other):
        return RAGState(chunks=self.chunks + other.chunks)

class SQLState(BaseModel):
    query: Optional[str] = None
    rows: Optional[List[List[Any]]] = None
    columns: Optional[List[str]] = None
    error: Optional[str] = None
    def __add__(self, other):
        # last-write-wins for query, merge rows
        return SQLState(
            query=other.query or self.query,
            rows=(self.rows or []) + (other.rows or []),
            columns=other.columns or self.columns,
            error=other.error or self.error,
        )

class PlannerState(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    kpi_expr: Optional[str] = None
    entities: List[str] = []

class GlobalState(BaseModel):
    user_query: str
    router_decision: Optional[str] = None
    planner: PlannerState = PlannerState()
    rag: Annotated[RAGState, operator.add] = RAGState()
    sql: Annotated[SQLState, operator.add] = SQLState()
    final_answer: str = None
    meta: Dict[str, int] = Field(default_factory=lambda: {"retries": 0})
    event_log: List[Dict] = []
    retries: Annotated[int, operator.add] = 0 


# {"id":"rag_policy_beverages_return_days","question":"According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.","format_hint":"int"}

# keep a replayable event log (file/console)
def checkpoint_node(state: GlobalState, node_name: str, output: Dict):
    entry = {
        "node": node_name,
        "query": state.user_query,
        "router": state.router_decision,
        "output": output,
    }
    state.event_log.append(entry)
    print(f"[TRACE] {entry}")
    return {"event_log": state.event_log}

# Router (DSPy classifier encouraged): rag | sql | hybrid
def router_node(state: GlobalState):
    decision = router_model.predict(state.user_query)
    state.router_decision = decision
    checkpoint_node(state, "Router", {"router_decision": decision})
    return {"router_decision": decision}

# Retriever: top-k doc chunks + scores (include chunk IDs)
def retriever_node(state: GlobalState, docs_dir="docs/"):
    # Dummy TF-IDF or simple retrieval placeholder
    # Assign chunk IDs manually or from filenames
    # chunks = [
    #     {"chunk_id": "marketing_calendar::chunk0", "text": "Summer Beverages 1997: 1997-06-01 to 1997-06-30", "score": 0.95},
    #     {"chunk_id": "kpi_definitions::chunk0", "text": "AOV = SUM(UnitPrice*Quantity*(1-Discount))/COUNT(DISTINCT OrderID)", "score": 0.9},
    # ]
    chunks = retrieval_tool.retrieve(state.user_query)
    state.rag.chunks = chunks
    return {"rag": state.rag}

# Planner: extract constraints (date ranges, KPI formula, categories/entities)
def planner_node(state: GlobalState):
    q = state.user_query.lower()
    # Dummy extraction rules
    if "1997-06" in q or "summer" in q:
        state.planner.date_from = "1997-06-01"
        state.planner.date_to = "1997-06-30"
    if "1997-12" in q or "winter" in q:
        state.planner.date_from = "1997-12-01"
        state.planner.date_to = "1997-12-31"
    if "revenue" in q:
        state.planner.kpi_expr = "SUM(UnitPrice*Quantity*(1-Discount))"
    # Fake entities
    state.planner.entities = ["Orders", "Products", "Customers"]
    return {"planner": state.planner}

# NL→SQL (DSPy): generate SQLite queries using live schema (PRAGMA)
def nl2sql_node(state: GlobalState):
    sql = nl2sql_model.predict(state.user_query, state.planner.model_dump())
    state.sql.query = sql
    checkpoint_node(state, "NL2SQL", {"sql": sql})
    return {"sql": state.sql}

# Executor: run SQL; capture columns, rows, error
def executor_node(state: GlobalState, db_path="northwind.db"):

    try:
        SQL_tool = SQLiteTool("northwind.db")
        if rows and isinstance(rows[0], dict):
            rows = [list(r.values()) for r in rows]
        rows, columns = SQL_tool.execute_query(state.sql.query)
        state.sql.rows = rows
        state.sql.columns = columns
        state.sql.error = None
        return {"status": "ok","sql":state.sql }
    except Exception as e:
        state.sql.error = str(e)
        state.sql.rows = None
        state.sql.columns = None
        return {"status": "error","sql":state.sql }


# Synthesizer (DSPy): produce typed answer matching format_hint, with citations
def synthesizer_node(state: GlobalState):
    print("RAG CHUNKS:", state.rag.chunks)
    print("SQL STATE:", state.sql)
    answer = synth_model.predict(user_query=state.user_query, rag_chunks = state.rag.chunks, sql_state = state.sql.model_dump(), format_hint="str")
    print("SYNTHESIZED ANSWER:", answer)
    state.final_answer = answer
    checkpoint_node(state, "Synthesizer", {"final_answer": answer})
    return {"final_answer": answer, "sql": state.sql, "rag": state.rag}


# Checkpointer/trace: keep a replayable event log (file/console)
def repair_node(state: GlobalState):
    global retries_golbal
    retries_golbal += 1
    state.retries = retries_golbal
    print(f"[REPAIR] Attempt #{state.meta['retries']}")
    print(f"[REPAIR] Attempt #{state.retries}")
    if retries_golbal >= 2:
        print("[REPAIR] Max retries reached. Aborting.")
        return {"final_answer": "Error could not be fixed after 2 retries."}
    # Simple repair strategy
    if state.sql.error:
        state.sql.query = "SELECT * FROM orders LIMIT 3;"
        return {"sql": state.sql}
    if not state.rag.chunks:
        state.rag.chunks = [{"doc_id": "fallback::chunk0", "content": "Fallback doc", "score": 0.5}]
        return {"rag": state.rag}
    return {}



def build_hybrid_graph(State = GlobalState):
    graph = StateGraph(State)

    # 1) Router
    graph.add_node("router", router_node)

    # 2) Retriever (RAG)
    graph.add_node("retriever", retriever_node)

    # 3) Planner (extract constraints)
    graph.add_node("planner", planner_node)

    # 4) NL→SQL Generator
    graph.add_node("nl2sql", nl2sql_node)

    # 5) SQL Executor
    graph.add_node("executor", executor_node)

    # 6) Synthesizer (DSPy)
    graph.add_node("synth", synthesizer_node)

    # 7) Repair node (if SQL fails or output invalid)
    graph.add_node("repair", repair_node)
    

    # ---------------------------------------------------------
    # FLOW: START → router
    # ---------------------------------------------------------
    graph.add_edge(START, "router")

    # ROUTER branch:
    #   rag     → retriever
    #   sql     → nl2sql
    #   hybrid  → retriever (then planner)
    graph.add_conditional_edges(
        "router",
        lambda state: state.router_decision,
        {
            "rag": "retriever",
            "sql": "nl2sql",
            "hybrid": "retriever",
        }
    )

    # ---------------------------------------------------------
    # RAG Path (rag or hybrid)
    # ---------------------------------------------------------
    graph.add_edge("retriever", "planner")    # retrieve docs → constraints
    graph.add_edge("planner", "nl2sql")       # generate SQL when needed

    # ---------------------------------------------------------
    # SQL path (router said 'sql' or planner → nl2sql)
    # ---------------------------------------------------------
    graph.add_edge("nl2sql", "executor")      # SQL → execute

    # ---------------------------------------------------------
    # After exec: If success → synth; If fail → repair
    # ---------------------------------------------------------
    graph.add_conditional_edges(
        "executor",
        lambda state: "ok" if not state.sql.error else ("ok" if not state.retries >2 else "error"),
        {
            "ok": "synth",
            "error": "repair",
            
        }
    )

    # ---------------------------------------------------------
    # Repair node loops back to nl2sql
    # (max 2 attempts — enforce inside repair_node or state)
    # ---------------------------------------------------------
    graph.add_edge("repair", "nl2sql")

    # ---------------------------------------------------------
    # Synth → END
    # ---------------------------------------------------------
    graph.add_edge("synth", END)

    return graph.compile()

graph = build_hybrid_graph()

def extract_synth_fields(state):
    # 1. final answer (already computed by your agent)
    final_answer = state.get("final_answer")

    # 2. SQL query used
    sql_query = None
    if "sql" in state and state["sql"] and hasattr(state["sql"], "query"):
        sql_query = state["sql"].query

    # 3. Explanation (based on router decision)
    if state.get("router_decision") == "sql":
        explanation = "Answer was produced entirely using SQL."
    elif state.get("router_decision") == "rag":
        explanation = "Answer was produced using RAG only."
    else:
        explanation = "Answer was produced using a hybrid of SQL + RAG."

    # 4. Citations (tables + RAG chunks)
    citations = []

    # Allowed canonical table names
    canonical_tables = ["Products", "Sales", "Orders", "Employees", "Customers", "Policies"]

    # Add table names by looking for them inside the SQL query
    if sql_query:
        q = sql_query.lower()
        for tbl in canonical_tables:
            if tbl.lower() in q:
                citations.append(tbl)

    # Add RAG citations
    if "rag" in state and hasattr(state["rag"], "chunks"):
        for c in state["rag"].chunks:
            citations.append(str(c["doc_id"]))

    return {
        "final_answer": final_answer,
        "sql": sql_query,
        "explanation": explanation,
        "citations": citations,
        "confidence": 0.9,
    }



def run_agent(query: str, format_hint: Any):
    state = GlobalState(user_query=query)
    state = graph.invoke(state)
    answer= extract_synth_fields(state)
    return answer

if __name__ == "__main__":
    query = "Average Order Value during 'Winter Classics 1997'? Return a float rounded to 2 decimals."
    state = run_agent(query)
    print("\nFINAL ANSWER:\n", state)



