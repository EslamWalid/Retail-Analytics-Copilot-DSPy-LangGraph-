from .rag.retrieval import SimpleRetriever
from .tools.sqlite_tool import SQLiteTool
from .dspy_signatures import NL2SQLSignature
import json

class RetailAnalyticsCopilot:
    def __init__(self, db_path="data/northwind.sqlite", docs_dir="docs/"):
        self.retriever = SimpleRetriever(docs_dir)
        self.sqlite_tool = SQLiteTool(db_path)
        self.nl2sql = NL2SQLSignature()
        self.max_repair = 2

    def router(self, question: str):
        """
        Simple rule: if keywords match docs, use RAG; if contains numbers or KPI, use SQL; else hybrid
        """
        keywords_docs = ["policy", "return", "kpi", "definition"]
        if any(k in question.lower() for k in keywords_docs):
            if "average order value" in question.lower() or "gross margin" in question.lower():
                return "hybrid"
            return "rag"
        return "sql"

    def planner(self, question: str):
        # Extract constraints, dates, KPIs
        return {"question": question}

    def synthesizer(self, sql_result, rag_result, format_hint: str):
        if sql_result:
            if format_hint == "int":
                return int(sql_result[0][list(sql_result[0].keys())[0]])
            elif format_hint == "float":
                return round(float(sql_result[0][list(sql_result[0].keys())[0]]), 2)
            elif format_hint.startswith("{"):
                return sql_result[0]
            elif format_hint.startswith("list"):
                return sql_result
        elif rag_result:
            return rag_result[0]["content"]
        return None

    def execute(self, question: str, format_hint: str):
        route = self.router(question)
        rag_result = []
        sql_result = []
        last_sql = ""
        for attempt in range(self.max_repair):
            if route in ["rag", "hybrid"]:
                rag_result = self.retriever.retrieve(question)
            if route in ["sql", "hybrid"]:
                schema_info = "Tables: Orders, 'Order Details', Products, Customers"
                sql = self.nl2sql.run(question, schema_info)
                last_sql = sql
                try:
                    sql_result = self.sqlite_tool.execute_query(sql)
                    break
                except:
                    continue
        answer = self.synthesizer(sql_result, rag_result, format_hint)
        citations = ["Orders", "Order Details", "Products", "Customers"]
        citations += [r["doc_id"] for r in rag_result] if rag_result else []
        return {
            "final_answer": answer,
            "sql": last_sql,
            "confidence": 0.9,
            "explanation": "Hybrid answer using SQL + RAG.",
            "citations": citations
        }
