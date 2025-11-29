from dspy import Signature, Example
from typing import Dict, Any, List,Literal
import dspy


lm = dspy.LM('ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)


class Router(dspy.Signature):
    """Classify which node to use."""

    sentence: str = dspy.InputField()
    node: Literal['rag', 'sql', 'hybrid'] = dspy.OutputField()

class TextToSQL(dspy.Signature):
    """Generates a SQL query from a natural language request."""
    user_request: str = dspy.InputField(desc="The natural language request for data.")
    sql_query: str = dspy.OutputField(desc="The generated SQL only.")


class SynthSig(dspy.Signature):  
    """produce typed answer """
    context: str =dspy.InputField(desc ="retrieved docs")
    user: str =dspy.InputField(desc="input of user")
    response : Any = dspy.OutputField()

# ------------------------
# 1. Router Signature
# ------------------------
class RouterSignature(Signature):
    """
    DSPy signature for routing a user query to 'rag', 'sql', or 'hybrid'
    """
    def predict(self, query: str) -> str:
        """
        Return one of 'rag', 'sql', or 'hybrid'
        """

        classify = dspy.ChainOfThought(Router,n=3, allowed_openai_params=['n'])
        classification = classify(sentence = query)
        
        return classification.node

# Example usage:
# router = RouterSignature()
# decision = router.predict("Top 3 products by revenue all-time")

# ------------------------
# 2. NL→SQL Signature
# ------------------------
class NL2SQLSignature(Signature):
    """
    DSPy signature to generate SQL from user query + constraints + schema
    """
    def predict(self, query: str, planner: Dict[str, Any]) -> str:
        """
        Return SQL string
        """
        # Rules / template-based SQL generation   
        sql_query = dspy.ChainOfThought(TextToSQL,n=3,allowed_openai_params=['n'])
        sql = sql_query(user_request = query )

        return sql.sql_query

# ------------------------
# 3. Synthesizer Signature
# ------------------------
class SynthesizerSignature(Signature):
    """
    DSPy signature to produce typed answer + citations
    """
    def predict(self, user_query:str, rag_chunks: List[Dict], sql_state: Dict[str, Any], format_hint: str) -> Any:
        """
        rag_chunks: list of {doc_id, content, score}
        sql_state: {rows, columns, query}
        format_hint: expected output type (e.g., int, float, list, dict)
        """
        respond = dspy.ChainOfThought(SynthSig,n=3,allowed_openai_params=['n'])
        # print("respond:",respond)
        rag_text = "\n".join(f"[{c['doc_id']}] {c['content']}" for c in rag_chunks)
        if sql_state.get("rows"):
            sql_summary = f"SQL returned {len(sql_state['rows'])} rows."
        else:
            sql_summary = "No SQL rows."
        user_input = user_query + f" {format_hint}"
        answer=respond(context =f"### Final Answer\nUsing documents:\n{rag_text}\nUsing SQL: {sql_summary}", user= user_input )

        
        print("RESPOND:",answer.response)
        # Optionally, you can cast answer to exact format_hint here
        return answer.response


