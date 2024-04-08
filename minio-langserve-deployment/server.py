"""
This server.py file is generated with langchain app new <app-name>
"""
from packages.agent import agent_executor
from langserve import add_routes

# Depending on whether or not `input` and `output` are handled via the `agent.py` logic.
# from langserve.pydantic_v1 import BaseModel
#
# class Input(BaseModel):
#     input: str
#
# class Output(BaseModel):
#     output: Any

add_routes(app, agent_executor, path="/invoke-without-types")
add_routes(
   app,
   agent_executor.with_types(input_type=Input, output_type=Output).with_config(
       {"run_name": "agent"}
   ), path="/invoke"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
