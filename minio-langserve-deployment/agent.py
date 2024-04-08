# Helper function to set undefined vars with getpass
import getpass
import os
import uuid

os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")

# Optional, add tracing in LangSmith.
# This will help you visualize and debug the control flow
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Langchain MinIO Tool"

from langchain_openai import ChatOpenAI
# Initialize the LLM with your OpenAI API key
llm = ChatOpenAI(api_key="")

# Necessary imports
import io
from langchain.agents import tool
from minio import Minio
from minio.error import S3Error

# Initialize MinIO client
minio_client = Minio('play.min.io:443',
                     access_key='minioadmin',
                     secret_key='minioadmin',
                     secure=True)

# This variable will check if bucket exisits  
bucket_name = "test"

try:
    # Check if bucket exists
    if not minio_client.bucket_exists(bucket_name):
        # Create the bucket because it does not exist
        minio_client.make_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' created successfully.")
    else:
        print(f"Bucket '{bucket_name}' already exists.")
except S3Error as err:
    print(f"Error encountered: {err}")

# This is the upload function
@tool
def upload_file_to_minio(bucket_name: str, object_name: str, data_bytes: bytes):
    """
    Uploads a file to MinIO.
    Parameters:
        bucket_name (str): The name of the bucket.
        object_name (str): The name of the object to create in the bucket.
        data_bytes (bytes): The raw bytes of the file to upload.
    """
    data_stream = io.BytesIO(data_bytes)
    minio_client.put_object(bucket_name, object_name, data_stream, length=len(data_bytes))
    return f"File {object_name} uploaded successfully to bucket {bucket_name}."

# This is the download function
@tool
def download_file_from_minio(file_info):
    """
    Custom function to download a file from MinIO.
    Expects file_info dict with 'bucket_name', 'object_name', and 'save_path' keys.
    'save_path' should be the local path where the file will be saved.
    """
    bucket_name = file_info['bucket_name']
    object_name = file_info['object_name']
    save_path = file_info['save_path']

    minio_client.get_object(bucket_name, object_name, save_path)

# This is the list function
@tool
def list_objects_in_minio_bucket(file_info):
    """
    Custom function to list objects in a MinIO bucket.
    Expects file_info dict with 'bucket_name' key.
    Returns a list of dictionaries containing 'ObjectKey' and 'Size' keys.
    """
    bucket_name = file_info['bucket_name']

    response = minio_client.list_objects(bucket_name)

    return [{'ObjectKey': obj.object_name, 'Size': obj.size} for obj in response.items]

from langchain_core.runnables import RunnableLambda
# Create a RunnableLambda for each function

upload_file_runnable = RunnableLambda(upload_file_to_minio)
download_file_runnable = RunnableLambda(download_file_from_minio)
list_objects_runnable = RunnableLambda(list_objects_in_minio_bucket)

#### 4. Create Prompt Template
# This step involves creating a ChatPromptTemplate that incorporates the MinIO upload tool and any additional tools.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a powerful assistant equipped with file management capabilities."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

#### 5. Bind Tools to LLM
tools = [upload_file_to_minio, download_file_from_minio, list_objects_in_minio]
llm_with_tools = llm.bind_tools(tools)

#### 6. Create Agent and Executor
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

""" #### 7. Add Memory Management
# Integrating memory management allows the agent to maintain context across user interactions, enhancing its ability to handle follow-up queries effectively.
from langchain_core.messages import AIMessage, HumanMessage

# Update the prompt to include memory management
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a powerful assistant with memory capabilities."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Initialize chat history
chat_history = []

# Implementing RunnableParallel/RunnablePassthrough in agent
agent = (
     RunnableParallel({
         "input": RunnableLambda(lambda x: x["input"]),
         "agent_scratchpad": RunnableLambda(lambda x: format_to_openai_tool_messages(x["intermediate_steps"])),
         "chat_history": RunnablePassthrough(),
     })
     | prompt
     | llm_with_tools
     | OpenAIToolsAgentOutputParser()
)
"""

# Update agent definition without chat_history
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

""" 
# Test prompting and file structure
upload_object_prompt = "Upload an object_name with some funny name to the 'test' bucket with some example content"
list_objects_prompt = "List any path and {object_name} within {bucket_name}"
retrieve_object_prompt = "Retrieve the {object_name} from the {bucket_name}"

# Simulate the agent executor invoking the MinIO upload tool
result = agent_executor.invoke({"input": input1, "chat_history": chat_history, "file_info": file_info})
chat_history.extend([
    HumanMessage(content=input1),
    AIMessage(content=result["output"]),
])
"""
