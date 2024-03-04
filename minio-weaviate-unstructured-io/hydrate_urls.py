import requests
from minio import Minio
import weaviate
import os
import tempfile
import re
from unstructured.partition.auto import partition
import io

# Setup for MinIO and Weaviate
minio_client = Minio("192.168.0.25:9000", access_key="cda_cdaprod", secret_key="cda_cdaprod", secure=False)
# minio_client = Minio("play.min.io:443", access_key="minioadmin", secret_key="minioadmin", secure=True)
print("MinIO client initialized.")

client = weaviate.Client("http://192.168.0.25:8080")
print("Weaviate client initialized.")

bucket_name = "cda-datasets"

def sanitize_url_to_object_name(url):
    clean_url = re.sub(r'^https?://', '', url)
    clean_url = re.sub(r'[^\w\-_\.]', '_', clean_url)
    return clean_url[:250] + '.txt'

def prepare_text_for_tokenization(text):
    clean_text = re.sub(r'\s+', ' ', text).strip()
    return clean_text

urls = [
    "https://nanonets.com/blog/langchain/amp/",
    "https://www.sitepoint.com/langchain-python-complete-guide/",
    "https://medium.com/@aisagescribe/langchain-101-a-comprehensive-introduction-guide-7a5db81afa49",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://quickaitutorial.com/langgraph-create-your-hyper-ai-agent/",
    "https://python.langchain.com/docs/langserve",
    "https://python.langchain.com/docs/expression_language/interface",
    "https://blog.min.io/minio-langchain-tool",
    "https://python.langchain.com/docs/langgraph",
    "https://www.33rdsquare.com/langchain/",
    "https://medium.com/widle-studio/building-ai-solutions-with-langchain-and-node-js-a-comprehensive-guide-widle-studio-4812753aedff", "https://blog.min.io/", "https://sanity.cdaprod.dev/"]

if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created.")

for url in urls:
    print(f"Fetching URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP issues

        html_content = io.BytesIO(response.content)
        elements = partition(file=html_content, content_type="text/html")
        combined_text = "\n".join([e.text for e in elements if hasattr(e, 'text')])
        combined_text = prepare_text_for_tokenization(combined_text)
        object_name = sanitize_url_to_object_name(url)

        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as tmp_file:
            tmp_file.write(combined_text)
            tmp_file_path = tmp_file.name
        
        minio_client.fput_object(bucket_name, object_name, tmp_file_path)
        print(f"Stored '{object_name}' in MinIO bucket '{bucket_name}'.")
        os.remove(tmp_file_path)  # Clean up

    except requests.RequestException as e:
        print(f"Failed to fetch URL {url}: {e}")
    except Exception as e:
        print(f"Error processing {url}: {e}")

for obj in minio_client.list_objects(bucket_name, recursive=True):
    if obj.object_name.endswith('.txt'):
        print(f"Processing document: {obj.object_name}")
        file_path = obj.object_name
        minio_client.fget_object(bucket_name, obj.object_name, file_path)
        
        elements = partition(filename=file_path)
        text_content = "\n".join([e.text for e in elements if hasattr(e, 'text')])
        
        data_object = {"source": obj.object_name, "content": text_content}
        client.data_object.create(data_object, "Document")
        print(f"Inserted document '{obj.object_name}' into Weaviate.")
        
        os.remove(file_path)
        print(f"MinIO and Weaviate have ingested '{obj.object_name}'! :)")
