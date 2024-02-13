import weaviate
import json

# Configuration
WEAVIATE_ENDPOINT = "http://weaviate:8080"
OUTPUT_FILE = "data.json"

# Initialize the client
client = weaviate.Client(WEAVIATE_ENDPOINT)
schema = {
    "classes": [
        {
            "class": "Article",
            "description": "A class to store articles",
            "properties": [
                {"name": "title", "dataType": ["string"], "description": "The title of the article"},
                {"name": "content", "dataType": ["text"], "description": "The content of the article"},
                {"name": "datePublished", "dataType": ["date"], "description": "The date the article was published"},
                {"name": "url", "dataType": ["string"], "description": "The URL of the article"}
            ]
        },
        {
            "class": "Author",
            "description": "A class to store authors",
            "properties": [
                {"name": "name", "dataType": ["string"], "description": "The name of the author"},
                {"name": "articles", "dataType": ["Article"], "description": "The articles written by the author"}
            ]
        }
    ]
}

# Fresh delete classes
try:
    client.schema.delete_class('Article')
    client.schema.delete_class('Author')
except Exception as e:
    print(f"Error deleting classes: {str(e)}")

# Create new schema
try:
    client.schema.create(schema)
except Exception as e:
    print(f"Error creating schema: {str(e)}")

data = [
    {
        "class": "Article",
        "properties": {
            "title": "LangChain: OpenAI + S3 Loader",
            "content": "This article discusses the integration of LangChain with OpenAI and S3 Loader...",
            "url": "https://blog.min.io/langchain-openai-s3-loader/"
        }
    },
    {
        "class": "Article",
        "properties": {
            "title": "MinIO Webhook Event Notifications",
            "content": "Exploring the webhook event notification system in MinIO...",
            "url": "https://blog.min.io/minio-webhook-event-notifications/"
        }
    },
    {
        "class": "Article",
        "properties": {
            "title": "MinIO Postgres Event Notifications",
            "content": "An in-depth look at Postgres event notifications in MinIO...",
            "url": "https://blog.min.io/minio-postgres-event-notifications/"
        }
    },
    {
        "class": "Article",
        "properties": {
            "title": "From Docker to Localhost",
            "content": "A guide on transitioning from Docker to localhost environments...",
            "url": "https://blog.min.io/from-docker-to-localhost/"
        }
    }
]

for item in data:
    try:
        client.data_object.create(
            data_object=item["properties"],
            class_name=item["class"]
        )
    except Exception as e:
        print(f"Error indexing data: {str(e)}")

# Fetch and export objects
try:
    query = '{ Get { Article { title content datePublished url } } }'
    result = client.query.raw(query)
    articles = result['data']['Get']['Article']
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    print(f"Exported {len(articles)} articles to {OUTPUT_FILE}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# Create backup
try:
    result = client.backup.create(
        backup_id="backup-id-2",
        backend="s3",
        include_classes=["Article", "Author"],
        wait_for_completion=True,
    )
    print("Backup created successfully.")
except Exception as e:
    print(f"Error creating backup: {str(e)}")


