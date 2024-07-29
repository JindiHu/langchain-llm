import os

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Define the proxy URL with the correct scheme
proxy_url = "http://proxy.ci.mcf.sh:3128"

# Define the client with proxy settings
http_client = httpx.Client(proxies={
    "http://": proxy_url,
    "https://": proxy_url,
})

client = AzureOpenAI(
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    http_client=http_client
)

response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], # model = "deployment_name".
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure AI services support this too?"}
    ]
)

print(response.choices[0].message.content)