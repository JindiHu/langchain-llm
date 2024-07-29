import os

import httpx
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from callback import AgentCallbackHandler

load_dotenv()

# Define the proxy URL with the correct scheme
proxy_url = "http://proxy.ci.mcf.sh:3128"

# Define the client with proxy settings
http_client = httpx.Client(proxies={
    "http://": proxy_url,
    "https://": proxy_url,
})

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    http_client=http_client,
    request_timeout=10,
    callbacks=[AgentCallbackHandler()],
    verbose=True
)

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

res = model.invoke(messages)

print(res)