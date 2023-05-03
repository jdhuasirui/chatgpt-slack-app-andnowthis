import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import re 

#from dotenv import load_dotenv
#load_dotenv()

# Initializes your app with your bot token and socket mode handler
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

#Langchain implementation

brainstorm_template = """You are an large languange model assistant to a journalist trained by OpenAI. 
    You will pitch ideas from different angles of news, feature stories, news analysis or explainers, and opinion pieces.
    You need to adhere to journalistic ethics, and deliver accurate reporting, try your best not to make up things. 
    If it makes sense, you may pay attention to what Millennials and Gen Z care about and can write in a way that is relatable to them and explain why the story matters to them. 
    {history}
    Human: {human_input}
    Assistant:"""

template = """You are an large languange model assistant to a journalist trained by OpenAI.
    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    {history}
    Human: {human_input}
    Assistant:"""


prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=brainstorm_template
)          

# Dictionary mapping user IDs to their LLMChain instances
user_chains = {}

def get_user_chain(user_id):
    if user_id not in user_chains:
        user_chains[user_id] = LLMChain(
            llm=ChatOpenAI(temperature=0), 
            prompt=prompt, 
            verbose=True, 
            memory=ConversationBufferWindowMemory(k=10),
        )
    return user_chains[user_id]

#Message handler for Slack
@app.message(".*")
def message_handler(event, say, logger):
    human_input = re.sub(r'<@U[A-Z0-9]+>', '', event['text'])
    user_id = event['user']
    
    chatgpt_chain = get_user_chain(user_id)
    output = chatgpt_chain.predict(human_input=human_input)
    say(output)

@app.event("app_mention")
def event_test(event, say):
    thread_ts = event['ts']
    human_input = re.sub(r'<@U[A-Z0-9]+>', '', event['text'])
    user_id = event['user']
    
    chatgpt_chain = get_user_chain(user_id)
    output = chatgpt_chain.predict(human_input=human_input)
    say(f"<@{event['user']}> {output}", thread_ts=thread_ts)

@app.command("/brainstorm")
def brainstorm_handler(ack, body, say):
    ack()  # Acknowledge the command request
    human_input = body['text']
    user_id = body['user_id']
    
    chatgpt_chain = get_user_chain(user_id)
    output = chatgpt_chain.predict(human_input=human_input)
    say(f"<@{body['user_id']}> {output}")
    
# Start your app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.start(port=port)
