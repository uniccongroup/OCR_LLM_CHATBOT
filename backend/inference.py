
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from trt_inference_api import CustomLLM 

#rag files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



class chatbotInference:

    def __init__(self,model_path, model_token_path) -> None:

        self.model_path = model_path
        self.model_token_path = model_token_path
        self.retriever = None
        self.rag_chain = None

        #load embedding 
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)

        # load model
        self.llm = CustomLLM(n=5,tokenizer_dir = self.model_token_path, engine_dir = self.model_path)


        # Prompt
        self.template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        self.prompt = ChatPromptTemplate.from_template(self.template)


    def createRetriever(self,file_path):

        text = ""
        text = file_path
        """with open(file_path,'r') as file:
            text = file.read()"""

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        documents = text_splitter.create_documents([text])

        # Embed
        vectorstore = Chroma.from_documents(documents=documents, embedding=self.embedding)
        self.retriever = vectorstore.as_retriever()
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return  self.rag_chain



