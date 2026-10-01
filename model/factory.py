from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod     #抽象方法
    def generator(self)->Optional[Embeddings | BaseChatModel]:  #定义抽象方法
        pass


class ChatModelFactory(BaseModelFactory):
    #继承BaseModelFactory,集成父类定义了抽象方法,内容是pass,子类必须重写,此处表现为实现generator类
    def generator(self) ->Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) ->Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model=ChatModelFactory().generator()
embed_model=EmbeddingsFactory().generator()


