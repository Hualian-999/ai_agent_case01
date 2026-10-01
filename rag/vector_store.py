from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader,txt_loader,listdir_with_allowed_type,get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:
    def __init__(self):
        self.vector_store=Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):        #获取检索器对象
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件内读取数据文件,转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):      #检查配置的md5文件是否存在
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):  #相对路径转为绝对路径,如果配置文件不存在:
                #创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w",encoding="utf-8").close()
                return False             #md5没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r",encoding="utf-8") as f:   #如果文件存在,则读取内容
                for line in f.readlines():
                    line=line.strip()
                    if line==md5_for_check:
                        return True     #md5处理过

                return False            #md5没处理过

        def save_md5_hex(md5_for_check: str):       #将md5存入绝对路径中
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check+"\n")

        def get_file_documents(read_path: str):     #将文件的内容读取出来,得到document对象
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str]=listdir_with_allowed_type(        #拿到允许处理的文件类型
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            #获取文件的md5
            md5_hex=get_file_md5_hex(path)

            if check_md5_hex(md5_hex):      #如果返回值为True,表示md5文件已经处理过
                logger.info(f"[加载知识库]{path}内容已经存在知识库内,跳过")
                continue

            try:        #如果没有处理过,可用
                documents: list[Document]=get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内无有效文本内容,跳过")
                    continue

                split_document: list[Document]=self.spliter.split_documents(documents)  #分割文本

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后无有效文本内容,跳过")
                    continue

                #将分割好的内容存入向量库
                self.vector_store.add_documents(split_document)

                #记录这个已经处理好的文件md5,避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as e:
                #exc_info为True会记录详细的报错堆栈,如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败:{str(e)}",exc_info=True)
                continue


if __name__ == '__main__':
    vs=VectorStoreService() #创建对象

    vs.load_document()  #加载知识库

    retriever=vs.get_retriever()    #拿到检索器

    res=retriever.invoke("迷路")  #通过检索器检索输入的内容,此处为"迷路"
    for r in res:
        print(r.page_content)
        print("-"*20)