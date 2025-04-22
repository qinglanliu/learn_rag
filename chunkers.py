from typing import List, Dict, Any, Optional, Literal
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
    TextSplitter, # 基类
    SemanticChunker
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings # SemanticChunker的示例
# 注意：SemanticChunker可能在langchain_experimental或直接在langchain_text_splitters中
# 取决于LangChain的版本。

import logging
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义可用的分块策略
ChunkingStrategy = Literal[
    "character",
    "recursive",
    "semantic",
    "code_python",
    "code_javascript",
    "code_markdown"
    # 根据需要添加更多语言/语义策略
]

def get_text_splitter(strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int, **kwargs) -> TextSplitter:
    """
    工厂函数，根据策略创建LangChain TextSplitter。

    参数:
        strategy: 要使用的分块策略。
        chunk_size: 每个块的目标大小。
        chunk_overlap: 连续块之间的重叠部分。
        **kwargs: 特定分割器的额外关键字参数
                  (例如，recursive的separators，code的language，
                   semantic的embeddings)。

    返回:
        LangChain TextSplitter的实例。
    """
    logging.info(f"创建分割器: strategy='{strategy}', size={chunk_size}, overlap={chunk_overlap}")
    
    if strategy == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=kwargs.get("separator", "\n"), # 基本字符分割的默认分隔符
            is_separator_regex=kwargs.get("is_separator_regex", False),
        )
    elif strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=kwargs.get("separators", ["\n\n", "\n", ".", " ", ""]), # 默认分隔符
            keep_separator=kwargs.get("keep_separator", True),
        )
    elif strategy == "semantic":
        embeddings = kwargs.get("embeddings")
        if not embeddings:
            # 如果未指定嵌入模型，则提供默认值
            logging.warning("语义分块未提供嵌入模型，使用OpenAIEmbeddings默认值。")
            embeddings = OpenAIEmbeddings() # 确保设置了OPENAI_API_KEY
            
        breakpoint_threshold_type = kwargs.get("breakpoint_threshold_type", "percentile") # 或 "standard_deviation", "interquartile"
        breakpoint_threshold_amount = kwargs.get("breakpoint_threshold_amount", 0.95) # 值取决于类型
            
        return SemanticChunker( # 此处使用langchain_experimental版本
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
             # SemanticChunker不直接使用chunk_size/overlap，但我们为了一致性而传递
             # 它基于语义相似性中断来确定分割。
        )
    elif strategy.startswith("code_"):
        language_map = {
            "code_python": Language.PYTHON,
            "code_javascript": Language.JS,
            "code_markdown": Language.MARKDOWN,
            # 从langchain_text_splitters.Language添加其他语言
        }
        language = language_map.get(strategy)
        if language:
            return RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=kwargs.get("keep_separator", True),
            )
        else:
            raise ValueError(f"不支持的代码语言策略: {strategy}")
    else:
        raise ValueError(f"未知的分块策略: {strategy}")

def chunk_documents(
    documents: List[Document],
    strategy: ChunkingStrategy = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    splitter_kwargs: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    使用指定的策略对文档列表进行分块。

    参数:
        documents: 要分块的LangChain Document对象列表。
        strategy: 要使用的分块策略（例如，"recursive", "semantic"）。
        chunk_size: 块的目标大小（解释取决于策略）。
        chunk_overlap: 块之间的重叠（解释取决于策略）。
        splitter_kwargs: 传递给分割器工厂的额外参数（例如，separators, embeddings）。

    返回:
        分块后的Document对象列表，保留原始元数据并可能添加块特定的元数据。
    """
    if not documents:
        logging.warning("收到空的文档列表进行分块。")
        return []
        
    splitter_kwargs = splitter_kwargs or {}
    try:
        splitter = get_text_splitter(strategy, chunk_size, chunk_overlap, **splitter_kwargs)
    except ValueError as e:
        logging.error(f"创建分割器失败: {e}")
        return [] # 或者根据所需行为重新引发异常

    all_chunks = []
    for i, doc in enumerate(documents):
        original_metadata = copy.deepcopy(doc.metadata) # 避免修改原始doc元数据
        try:
            if strategy == "semantic":
                # SemanticChunker有不同的接口，直接接收文本
                chunks_data = splitter.create_documents([doc.page_content])
                # 需要手动重新附加语义块的元数据
                for chunk_idx, chunk_doc in enumerate(chunks_data):
                    chunk_doc.metadata = copy.deepcopy(original_metadata) # 从原始开始
                    chunk_doc.metadata["chunk_id"] = f"{original_metadata.get('source', 'doc')}_{i}_semantic_{chunk_idx}"
                    all_chunks.append(chunk_doc)
            else:
                 # 标准分割器直接处理文档
                split_docs = splitter.split_documents([doc]) # 一次处理一个文档以管理元数据
                for chunk_idx, chunk_doc in enumerate(split_docs):
                     # Langchain分割器通常保留元数据，但添加chunk ID
                     chunk_doc.metadata["chunk_id"] = f"{original_metadata.get('source', 'doc')}_{i}_{strategy}_{chunk_idx}"
                     all_chunks.append(chunk_doc)
            logging.info(f"使用'{strategy}'策略分块文档 {i+1} ({original_metadata.get('source', 'N/A')})。")
        except Exception as e:
            logging.error(f"分块来自 {original_metadata.get('source', 'N/A')} 的文档 {i+1} 时出错: {e}", exc_info=True)
            # 可选跳过此文档或重新引发异常
            continue

    logging.info(f"分块的文档总数: {len(documents)}。创建的块总数: {len(all_chunks)}。")
    return all_chunks


if __name__ == '__main__':
    # --- 示例用法 ---
    # 创建一些示例文档
    doc1 = Document(
        page_content="这是第一个文档。它相当短。\n它包含多个句子。和多行内容。",
        metadata={"source": "doc1.txt", "author": "测试"}
    )
    doc2 = Document(
        page_content="""
        def hello_world():
            print("你好，世界！")

        class MyClass:
            def __init__(self, name):
                self.name = name
            
            def greet(self):
                 print(f"你好，{self.name}！")
        """,
        metadata={"source": "code.py", "language": "python"}
    )
    docs_to_chunk = [doc1, doc2]

    print("--- 递归分块 ---")
    recursive_chunks = chunk_documents(
        docs_to_chunk,
        strategy="recursive",
        chunk_size=50,
        chunk_overlap=10
    )
    for chunk in recursive_chunks:
        print(f"  块ID: {chunk.metadata['chunk_id']}, 大小: {len(chunk.page_content)}")
        print(f"  内容: '{chunk.page_content}'")
        print(f"  元数据: {chunk.metadata}")
        print("-")
    print(f"递归块总数: {len(recursive_chunks)}\n")

    print("--- Python代码分块 ---")
    python_chunks = chunk_documents(
        docs_to_chunk, # 将仅有效地分块doc2
        strategy="code_python",
        chunk_size=100,
        chunk_overlap=0
    )
    for chunk in python_chunks:
        print(f"  块ID: {chunk.metadata['chunk_id']}, 大小: {len(chunk.page_content)}")
        print(f"  内容:\n{chunk.page_content}")
        print(f"  元数据: {chunk.metadata}")
        print("-")
    print(f"Python块总数: {len(python_chunks)}\n")

    # --- 语义分块（需要OpenAI API密钥） ---
    # 在运行此部分之前设置OPENAI_API_KEY环境变量
    print("--- 语义分块（需要OpenAI API密钥） ---")
    try:
        semantic_chunks = chunk_documents(
            [doc1], # 仅分块文本文档以简化示例
            strategy="semantic",
            # chunk_size/overlap被SemanticChunker忽略，但我们保留函数签名
            splitter_kwargs={"embeddings": OpenAIEmbeddings()} # 传递嵌入
        )
        for chunk in semantic_chunks:
            print(f"  块ID: {chunk.metadata['chunk_id']}, 大小: {len(chunk.page_content)}")
            print(f"  内容: '{chunk.page_content}'")
            print(f"  元数据: {chunk.metadata}")
            print("-")
        print(f"语义块总数: {len(semantic_chunks)}\n")
    except Exception as e:
         print(f"无法运行语义分块示例: {e}")
         print("(确保已设置OPENAI_API_KEY并安装了依赖项)") 