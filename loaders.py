import os
from typing import List, Dict, Any, Optional, Type
from langchain_core.documents import Document
# 从langchain_core导入BaseLoader
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
    WebBaseLoader,
    # 从langchain_community正确导入Unstructured相关类
    UnstructuredMarkdownLoader,
    UnstructuredImageLoader,
    UnstructuredPowerPointLoader,
    UnstructuredEmailLoader
)
# 从langchain_unstructured正确导入UnstructuredLoader
from langchain_unstructured import UnstructuredLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 文件扩展名到默认加载器的映射
DEFAULT_LOADERS: Dict[str, Type[BaseLoader]] = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader, # 基本PDF加载器，对于复杂PDF考虑使用UnstructuredLoader
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".eml": UnstructuredEmailLoader,
    ".py": TextLoader, # 使用TextLoader处理Python文件
    # 根据需要添加更多映射
}

# 已知可以从unstructured策略中受益的加载器
UNSTRUCTURED_LOADERS = {UnstructuredLoader, UnstructuredMarkdownLoader, UnstructuredPowerPointLoader, UnstructuredEmailLoader}

def load_single_file(
    file_path: str,
    loader_cls: Optional[Type[BaseLoader]] = None,
    loader_kwargs: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    使用指定的LangChain加载器或从扩展名推断的加载器加载单个文件。

    参数:
        file_path: 文件路径。
        loader_cls: 要使用的特定LangChain加载器类。如果为None，则从扩展名推断。
        loader_kwargs: 可选的关键字参数，传递给加载器构造函数。

    返回:
        Document对象列表。
    """
    if not os.path.exists(file_path):
        logging.error(f"文件未找到: {file_path}")
        return []
        
    resolved_loader_cls = loader_cls
    if resolved_loader_cls is None:
        _, ext = os.path.splitext(file_path)
        resolved_loader_cls = DEFAULT_LOADERS.get(ext.lower())
        if resolved_loader_cls is None:
            logging.warning(f"没有'{ext}'扩展名的默认加载器。尝试使用UnstructuredLoader。")
            resolved_loader_cls = UnstructuredLoader # 回退到Unstructured

    loader_kwargs = loader_kwargs or {}
    
    # 一致地处理路径参数
    path_arg_name = "file_path"
    if resolved_loader_cls is UnstructuredLoader:
         # UnstructuredLoader直接接受file_path
         pass
    elif resolved_loader_cls is WebBaseLoader:
         path_arg_name = "web_paths"
         loader_kwargs[path_arg_name] = [file_path] # WebBaseLoader需要一个列表
         file_path = None # 如果在kwargs中处理，则避免直接传递file_path
    # 根据需要添加其他加载器特定的路径参数名称

    try:
        # 如果是预期的参数，则直接传递file_path，否则依赖loader_kwargs
        if file_path and path_arg_name=="file_path":
             loader = resolved_loader_cls(file_path, **loader_kwargs)
        else:
             loader = resolved_loader_cls(**loader_kwargs)
             
        logging.info(f"使用{resolved_loader_cls.__name__}加载文件'{file_path or loader_kwargs.get(path_arg_name)}'")
        documents = loader.load()
        # 确保元数据始终有source字段
        for doc in documents:
            if 'source' not in doc.metadata:
                 doc.metadata['source'] = file_path or loader_kwargs.get(path_arg_name)
                 if isinstance(doc.metadata['source'], list): # 处理WebBaseLoader情况
                     doc.metadata['source'] = doc.metadata['source'][0]

        logging.info(f"成功加载了{len(documents)}个文档。")
        return documents
    except Exception as e:
        logging.error(f"使用{resolved_loader_cls.__name__}加载文件{file_path or loader_kwargs.get(path_arg_name)}时出错: {e}", exc_info=True)
        return []


def load_directory(
    dir_path: str,
    glob_pattern: str = "**/[!.]*", # 默认：所有非隐藏文件
    loader_cls: Optional[Type[BaseLoader]] = None, # 如果为None，则按文件自动检测
    loader_kwargs: Optional[Dict[str, Any]] = None,
    recursive: bool = True,
    use_multithreading: bool = True,
    show_progress: bool = True,
    silent_errors: bool = True # 推荐避免一个错误文件导致整个过程停止
) -> List[Document]:
    """
    使用DirectoryLoader加载目录中的文档。

    参数:
        dir_path: 目录路径。
        glob_pattern: 匹配文件的glob模式（例如，"**/*.txt"，"*.pdf"）。
        loader_cls: 用于*所有文件*的特定LangChain加载器类。
                   如果为None，DirectoryLoader会尝试基于文件扩展名推断，
                   回退到UnstructuredLoader。
        loader_kwargs: 可选的关键字参数，传递给加载器构造函数
                       （仅在指定loader_cls时应用）。
        recursive: 是否搜索子目录。
        use_multithreading: 是否使用多线程进行加载。
        show_progress: 是否显示进度条。
        silent_errors: 如果为True，加载单个文件的错误会被记录但不会停止进程。

    返回:
        来自所有成功加载文件的Document对象列表。
    """
    if not os.path.isdir(dir_path):
        logging.error(f"目录未找到: {dir_path}")
        return []

    loader_kwargs = loader_kwargs or {}

    try:
        # 默认使用TextLoader作为回退，而不是让DirectoryLoader决定
        # 这是因为它可能会尝试使用UnstructuredLoader，而这需要unstructured包
        fallback_loader = loader_cls or TextLoader
        
        # 自定义加载器函数，手动检测文件类型并使用合适的加载器
        def _load_file(filepath: str) -> List[Document]:
            try:
                return load_single_file(filepath, loader_kwargs=loader_kwargs)
            except Exception as e:
                if silent_errors:
                    logging.warning(f"加载文件 {filepath} 时出错: {e}")
                    return []
                else:
                    raise
                    
        # 手动查找文件并加载它们
        all_docs = []
        import glob
        pattern = os.path.join(dir_path, glob_pattern)
        files = glob.glob(pattern, recursive=recursive)
        
        # 显示进度信息
        if show_progress:
            logging.info(f"找到 {len(files)} 个文件...")
            
        # 处理文件
        for file_path in files:
            if os.path.isfile(file_path):
                try:
                    docs = load_single_file(file_path, loader_kwargs=loader_kwargs)
                    all_docs.extend(docs)
                except Exception as e:
                    if silent_errors:
                        logging.warning(f"加载文件 {file_path} 时出错: {e}")
                    else:
                        raise
        
        logging.info(f"从目录成功加载了{len(all_docs)}个文档。")
        return all_docs
    except Exception as e:
        logging.error(f"加载目录{dir_path}时出错: {e}", exc_info=True)
        return []

def load_web_page(
    url: str,
    bs_kwargs: Optional[Dict[str, Any]] = None,
    bs_get_text_kwargs: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    使用WebBaseLoader加载网页内容。

    参数:
        url: 网页的URL。
        bs_kwargs: BeautifulSoup实例化的关键字参数（例如，parse_only）。
        bs_get_text_kwargs: BeautifulSoup的get_text方法的关键字参数（例如，separator，strip）。

    返回:
        包含网页的单个Document对象的列表。
    """
    loader_kwargs = {}
    if bs_kwargs:
        loader_kwargs['bs_kwargs'] = bs_kwargs
    if bs_get_text_kwargs:
        loader_kwargs['bs_get_text_kwargs'] = bs_get_text_kwargs
        
    # WebBaseLoader需要'web_paths'作为列表
    loader_kwargs_for_single = {'web_paths': [url], **loader_kwargs}

    return load_single_file(file_path=None, loader_cls=WebBaseLoader, loader_kwargs=loader_kwargs_for_single)


if __name__ == '__main__':
    # --- 示例用法 ---
    print("--- 加载单个文件 ---")
    
    # 创建用于测试的临时文件
    os.makedirs("temp_load_data", exist_ok=True)
    with open("temp_load_data/sample.txt", "w") as f:
        f.write("这是一个示例文本文件。\n它有多行。")
    with open("temp_load_data/another.txt", "w") as f:
        f.write("另一个文本文件内容。")
    # 创建一个临时markdown文件
    with open("temp_load_data/sample.md", "w") as f:
        f.write("# 示例Markdown\n\n这是第一段。\n\n## 子部分\n\n- 项目1\n- 项目2")
        
    # 示例1: 加载单个TXT文件（推断加载器）
    txt_docs = load_single_file("temp_load_data/sample.txt")
    if txt_docs:
        print(f"加载了sample.txt ({len(txt_docs)}个文档):")
        print(txt_docs[0].page_content)
        print(txt_docs[0].metadata)
        print("-" * 20)

    # 示例2: 使用UnstructuredMarkdownLoader显式加载单个MD文件
    md_docs = load_single_file(
        "temp_load_data/sample.md",
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={"mode": "elements"} # 传递特定参数
    )
    if md_docs:
        print(f"显式加载了sample.md ({len(md_docs)}个文档):")
        for i, doc in enumerate(md_docs):
            print(f"  元素 {i}: {doc.page_content[:50]}... (类别: {doc.metadata.get('category')})")
        print("-" * 20)

    # 示例3: 加载不存在的文件
    print("尝试加载non_existent_file.txt:")
    load_single_file("non_existent_file.txt")
    print("-" * 20)

    print("\n--- 加载目录 ---")
    # 示例4: 加载目录中的所有文件（自动检测加载器）
    all_docs = load_directory("temp_load_data")
    print(f"从temp_load_data加载了所有文件 ({len(all_docs)}个文档):")
    for doc in all_docs:
        print(f"  来源: {doc.metadata.get('source')}, 内容: {doc.page_content[:30]}...")
    print("-" * 20)
    
    # 示例5: 仅使用TextLoader显式递归加载TXT文件
    txt_only_docs = load_directory(
        "temp_load_data",
        glob_pattern="**/*.txt",
        loader_cls=TextLoader, # 为所有.txt文件使用特定的加载器
        silent_errors=False # 示例：如果一个失败则引发错误
    )
    print(f"仅从temp_load_data加载了.txt文件 ({len(txt_only_docs)}个文档):")
    for doc in txt_only_docs:
         print(f"  来源: {doc.metadata.get('source')}")
    print("-" * 20)
    
    # 示例6: 加载已知的网页（需要互联网）
    # print("\n--- 加载网页 ---")
    # wiki_url = "https://en.wikipedia.org/wiki/Large_language_model"
    # wiki_docs = load_web_page(wiki_url)
    # if wiki_docs:
    #     print(f"加载了 {wiki_url} ({len(wiki_docs)}个文档):")
    #     print(f"内容预览: {wiki_docs[0].page_content[:200]}...")
    #     print(f"元数据: {wiki_docs[0].metadata}")
    # else:
    #      print(f"无法加载 {wiki_url}")
    # print("-" * 20)

    # 清理临时文件
    # import shutil
    # shutil.rmtree("temp_load_data")
    # print("\n清理了temp_load_data目录。") 