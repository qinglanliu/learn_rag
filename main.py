import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document

# 导入自定义模块
from loaders import load_single_file, load_directory
from chunkers import chunk_documents
from parsers import parse_file_to_elements, elements_to_langchain_docs, PDF_HI_RES_STRATEGY

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_save_to_json(
    input_path: str,
    output_path: str,
    process_type: str = "load_and_chunk",
    chunk_strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    is_directory: bool = False,
    recursive: bool = True,
    additional_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    统一的数据处理和保存函数：可以加载文件，分块文档，或直接解析结构化文档，
    然后将结果保存为JSON格式。

    参数:
        input_path: 输入文件或目录的路径
        output_path: 输出JSON文件的路径
        process_type: 处理类型，可选择 "load_only", "load_and_chunk", "parse"
        chunk_strategy: 如果进行分块，要使用的分块策略
        chunk_size: 块的大小（取决于策略）
        chunk_overlap: 块之间的重叠量
        is_directory: 输入路径是否为目录
        recursive: 如果输入是目录，是否递归处理
        additional_params: 传递给加载器、分块器或解析器的额外参数
    """
    additional_params = additional_params or {}
    documents = []
    
    try:
        # 第1步：根据指定的处理类型加载或解析文件
        if process_type in ["load_only", "load_and_chunk"]:
            # 情况1：使用loaders模块加载文件
            if is_directory:
                logging.info(f"从目录 '{input_path}' 加载文档...")
                documents = load_directory(
                    dir_path=input_path,
                    recursive=recursive,
                    **additional_params.get("loader_params", {})
                )
            else:
                logging.info(f"加载单个文件 '{input_path}'...")
                documents = load_single_file(
                    file_path=input_path,
                    **additional_params.get("loader_params", {})
                )
            
            # 第2步：如果选择了load_and_chunk，应用分块
            if process_type == "load_and_chunk" and documents:
                logging.info(f"使用'{chunk_strategy}'策略对{len(documents)}个文档进行分块...")
                documents = chunk_documents(
                    documents=documents,
                    strategy=chunk_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    splitter_kwargs=additional_params.get("chunker_params", {})
                )
        
        elif process_type == "parse":
            # 情况2：使用parsers模块直接解析文件
            logging.info(f"直接解析文件 '{input_path}'...")
            elements = parse_file_to_elements(
                file_path=input_path,
                **additional_params.get("parser_params", {})
            )
            
            # 将解析的元素转换为Document对象
            if elements:
                documents = elements_to_langchain_docs(elements)
        
        else:
            logging.error(f"未知的处理类型: {process_type}")
            return
        
        # 第3步：将结果保存为JSON
        if documents:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 将Document对象转换为可序列化的字典
            serializable_docs = []
            for doc in documents:
                # 确保元数据是可序列化的（转换复杂对象，如果有的话）
                serializable_metadata = {
                    k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v 
                    for k, v in doc.metadata.items()
                }
                serializable_docs.append({
                    "page_content": doc.page_content,
                    "metadata": serializable_metadata
                })
            
            # 保存为JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
            
            logging.info(f"成功将{len(documents)}个文档保存到 '{output_path}'")
        else:
            logging.warning(f"没有文档可保存。检查输入路径和处理参数。")
    
    except Exception as e:
        logging.error(f"处理文档时出错: {e}", exc_info=True)


if __name__ == "__main__":
    # 确保测试目录存在
    os.makedirs("temp_output", exist_ok=True)
    
    # 测试场景1：加载单文件并分块
    # 先创建一个测试文件
    os.makedirs("temp_load_data", exist_ok=True)
    with open("temp_load_data/large_text.txt", "w", encoding="utf-8") as f:
        f.write("这是一个大型文本文件的示例。" * 50 + "\n")
        f.write("它包含多个段落。\n" * 20)
        f.write("这些句子将被分成较小的块。" * 30)
    
    # 加载并分块
    process_and_save_to_json(
        input_path="temp_load_data/large_text.txt",
        output_path="temp_output/large_text_chunked.json",
        process_type="load_and_chunk",
        chunk_strategy="recursive",
        chunk_size=200,
        chunk_overlap=50
    )
    
    # 测试场景2：加载目录中的文件但不分块
    process_and_save_to_json(
        input_path="temp_load_data",
        output_path="temp_output/directory_loaded.json",
        process_type="load_only",
        is_directory=True,
        recursive=True,
        additional_params={"loader_params": {"glob_pattern": "**/*.txt"}}
    )
    
    # 测试场景3：直接解析PDF（如果存在）
    # 尝试解析前面示例中的PDF
    pdf_path = "90-文档-Data/复杂PDF/billionaires_page-1-5.pdf"
    if os.path.exists(pdf_path):
        process_and_save_to_json(
            input_path=pdf_path,
            output_path="temp_output/parsed_pdf.json",
            process_type="parse",
            additional_params={
                "parser_params": {
                    "strategy": PDF_HI_RES_STRATEGY,
                    "infer_table_structure": True,
                    "extract_images_in_pdf": True
                }
            }
        )
    else:
        print(f"示例PDF不存在: {pdf_path}。跳过场景3。")
        
    print("\n所有处理完成。结果保存在 'temp_output' 目录。")
    print("你可以检查以下文件：")
    print("1. temp_output/large_text_chunked.json - 单个文本文件加载并分块")
    print("2. temp_output/directory_loaded.json - 从目录加载的文件")
    if os.path.exists(pdf_path):
        print("3. temp_output/parsed_pdf.json - 直接解析的PDF文件") 