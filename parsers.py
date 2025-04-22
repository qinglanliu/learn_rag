import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean_extra_whitespace
# 用于潜在的HTML到Markdown转换
try:
    import markdownify
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False
    
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义unstructured的默认分区策略
# 参见: https://unstructured-io.github.io/unstructured/using/strategies.html
DEFAULT_STRATEGY = "auto" # 让unstructured根据文件类型和设置自动选择
PDF_HI_RES_STRATEGY = "hi_res" # 适用于具有布局分析的复杂PDF
OCR_ONLY_STRATEGY = "ocr_only" # 强制对基于图像的文档使用OCR

def parse_file_to_elements(
    file_path: str,
    strategy: str = DEFAULT_STRATEGY,
    infer_table_structure: bool = True,
    extract_images_in_pdf: bool = False, # 如果要提取图像*文本*，设置为True
    **partition_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    使用unstructured的partition函数解析文件并将元素以字典形式返回，
    包含page_content和metadata。

    参数:
        file_path: 要解析的文件路径。
        strategy: 分区策略('auto', 'hi_res', 'ocr_only', 'fast'等)。
        infer_table_structure: 是否推断表格结构（如果为True，会在元数据中添加HTML表示）。
        extract_images_in_pdf: 是否在PDF中的图像上运行OCR（需要tesseract等依赖项）。
        **partition_kwargs: 传递给unstructured.partition的附加关键字参数。

    返回:
        字典列表，每个字典表示一个解析的元素，符合{"page_content": str, "metadata": dict}结构。
    """
    if not os.path.exists(file_path):
        logging.error(f"找不到要解析的文件: {file_path}")
        return []

    logging.info(f"使用'{strategy}'策略解析文件'{file_path}'")

    try:
        # 合并默认和传递的kwargs，优先使用传递的
        combined_kwargs = {
            "strategy": strategy,
            "infer_table_structure": infer_table_structure,
            "extract_images_in_pdf": extract_images_in_pdf,
             # 根据需要添加其他有用的默认值或覆盖
            # "languages": ["eng", "chi_sim"], # 示例：指定OCR的语言
             "skip_infer_table_types": [], # 默认处理所有表格
        }
        combined_kwargs.update(partition_kwargs) # 用户提供的kwargs优先
        
        elements = partition(filename=file_path, **combined_kwargs)

        parsed_docs = []
        for i, element in enumerate(elements):
            metadata = element.metadata.to_dict()
            metadata['category'] = element.category # 确保category在元数据字典中
            metadata['element_id'] = element.id # 捕获元素ID
            
            content = element.text
            content = clean_extra_whitespace(content) # 基本清理

            # 处理表格：可选将HTML转换为Markdown
            if element.category == "Table" and infer_table_structure:
                table_html = metadata.get('text_as_html')
                if table_html and HAS_MARKDOWNIFY:
                    try:
                        # 将HTML表格转换为Markdown以获得更好的文本表示
                        content = markdownify.markdownify(table_html, heading_style="ATX")
                        logging.debug(f"已将元素{i}的表格HTML转换为Markdown")
                        # 如果需要稍后使用，可以在元数据中保留HTML
                        # metadata['table_html'] = table_html 
                    except Exception as md_err:
                        logging.warning(f"无法将元素{i}的表格HTML转换为Markdown: {md_err}。使用纯文本。")
                elif table_html:
                    logging.warning("未找到markdownify库。表格内容将为纯文本。使用pip install markdownify安装。")
                    # 如果markdownify不可用，则回退到纯文本
                # 否则：content保持为element.text
                    
            # 处理图像：如果extract_images_in_pdf=True，内容通常是OCR文本
            if element.category == "Image":
                 # 内容已经是OCR文本（如果启用了提取）
                 # 你可以在这里添加逻辑，使用图像路径/字节来调用多模态模型进行图像*描述*
                 # （取决于partition参数，如果在元数据中可用）
                 logging.debug(f"找到图像元素{i}。内容为OCR文本: '{content[:50]}...'")

            parsed_doc = {
                "page_content": content,
                "metadata": metadata
            }
            parsed_docs.append(parsed_doc)

        logging.info(f"成功从'{file_path}'解析了{len(elements)}个元素。")
        return parsed_docs

    except Exception as e:
        logging.error(f"使用'{strategy}'策略解析文件{file_path}时出错: {e}", exc_info=True)
        return []

def elements_to_langchain_docs(elements_data: List[Dict[str, Any]]) -> List[Document]:
    """将解析的元素字典转换为LangChain Document对象。"""
    return [Document(page_content=el['page_content'], metadata=el['metadata']) for el in elements_data]


if __name__ == '__main__':
    # --- 示例用法 ---
    # 确保你有一个测试PDF，例如来自unstructured示例的
    # 或者使用之前示例中的PDF之一（如果可用）。
    # 假设'90-文档-Data/复杂PDF/billionaires_page-1-5.pdf'存在
    example_pdf_path = "90-文档-Data/复杂PDF/billionaires_page-1-5.pdf"
    example_md_path = "temp_load_data/sample.md" # 使用loaders.py创建的
    
    if not os.path.exists(example_pdf_path):
        print(f"警告: 示例PDF在{example_pdf_path}未找到。跳过PDF解析示例。")
    else:
        print(f"--- 解析PDF: {example_pdf_path} (hi_res策略) ---")
        # 使用hi_res以获得潜在更好的PDF布局分析
        pdf_elements = parse_file_to_elements(
            example_pdf_path,
            strategy=PDF_HI_RES_STRATEGY, 
            infer_table_structure=True,
            extract_images_in_pdf=True # 尝试对PDF中的图像进行OCR
        )
        
        if pdf_elements:
            print(f"解析了{len(pdf_elements)}个元素。")
            # 打印前几个元素的信息以及找到的任何表格
            tables_found = 0
            for i, element in enumerate(pdf_elements):
                if i < 3:
                    print(f"\n元素 {i}: 类别: {element['metadata'].get('category')}")
                    print(f"  内容: {element['page_content'][:150]}...")
                    print(f"  元数据键: {list(element['metadata'].keys())}")
                if element['metadata'].get('category') == "Table":
                    tables_found += 1
                    if tables_found == 1: # 打印第一个表格的详细信息
                        print(f"\n元素 {i} (第一个表格): 类别: Table")
                        print(f"  页码: {element['metadata'].get('page_number')}")
                        print(f"  内容 (Markdown/文本):\n{element['page_content']}")
                        print(f"  元数据键: {list(element['metadata'].keys())}")
            print(f"\n找到的表格总数 (具有推断结构): {tables_found}")
            
            # 如果需要，转换为LangChain Documents
            # langchain_docs = elements_to_langchain_docs(pdf_elements)
            # print(f"\n转换为 {len(langchain_docs)} 个LangChain Documents。")
        else:
            print("解析失败或未产生元素。")
        print("-" * 50)

    if not os.path.exists(example_md_path):
         print(f"警告: 示例Markdown在{example_md_path}未找到。跳过MD解析示例。")
    else:
        print(f"\n--- 解析Markdown: {example_md_path} (auto策略) ---")
        md_elements = parse_file_to_elements(example_md_path, strategy="auto") # auto通常对MD效果很好
        if md_elements:
            print(f"解析了{len(md_elements)}个元素。")
            for i, element in enumerate(md_elements):
                 print(f"\n元素 {i}: 类别: {element['metadata'].get('category')}")
                 print(f"  内容: {element['page_content']}")
                 print(f"  元数据: {element['metadata']}")
        else:
            print("解析失败或未产生元素。")
        print("-" * 50) 