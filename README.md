

# Learn RAG (Retrieval-Augmented Generation)

这个项目提供了一套增强型数据处理工具，专注于改进文件加载、分块和解析的功能，为构建高效的检索增强生成(RAG)系统提供基础支持。

## 项目概述

Learn RAG 项目旨在解决传统文档处理流程中的局限性，通过提供更多的加载器和分块策略，显著提高代码的健壮性和处理效率。无论是处理大型文档集合还是复杂格式的文件，本项目都能提供稳定可靠的解决方案。

## 主要功能

### 1. 增强文件加载 (Enhanced File Loading)
- 支持多种文件格式 (PDF, DOCX, TXT, CSV, JSON等)
- 提供灵活的加载策略，适应不同的数据源
- 优化大文件处理性能

### 2. 智能文件分块 (Intelligent Chunking)
- 多种分块策略 (固定大小、语义分块、递归分块等)
- 可配置的重叠区域设置
- 针对不同文档类型的优化分块方案

### 3. 高级文件解析 (Advanced Parsing)
- 结构化数据提取
- 元数据保留与处理
- 多语言文档支持

### 4. 系统集成能力
- 与主流向量数据库无缝集成
- 支持流行的LLM框架
- 可扩展的API设计

## 快速开始

### 安装

```bash
git clone https://github.com/qinglanliu/learn_rag.git
cd learn_rag
pip install -r requirements.txt
```

### 基本用法

```python
from learn_rag import FileLoader, FileChunker, FileParser

# 加载文件
loader = FileLoader()
document = loader.load("path/to/your/document.pdf")

# 文档分块
chunker = FileChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(document)

# 解析文档
parser = FileParser()
parsed_content = parser.parse(chunks)

# 使用解析后的内容进行后续处理
# ...
```

## 高级用例

### 自定义分块策略

```python
from learn_rag import FileChunker, SemanticChunkingStrategy

# 创建语义分块策略
semantic_strategy = SemanticChunkingStrategy(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 使用自定义策略进行分块
chunker = FileChunker(strategy=semantic_strategy)
semantic_chunks = chunker.chunk(document)
```

### 批量处理文件

```python
from learn_rag import BatchProcessor

# 创建批处理器
processor = BatchProcessor()

# 处理整个目录
results = processor.process_directory(
    "path/to/documents/",
    file_types=["pdf", "docx"],
    chunk_size=800
)
```

## 项目结构

```
learn_rag/
├── __init__.py
├── loaders/         # 文件加载器
├── chunkers/        # 分块策略
├── parsers/         # 解析器
├── utils/           # 工具函数
├── models/          # 数据模型
└── examples/        # 使用示例
```

## 贡献指南

欢迎对本项目做出贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

## 许可证

[MIT License](LICENSE)

## 联系方式

- 项目维护者: Qinglan Liu
- GitHub: [https://github.com/qinglanliu](https://github.com/qinglanliu)

---

如需更多信息，请查看我们的[文档](docs/README.md)或[示例](examples/)。
