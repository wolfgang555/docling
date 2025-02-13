from fastapi import FastAPI, UploadFile, HTTPException, File, Request, Body
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Union, List
import aiohttp
import asyncio
import uuid
import os
import base64
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    ImageFormatOption
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
import logging
from datetime import datetime
from asyncio import TimeoutError
import subprocess
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import psutil
import json
from contextlib import asynccontextmanager

# 系统配置
OUTPUT_DIRECTORY = os.getenv('DOCLING_OUTPUT_DIR', 'temp')
# 预留一个核心给系统和其他进程
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  

# 在 main.py 的开头添加配置
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# 添加内存使用限制
MEMORY_LIMIT_PERCENT = 80  # 最大使用80%的系统内存

# 使用 lifespan 替代 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    redis = aioredis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="docling-cache:")
    yield
    # 关闭时执行
    await redis.close()

app = FastAPI(title="Document Conversion Service", lifespan=lifespan)

# 存储转换任务的状态
conversion_tasks: Dict[str, dict] = {}

# 创建进程池
process_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversionRequest(BaseModel):
    source: str
    target_format: str
    is_url: bool = False
    do_ocr: bool = False  # 是否启用 OCR
    table_mode: str = "fast"  # "fast" 或 "accurate"
    max_pages: Optional[int] = None  # 最大页数限制

    @field_validator('target_format')
    def validate_format(cls, v):
        allowed_formats = {'markdown', 'text', 'json'}
        if v.lower() not in allowed_formats:
            raise ValueError(f"Unsupported format. Allowed formats: {', '.join(allowed_formats)}")
        return v.lower()
    
    @field_validator('table_mode')
    def validate_table_mode(cls, v):
        if v not in ['fast', 'accurate']:
            raise ValueError("Table mode must be either 'fast' or 'accurate'")
        return v

class ConversionStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[str] = None
    start_time: Optional[str] = None

class ErrorDetail(BaseModel):
    msg: str
    type: str = "error"
    data: Optional[str] = None

class ProgressTracker:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def update_progress(self, current: int, total: int, message: str):
        progress = int((current / total) * 100) if total > 0 else 0
        conversion_tasks[self.task_id].update({
            "progress": f"{message} ({progress}%)"
        })

@app.post("/convert", response_model=ConversionStatus)
async def convert_document_endpoint(request: ConversionRequest = Body(...)):
    try:
        task_id = str(uuid.uuid4())
        logger.info(f"Received conversion request: {request.model_dump()}")
        
        conversion_tasks[task_id] = {
            "status": "pending",
            "start_time": datetime.now().isoformat(),
            "progress": "Initializing"
        }
        
        # 立即返回任务ID
        response = ConversionStatus(
            task_id=task_id, 
            status="pending",
            progress="Task queued. This might take a few minutes."
        )
        
        # 在后台启动转换任务
        asyncio.create_task(process_conversion(task_id, request))
        
        return response
        
    except Exception as e:
        logger.error(f"Error in convert endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}", response_model=ConversionStatus)
async def get_conversion_status(task_id: str):
    if task_id not in conversion_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = conversion_tasks[task_id]
    return ConversionStatus(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error"),
        progress=task.get("progress"),
        start_time=task.get("start_time")
    )

@app.get("/result/{task_id}")
@cache(expire=3600)  # 缓存1小时
async def get_conversion_result(task_id: str):
    if task_id not in conversion_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = conversion_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Task not completed. Current status: {task['status']}, Progress: {task.get('progress')}"
        )
    
    result_path = task.get("result")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return PlainTextResponse(content)
    except Exception as e:
        logger.error(f"Failed to read result file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read result file")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        return JSONResponse(
            status_code=422,
            content={"detail": {
                "msg": "Validation error",
                "type": "validation_error",
                "errors": [{"loc": err["loc"], "msg": err["msg"]} for err in exc.errors()]
            }}
        )
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"detail": {
                "msg": "Validation error occurred",
                "type": "validation_error"
            }}
        )

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, UnicodeDecodeError):
        error_detail = ErrorDetail(
            msg="Invalid file encoding or binary data",
            type="encoding_error"
        )
    elif isinstance(exc, HTTPException):
        error_detail = ErrorDetail(
            msg=str(exc.detail),
            type="http_error"
        )
    elif isinstance(exc, RequestValidationError):
        # 让验证错误由专门的处理器处理
        raise exc
    else:
        error_detail = ErrorDetail(
            msg=str(exc),
            type="error"
        )
    
    return JSONResponse(
        status_code=getattr(exc, 'status_code', 500),
        content={"detail": error_detail.dict(exclude_none=True)}
    )

@app.post("/upload", response_model=ConversionStatus)
async def upload_file(file: UploadFile = File(...), target_format: str = "markdown"):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
            
        # 检查文件扩展名
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}  # 添加支持的文件类型
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # 创建临时文件存储目录
        temp_dir = os.path.abspath(OUTPUT_DIRECTORY)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 生成任务ID和临时文件路径
        task_id = str(uuid.uuid4())
        temp_file_path = os.path.join(temp_dir, f"{task_id}_input{file_ext}")
        
        try:
            # 以二进制模式读取和保存文件
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file")
                
            with open(temp_file_path, "wb") as buffer:
                buffer.write(content)
                
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # 创建转换请求
        request = ConversionRequest(
            source=temp_file_path,
            target_format=target_format,  # 使用请求参数中的格式
            is_url=False
        )
        
        # 启动异步转换任务
        conversion_tasks[task_id] = {"status": "pending"}
        asyncio.create_task(process_conversion(task_id, request))
        
        return ConversionStatus(task_id=task_id, status="pending")
        
    except HTTPException as he:
        raise he
    except Exception as e:
        if isinstance(e, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="Invalid file encoding")
        raise HTTPException(status_code=500, detail=str(e))

def check_system_resources():
    memory = psutil.virtual_memory()
    if memory.percent > MEMORY_LIMIT_PERCENT:
        raise Exception(f"System memory usage too high: {memory.percent}%")

def convert_and_export_document(source_path: str, target_format: str, request: ConversionRequest):
    """在同一个进程中完成转换和导出，支持更多配置选项"""
    # 配置 PDF 处理选项
    pipeline_options = PdfPipelineOptions(
        do_ocr=request.do_ocr,
        do_table_structure=True,
    )
    
    # 设置表格识别模式
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if request.table_mode == 'accurate' 
        else TableFormerMode.FAST
    )

    # 创建转换器，配置支持的格式
    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            ),
        }
    )

    # 执行转换，只在指定了页数限制时传入参数
    convert_args = {'source': source_path}
    if request.max_pages is not None:
        convert_args['max_num_pages'] = request.max_pages
    
    result = converter.convert(**convert_args)
    
    # 导出结果
    if target_format == 'markdown':
        return result.document.export_to_markdown()
    elif target_format == 'text':
        return result.document.export_to_text()
    elif target_format == 'json':
        # 使用正确的导出方法
        return json.dumps(result.document.export_to_dict(), ensure_ascii=False)
    else:
        raise Exception(f"Unsupported target format: {target_format}")

async def process_conversion(task_id: str, request: ConversionRequest):
    temp_file_path = None
    try:
        # 检查系统资源
        check_system_resources()
        
        logger.info(f"Starting conversion for task {task_id}")
        logger.info(f"Request details: source={request.source}, format={request.target_format}, is_url={request.is_url}")
        
        # 使用系统配置的输出目录
        output_dir = os.path.abspath(OUTPUT_DIRECTORY)
        if not output_dir.startswith(os.path.abspath(os.curdir)):
            raise Exception("Invalid output directory path")
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果不是URL，检查是否是任务ID，并获取对应的文件路径
        if not request.is_url:
            if len(request.source) == 36:  # UUID长度
                # 假设source是任务ID，构建完整的文件路径
                source_file = os.path.join(output_dir, f"{request.source}_input.pdf")
                if not os.path.exists(source_file):
                    # 尝试其他可能的扩展名
                    for ext in ['.docx', '.doc', '.txt']:
                        alt_file = os.path.join(output_dir, f"{request.source}_input{ext}")
                        if os.path.exists(alt_file):
                            source_file = alt_file
                            break
                if not os.path.exists(source_file):
                    raise Exception(f"Source file not found for task ID: {request.source}")
                request.source = source_file
            
        # 检查源文件是否存在
        if not request.is_url and not os.path.exists(request.source):
            raise Exception(f"Source file not found: {request.source}")
            
        logger.info(f"Source file exists: {os.path.exists(request.source)}")

        # 如果是URL，先下载文件
        if request.is_url:
            temp_file_path = os.path.join(output_dir, f"{task_id}_download")
            async with aiohttp.ClientSession() as session:
                async with session.get(request.source) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download file: HTTP {response.status}")
                    content = await response.read()
                    with open(temp_file_path, "wb") as f:
                        f.write(content)
            source_path = temp_file_path
        else:
            source_path = request.source

        logger.info(f"Using source path: {source_path}")

        # 设置输出路径
        output_path = os.path.join(output_dir, f"{task_id}_output.{request.target_format}")
        logger.info(f"Output path set to: {output_path}")

        # 更新任务状态为处理中
        conversion_tasks[task_id].update({
            "status": "processing",
            "progress": "Processing document"
        })
        logger.info("Starting document conversion")
        
        # 执行文档转换和导出
        loop = asyncio.get_running_loop()
        conversion_tasks[task_id]["progress"] = f"Converting document using {MAX_WORKERS} CPU cores..."
        
        try:
            # 在同一个进程中完成转换和导出
            content = await loop.run_in_executor(
                process_pool, 
                convert_and_export_document, 
                source_path, 
                request.target_format,
                request
            )
            
            conversion_tasks[task_id]["progress"] = "Writing results to file"
            logger.info("Document conversion completed")
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Successfully exported to {request.target_format}")
            
        except Exception as e:
            logger.error(f"Conversion/Export error: {str(e)}")
            raise

        # 更新任务状态为完成
        conversion_tasks[task_id].update({
            "status": "completed",
            "result": output_path,
            "progress": "Conversion completed successfully"
        })
        logger.info("Task completed successfully")

    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        conversion_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "progress": f"Failed: {str(e)}"
        })
    finally:
        # 清理临时文件
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temp file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up temp file: {str(e)}")

@app.post("/batch", response_model=List[ConversionStatus])
async def batch_convert(requests: List[ConversionRequest]):
    # 限制批处理大小
    MAX_BATCH_SIZE = 5
    if len(requests) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size exceeds limit. Maximum allowed: {MAX_BATCH_SIZE}"
        )
    
    # 检查系统资源
    check_system_resources()
    
    task_ids = []
    for request in requests:
        task_id = str(uuid.uuid4())
        conversion_tasks[task_id] = {
            "status": "pending",
            "start_time": datetime.now().isoformat(),
            "progress": "Queued"
        }
        asyncio.create_task(process_conversion(task_id, request))
        task_ids.append(task_id)
    
    return [ConversionStatus(
        task_id=task_id,
        status="pending",
        progress="Task queued"
    ) for task_id in task_ids]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4567)