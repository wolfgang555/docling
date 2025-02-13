from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import aiohttp
import asyncio
import uuid
import os
from docling import convert_document

# 系统配置
OUTPUT_DIRECTORY = os.getenv('DOCLING_OUTPUT_DIR', 'temp')

app = FastAPI(title="Document Conversion Service")

# 存储转换任务的状态
conversion_tasks: Dict[str, dict] = {}

class ConversionRequest(BaseModel):
    source: str  # URL或文件上传ID
    target_format: str
    is_url: bool = False

class ConversionStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

@app.post("/convert", response_model=ConversionStatus)
async def convert_document_endpoint(request: ConversionRequest):
    task_id = str(uuid.uuid4())
    conversion_tasks[task_id] = {"status": "pending"}
    
    # 启动异步转换任务
    asyncio.create_task(process_conversion(task_id, request))
    
    return ConversionStatus(task_id=task_id, status="pending")

@app.get("/status/{task_id}", response_model=ConversionStatus)
async def get_conversion_status(task_id: str):
    if task_id not in conversion_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = conversion_tasks[task_id]
    return ConversionStatus(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error")
    )

async def process_conversion(task_id: str, request: ConversionRequest):
    try:
        # 使用系统配置的输出目录
        output_dir = os.path.abspath(OUTPUT_DIRECTORY)
        if not output_dir.startswith(os.path.abspath(os.curdir)):
            raise Exception("Invalid output directory path")
        os.makedirs(output_dir, exist_ok=True)

        # 设置输入输出路径
        input_path = os.path.join(output_dir, f"{task_id}_input")
        output_path = os.path.join(output_dir, f"{task_id}_output.{request.target_format}")

        # 下载URL文件或使用上传的文件
        if request.is_url:
            async with aiohttp.ClientSession() as session:
                async with session.get(request.source) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download file: {response.status}")
                    with open(input_path, "wb") as f:
                        f.write(await response.read())
        else:
            # 对于上传的文件，source应该是临时文件路径
            input_path = request.source

        # 更新任务状态为处理中
        conversion_tasks[task_id]["status"] = "processing"

        # 执行文档转换
        convert_document(input_path, output_path)

        # 更新任务状态为完成
        conversion_tasks[task_id].update({
            "status": "completed",
            "result": output_path
        })

    except Exception as e:
        # 更新任务状态为失败
        conversion_tasks[task_id].update({
            "status": "failed",
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4567)