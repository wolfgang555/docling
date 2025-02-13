import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
import os
import json
from main import app, OUTPUT_DIRECTORY
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient as StarletteTestClient

# 移除全局 client 变量，改用 fixture
@pytest.fixture
def client():
    """返回一个测试客户端实例"""
    client = TestClient(app)
    return client

@pytest.fixture
def sample_pdf():
    """提供一个示例 PDF 文件路径"""
    # 这里应该放一个真实的测试 PDF 文件
    return "tests/data/sample.pdf"

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """设置和清理测试环境"""
    # 设置
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    yield
    
    # 清理
    cleanup_test_files()

@pytest.fixture
async def async_client():
    """异步客户端"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def mock_converter():
    """模拟文档转换器"""
    with patch('main.DocumentConverter') as mock:  # 修改 mock 路径
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test markdown"
        mock_result.document.export_to_text.return_value = "Test text"
        mock_result.document.export_to_dict.return_value = {"content": "Test content"}
        mock.return_value.convert.return_value = mock_result
        yield mock

def test_upload_endpoint(client, sample_pdf, setup_and_cleanup):
    """测试文件上传端点"""
    with open(sample_pdf, 'rb') as f:
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", f, "application/pdf")},
            params={"target_format": "markdown"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"

def test_convert_endpoint(client, mock_converter, setup_and_cleanup):
    """测试转换端点"""
    request_data = {
        "source": "test-uuid",
        "target_format": "json",
        "is_url": False,
        "do_ocr": False,
        "table_mode": "fast"
    }
    
    response = client.post("/convert", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"

@pytest.mark.asyncio
async def test_conversion_process(mock_converter):
    """测试转换过程"""
    from main import process_conversion, ConversionRequest, conversion_tasks
    
    # 创建一个更有效的测试 PDF
    test_file = os.path.join(OUTPUT_DIRECTORY, "test_input.pdf")
    with open(test_file, "wb") as f:
        # 写入一个最小但有效的 PDF 内容
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000015 00000 n \n0000000061 00000 n \n0000000111 00000 n \ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n")
    
    request = ConversionRequest(
        source=test_file,
        target_format="markdown",
        is_url=False
    )
    
    task_id = "test-task-id"
    conversion_tasks[task_id] = {
        "status": "pending",
        "start_time": "2024-02-13T20:48:56.765090",
        "progress": "Initializing"
    }
    
    await process_conversion(task_id, request)
    assert conversion_tasks[task_id]["status"] == "completed"

def test_invalid_format(client):
    """测试无效格式"""
    request_data = {
        "source": "test-uuid",
        "target_format": "invalid",
        "is_url": False
    }
    
    response = client.post("/convert", json=request_data)
    assert response.status_code == 422  # 验证错误

def test_memory_limit(client):
    """测试内存限制"""
    with patch('psutil.virtual_memory') as mock_memory:
        # 模拟内存使用率为90%
        mock_memory.return_value = MagicMock(percent=90)
        
        request_data = {
            "source": "test-uuid",
            "target_format": "json",
            "is_url": False
        }
        
        response = client.post("/convert", json=request_data)
        assert response.status_code == 500
        assert "memory usage too high" in response.json()["detail"]

def test_batch_conversion(client):
    """测试批量转换"""
    requests = [
        {
            "source": "test-uuid-1",
            "target_format": "json",
            "is_url": False
        },
        {
            "source": "test-uuid-2",
            "target_format": "markdown",
            "is_url": False
        }
    ]
    
    response = client.post("/batch", json=requests)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(item["status"] == "pending" for item in data)

def test_result_endpoint(client, setup_and_cleanup):
    """测试结果获取端点"""
    # 首先创建一个转换任务
    from main import conversion_tasks
    task_id = "test-result-task"
    output_path = os.path.join(OUTPUT_DIRECTORY, f"{task_id}_output.markdown")
    
    # 写入一些测试内容
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Test content")
    
    conversion_tasks[task_id] = {
        "status": "completed",
        "result": output_path
    }
    
    response = client.get(f"/result/{task_id}")
    assert response.status_code == 200
    assert response.text == "Test content"

# 添加异步版本的其他测试
@pytest.mark.asyncio
async def test_async_batch_conversion(async_client):
    """测试异步批量转换"""
    requests = [
        {
            "source": "test-uuid-1",
            "target_format": "json",
            "is_url": False
        },
        {
            "source": "test-uuid-2",
            "target_format": "markdown",
            "is_url": False
        }
    ]
    
    # 使用 mock 来避免实际的网络请求
    with patch('main.process_conversion') as mock_process:
        mock_process.return_value = None  # 异步函数返回 None
        response = await async_client.post("/batch", json=requests)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(item["status"] == "pending" for item in data)

def setup_test_file(filename="test.pdf"):
    """创建测试文件并返回路径"""
    filepath = os.path.join(OUTPUT_DIRECTORY, filename)
    with open(filepath, "wb") as f:
        f.write(b"%PDF-1.4\n")  # 最小的有效 PDF
    return filepath

def cleanup_test_files():
    """清理测试文件"""
    for file in os.listdir(OUTPUT_DIRECTORY):
        try:
            os.remove(os.path.join(OUTPUT_DIRECTORY, file))
        except:
            pass 