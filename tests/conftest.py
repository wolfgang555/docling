import pytest
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 创建测试数据目录
@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """设置测试环境"""
    # 创建测试数据目录
    os.makedirs("tests/data", exist_ok=True)
    
    # 如果没有测试 PDF，创建一个简单的测试 PDF
    if not os.path.exists("tests/data/sample.pdf"):
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas("tests/data/sample.pdf")
            c.drawString(100, 750, "Test PDF Document")
            c.save()
        except ImportError:
            # 如果没有 reportlab，创建一个空文件
            with open("tests/data/sample.pdf", 'wb') as f:
                f.write(b"%PDF-1.4\n") 