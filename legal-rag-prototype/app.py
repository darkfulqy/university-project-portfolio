# app.py
from flask import Flask, render_template, request
import os
from docx import Document # 导入 python-docx 库

app = Flask(__name__)

# --- 辅助函数：读取 DOCX 文件内容 ---
def read_docx_content(file_path):
    """
    读取指定路径的 DOCX 文件，并返回其所有段落的文本内容。
    """
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        # 如果文件不是有效的 DOCX 或无法访问，则捕获错误
        raise Exception(f"无法读取 DOCX 文件 '{file_path}': {e}")

# --- 模拟你的同事的外部程序 (占位符) ---
def simulate_external_program(combined_article_text):
    """
    这个函数是你的同事程序处理合并文章的占位符。
    当你拿到同事的程序后，你需要用实际的调用代码替换这里。
    """
    print(f"--- 模拟外部程序处理中 ---")
    print(f"接收到的合并文章长度: {len(combined_article_text)} 字符")

    # 在这里，你会调用你同事的程序，并传递 combined_article_text
    # 例如：
    # import sys
    # sys.path.append('/path/to/your/colleague/program') # 如果程序在其他目录
    # from colleague_module import process_article
    # processed_output = process_article(combined_article_text)

    # 目前，我们返回一个模拟的处理结果
    processed_output = f"外部程序已成功处理您的文章！\n" \
                       f"总字数（估算，基于合并内容）：{len(combined_article_text)} 字\n" \
                       f"这是外部程序返回的模拟结果。当你接入实际程序后，这里会显示它的输出。" \
                       f"\n\n文章预览（前500字）：\n{combined_article_text[:500]}..."

    return processed_output

# --- 核心处理逻辑：接收路径，读取，合并，并调用外部程序 ---
def process_three_files(file_path1, file_path2, file_path3):
    print(f"--- 接收到文件路径进行处理 ---")
    print(f"文件1: {file_path1}")
    print(f"文件2: {file_path2}")
    print(f"文件3: {file_path3}")

    try:
        # 1. 检查文件是否存在
        if not (os.path.exists(file_path1) and os.path.exists(file_path2) and os.path.exists(file_path3)):
            missing_files = []
            if not os.path.exists(file_path1): missing_files.append(f"文件1: {file_path1}")
            if not os.path.exists(file_path2): missing_files.append(f"文件2: {file_path2}")
            if not os.path.exists(file_path3): missing_files.append(f"文件3: {file_path3}")
            return f"错误：以下文件路径不存在或无法访问：\n" + "\n".join(missing_files)

        # 2. 读取三个 DOCX 文件的内容
        doc1_content = read_docx_content(file_path1)
        doc2_content = read_docx_content(file_path2)
        doc3_content = read_docx_content(file_path3)

        # 3. 将三者内容合并成一个文章
        # 使用换行符和分隔符以保持可读性
        combined_article = doc1_content + "\n\n--- 文件2 开始 ---\n\n" + \
                           doc2_content + "\n\n--- 文件3 开始 ---\n\n" + \
                           doc3_content

        print(f"--- 已合并三个 DOCX 文件内容，总长度: {len(combined_article)} 字符 ---")

        # 4. 将合并后的文章交给另一个程序处理
        # 这里调用我们上面定义的模拟函数
        external_program_output = simulate_external_program(combined_article)

        # 5. 返回外部程序的处理结果
        return external_program_output

    except Exception as e:
        # 捕获任何在处理过程中发生的错误，并返回错误消息
        error_message = f"处理过程中发生意外错误: {e}"
        print(f"内部错误详情: {e}") # 在命令行中打印详细错误，便于调试
        return error_message


# -----------------------------------------------------------
# Flask 路由和应用逻辑 (这部分基本不变)
# -----------------------------------------------------------

@app.route('/')
def home():
    """
    当用户访问网站根目录时，显示文件路径输入表单。
    """
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """
    接收表单提交的数据（文件路径），然后调用 Python 处理程序，
    并将处理结果显示回页面。
    """
    if request.method == 'POST':
        # 从表单中获取用户输入的三个文件路径
        file1 = request.form['file1']
        file2 = request.form['file2']
        file3 = request.form['file3']

        # 调用你的 Python 处理程序
        processing_result_message = process_three_files(file1, file2, file3)

        # 将处理结果传递回 index.html 模板，以便在页面上显示
        return render_template('index.html', message=processing_result_message)

if __name__ == '__main__':
    # 运行 Flask 应用。debug=True 会在代码修改时自动重启服务器，
    # 并在开发过程中显示有用的调试信息。
    app.run(debug=True)
