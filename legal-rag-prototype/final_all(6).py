# app.py
from flask import Flask, render_template, request, url_for
import os
from docx import Document # 用于读取 .docx 文件
import win32com.client as win32 # 用于读取 .doc 文件，仅限 Windows
import requests # 用于 Ollama API
import json # 用于处理 JSON 格式数据
from collections import Counter # 用于统计案号和文件路径
import chromadb # 用于 ChromaDB 向量数据库
from openai import OpenAI # 用于大模型 API 调用
import pythoncom


app = Flask(__name__)

# --- 文件读取辅助函数 ---
# 读取 .doc 文件的函数 (仅限 Windows，需要安装 Word)
def read_doc_file(path):
    pythoncom.CoInitialize() 
    try:
        word = win32.gencache.EnsureDispatch('Word.Application')
        word.Visible = False  # 不显示 Word 界面
        doc = word.Documents.Open(path)
        text = doc.Content.Text
        doc.Close(SaveChanges=False)
        word.Quit()
        return text
    except Exception as e:
        print(f"Error reading .doc file '{path}': {e}")
        # 尝试清理 Word 进程 (有时 Word 会卡住)
        try:
            word.Quit()
        except:
            pass
        raise Exception(f"无法读取 .doc 文件 (可能不是有效的 .doc 或缺少 Word/权限): {e}")

# 读取 .docx 文件的函数
def read_docx_content(file_path):
    """
    读取指定路径的 DOCX 文件，并返回其所有段落的文本内容。
    """
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            # strip() 可以去除行首尾的空白字符，包括换行符
            if para.text.strip(): # 仅添加非空行
                full_text.append(para.text.strip())
        return '\n'.join(full_text)
    except Exception as e:
        raise Exception(f"无法读取 DOCX 文件 '{file_path}' (可能不是有效的 .docx): {e}")


# --- Ollama 嵌入函数 ---
def ollama_embedding(text):
    """
    使用本地 Ollama 服务生成文本嵌入向量。
    需要确保 Ollama 服务在 http://127.0.0.1:11434 运行，并且已拉取 nomic-embed-text:latest 模型。
    """
    try:
        res = requests.post(url="http://127.0.0.1:11434/api/embeddings",
                            json={"model": "nomic-embed-text:latest", "prompt": text})
        res.raise_for_status() # 检查 HTTP 请求是否成功
        embedding = res.json()['embedding']
        return embedding
    except requests.exceptions.ConnectionError:
        raise Exception("无法连接到 Ollama 服务，请确保 Ollama 已启动并在运行。")
    except Exception as e:
        print(f"Ollama embedding error: {e}")
        return [] # 返回空列表表示失败


# --- AI 模型调用函数 ---
def fenleiqs(qs):
    """
    调用大模型从文本中提取结构化信息。
    需要配置正确的 API Key 和 Base URL。
    """
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"], # Read the credential from the process environment.
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 使用 Qwen-plus 模型
            messages=[
                {"role": "system", "content": "你是一个法律问题专家，善于从user的文本中分辨用户已知的关于案件的信息。"},
                {"role": "user", "content": f"请你帮我从以文本中提取用户已知的信息，如果文本中没有提及请填未知,输出json格式，注意不要有多余的空格和换行符。请确保json格式正确，且所有字段都被填充。提取的结构化信息中只需要有以下几个字段：案号、法院、被告人姓名、被告人性别、被告人出生日期、被告人民族、被告人职业、指控罪名、诉讼请求、事实和理由、证据、判决结果、判决依据、审判日期。文本全文如下：{qs}"},
            ],
            # extra_body={"enable_thinking": False}, # 根据模型和需求决定是否需要
        )
        json_str = completion.choices[0].message.content
        # 移除可能的 markdown 围栏
        if json_str.startswith("```json"):
            json_str = json_str[7:].strip()
        if json_str.endswith("```"):
            json_str = json_str[:-3].strip()

        record = json.loads(json_str)
        return record
    except Exception as e:
        raise Exception(f"调用 fenleiqs 大模型失败: {e}. 检查 API Key/URL 或模型输出格式。")

def shengcheng(reference_text, user_question_text):
    """
    调用大模型根据参考判决书和用户文本生成判决书。
    需要配置正确的 API Key 和 Base URL。
    """
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"], # Read the credential from the process environment.
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 使用 Qwen-plus 模型
            messages=[
                {"role": "system", "content": "你是一个法律问题专家，善根据user的参考判决书和user的关于本案件的文本写出本案件的判决书。"},
                {"role": "user", "content": f"帮我从以下参考判决书和用户文本中生成一份判决书.参考判决书:{reference_text},用户文本:{user_question_text}。请确保判决书内容完整，格式正确，且所有字段都被填充。"},
            ],
            # extra_body={"enable_thinking": False}, # 根据模型和需求决定是否需要
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"调用 shengcheng 大模型失败: {e}. 检查 API Key/URL 或模型输出。")

cankao_path=[]
# 搜索
def search(record):
    client = chromadb.PersistentClient(path="db/chroma_db")
    collection= client.get_collection(name="test_1")
    query_fields = {
        "事实和理由": ollama_embedding(record.get("事实和理由", "")),
        "证据": ollama_embedding(record.get("证据", "")),
        "诉讼请求": ollama_embedding(record.get("诉讼请求", "")),
        "被告人性别": ollama_embedding(record.get("被告人性别", "")),
        "控诉罪名": ollama_embedding(record.get("控诉罪名", ""))
    }
    anhao_votes = []
    wenjian_vote = []
    text=[]
    for field, embedding in query_fields.items():
        collection= client.get_collection(name="test_1")
        if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
            print(f"⚠️ 字段【{field}】生成向量失败或为空，跳过查询")
            continue
        results = collection.query(
            query_embeddings=[query_fields[field]], 
            n_results=10,
            include=['metadatas']
        )
        
        print(query_fields[field])
        if not results["metadatas"] or len(results["metadatas"][0]) == 0:
            print(f"⚠️ 字段【{field}】没有查到任何相关案号")
            continue
        # 提取所有案号并投票
        for metadata in results["metadatas"][0]:
            anhao = metadata.get("案号", "未知")
            wenjianming=metadata.get("文件路径", "未知")
            anhao_votes.append(anhao)
            wenjian_vote.append(wenjianming)
            if field=="事实和理由":
                anhao_votes.append(anhao)
                wenjian_vote.append(wenjianming)
            elif field=="诉讼请求":
                anhao_votes.append(anhao)
                wenjian_vote.append(wenjianming)
    counter = Counter(anhao_votes)
    top3 = counter.most_common(3)
    counter1= Counter(wenjian_vote)
    top3wenjian = counter1.most_common(3)
    print("🔍 匹配到最多的3个相关案号为：")
    for anhao, count in top3:
        print(f"{anhao}（出现次数：{count}）")
    i=0
    for wenjian in top3wenjian:
        filepath = wenjian[0].replace("\\", "/")
        cankao_path.append(filepath)
        text.append(read_doc_file(filepath))
        i+=1
    return text



# --- 获取图片 URLs 的函数 ---
def get_image_urls(is_initial=True):
    """
    根据是否为初始状态，返回相应的图片 URLs。
    """
    if is_initial:
        image_filenames = ["initial_img_1.png", "initial_img_2.png", "initial_img_3.png"]
    else:
        # 在这里定义你的“处理后”图片文件名
        image_filenames = ["result_img_1.png", "result_img_2.png", "result_img_3.png"]

    # 确保这些图片文件存在于 static 文件夹中
    for filename in image_filenames:
        if not os.path.exists(os.path.join(app.static_folder, filename)):
            print(f"警告: 图片文件 '{filename}' 不存在于 static 文件夹中。请确保已放置。")

    return [url_for('static', filename=f) for f in image_filenames]


# --- 核心处理逻辑：接收路径，读取，合并，调用大模型和搜索 ---
def process_three_files(file_path1, file_path2, file_path3):
    print(f"--- 接收到文件路径进行处理 ---")
    print(f"文件1: {file_path1}")
    print(f"文件2: {file_path2}")
    print(f"文件3: {file_path3}")

    try:
        # 1. 检查输入 DOCX 文件是否存在
        if not (os.path.exists(file_path1) and os.path.exists(file_path2) and os.path.exists(file_path3)):
            missing_files = []
            if not os.path.exists(file_path1): missing_files.append(f"文件1: {file_path1}")
            if not os.path.exists(file_path2): missing_files.append(f"文件2: {file_path2}")
            if not os.path.exists(file_path3): missing_files.append(f"文件3: {file_path3}")
            return {
                "message": f"错误：以下文件路径不存在或无法访问：\n" + "\n".join(missing_files),
                "img_srcs": get_image_urls(is_initial=True) # 失败时仍然显示初始图片
            }

        # 2. 读取三个输入 DOCX 文件的内容并合并
        doc1_content = read_docx_content(file_path1)
        doc2_content = read_docx_content(file_path2)
        doc3_content = read_docx_content(file_path3)

        combined_user_article = doc1_content + "\n\n--- 文件2 开始 ---\n\n" + \
                               doc2_content + "\n\n--- 文件3 开始 ---\n\n" + \
                               doc3_content

        print(f"--- 已合并用户输入的 DOCX 文件内容，总长度: {len(combined_user_article)} 字符 ---")

        # 3. 使用 fenleiqs 从合并的用户文章中提取结构化信息
        extracted_record = fenleiqs(combined_user_article)
        print("--- 大模型提取的结构化信息 ---")
        print(json.dumps(extracted_record, ensure_ascii=False, indent=2))

        # 4. 使用提取的记录在 ChromaDB 中搜索参考判决书
        reference_judgments_text = search(extracted_record)
        print("--- 从 ChromaDB 检索到的参考判决书内容 ---")
        print(reference_judgments_text[:500] + "..." if len(reference_judgments_text) > 500 else reference_judgments_text)

        # 5. 使用 shengcheng 生成最终判决书
        final_judgment = shengcheng(reference_judgments_text, combined_user_article)
        print("--- 大模型生成的最终判决书 ---")
        print(final_judgment[:500] + "..." if len(final_judgment) > 500 else final_judgment)


        # 6. 返回处理结果和处理后的图片 URLs
        return {
            "message": final_judgment, # 将判决书内容作为消息显示
            "img_srcs": get_image_urls(is_initial=False) # 显示处理后的图片
        }

    except Exception as e:
        error_message = f"处理过程中发生意外错误: {e}"
        print(f"内部错误详情: {e}") # 在命令行中打印详细错误，便于调试
        return {
            "message": error_message,
            "img_srcs": get_image_urls(is_initial=True) # 失败时仍然显示初始图片
        }


# -----------------------------------------------------------
# Flask 路由和应用逻辑
# -----------------------------------------------------------

@app.route('/')
def home():
    """
    当用户访问网站根目录时，显示文件路径输入表单和初始参考文本框。
    """
    # 初始时，参考案件文本框为空或显示提示
    return render_template(
        'index.html',
        ref_text1="此处将显示第一个参考案件的内容...",
        ref_text2="此处将显示第二个参考案件的内容...",
        ref_text3="此处将显示第三个参考案件的内容..."
    )

@app.route('/process', methods=['POST'])
# def process():
#     """
#     接收表单提交的数据（文件路径），然后调用 Python 处理程序，
#     并将处理结果和新的参考文本显示回页面。
#     """
    
#     file1 = read_doc_file(cankao_path[0])
#     file2 = read_doc_file(cankao_path[1])
#     file3 = read_doc_file(cankao_path[2])

#     processing_result = process_three_files(file1, file2, file3)

#     message_to_display = processing_result["message"]
#         # 确保 reference_texts 存在，如果不存在则使用空列表
#     reference_texts_from_backend = processing_result.get("reference_texts", ["", "", ""])

#         # 确保有足够的元素传递给 HTML 模板，不足则用默认提示填充
#     ref_text1 = reference_texts_from_backend[0] if len(reference_texts_from_backend) > 0 else "未找到参考案件。"
#     ref_text2 = reference_texts_from_backend[1] if len(reference_texts_from_backend) > 1 else "未找到更多参考案件。"
#     ref_text3 = reference_texts_from_backend[2] if len(reference_texts_from_backend) > 2 else "未找到更多参考案件。"

#     return render_template(
#         'index.html',
#         message=message_to_display,
#         ref_text1=ref_text1,
#         ref_text2=ref_text2,
#         ref_text3=ref_text3
#         )

# if __name__ == '__main__':
#     # 确保 static 文件夹存在 (用于存放音乐)
#     if not os.path.exists('static'):
#         os.makedirs('static')
#     # 确保 db 文件夹存在 (用于 ChromaDB)
#     if not os.path.exists('db'):
#         os.makedirs('db')
#     # 确保 db/chroma_db 文件夹存在
#     if not os.path.exists('db/chroma_db'):
#         os.makedirs('db/chroma_db')

#     app.run(debug=True)

def process():
    """
    接收表单提交的数据（文件路径），然后调用 Python 处理程序，
    并将处理结果和新的参考文本显示回页面。
    """
    if request.method == 'POST':
        file1 = request.form['file1']
        file2 = request.form['file2']
        file3 = request.form['file3']

        processing_result = process_three_files(file1, file2, file3)

        message_to_display = processing_result["message"]
        # 确保 reference_texts 存在，如果不存在则使用空列表
        reference_texts_from_backend = processing_result.get("reference_texts", ["", "", ""])

        # 确保有足够的元素传递给 HTML 模板，不足则用默认提示填充
        ref_text1 = reference_texts_from_backend[0] if len(reference_texts_from_backend) > 0 else "未找到参考案件。"
        ref_text2 = reference_texts_from_backend[1] if len(reference_texts_from_backend) > 1 else "未找到更多参考案件。"
        ref_text3 = reference_texts_from_backend[2] if len(reference_texts_from_backend) > 2 else "未找到更多参考案件。"

        return render_template(
            'index.html',
            message=message_to_display,
            ref_text1=ref_text1,
            ref_text2=ref_text2,
            ref_text3=ref_text3
        )

if __name__ == '__main__':
    # 确保 static 文件夹存在 (用于存放音乐)
    if not os.path.exists('static'):
        os.makedirs('static')
    # 确保 db 文件夹存在 (用于 ChromaDB)
    if not os.path.exists('db'):
        os.makedirs('db')
    # 确保 db/chroma_db 文件夹存在
    if not os.path.exists('db/chroma_db'):
        os.makedirs('db/chroma_db')

    app.run(debug=True)