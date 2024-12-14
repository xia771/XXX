import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import hashlib
import re

# 设置页面配置
st.set_page_config(
    page_title="安全头盔检测系统",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = 0
if 'detection_times' not in st.session_state:
    st.session_state.detection_times = []
if 'model_path' not in st.session_state:
    st.session_state.model_path = 'best.pt'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

# 用户管理函数
def load_users():
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"users": []}

def save_users(users_data):
    with open('users.json', 'w') as f:
        json.dump(users_data, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_username(username):
    return bool(re.match("^[a-zA-Z0-9_-]{3,20}$", username))

def is_valid_password(password):
    return len(password) >= 6

def register_user(username, password):
    users_data = load_users()
    
    # 检查用户名是否已存在
    if any(user['username'] == username for user in users_data['users']):
        return False, "用户名已存在"
    
    # 验证用户名和密码格式
    if not is_valid_username(username):
        return False, "用户名必须是3-20个字符，只能包含字母、数字、下划线和连字符"
    if not is_valid_password(password):
        return False, "密码长度必须至少为6个字符"
    
    # 添加新用户
    users_data['users'].append({
        'username': username,
        'password': hash_password(password),
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_users(users_data)
    return True, "注册成功"

def login_user(username, password):
    users_data = load_users()
    hashed_password = hash_password(password)
    
    for user in users_data['users']:
        if user['username'] == username and user['password'] == hashed_password:
            return True
    return False

# 登录/注册界面
if not st.session_state.logged_in:
    st.title("🪖 安全头盔检测系统")
    
    tab1, tab2 = st.tabs(["登录", "注册"])
    
    with tab1:
        st.header("👤 用户登录")
        login_username = st.text_input("用户名", key="login_username")
        login_password = st.text_input("密码", type="password", key="login_password")
        
        if st.button("登录"):
            if login_user(login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success("✅ 登录成功！")
                st.rerun()
            else:
                st.error("❌ 用户名或密码错误")
    
    with tab2:
        st.header("📝 新用户注册")
        reg_username = st.text_input("用户名", key="reg_username")
        reg_password = st.text_input("密码", type="password", key="reg_password")
        reg_password2 = st.text_input("确认密码", type="password", key="reg_password2")
        
        if st.button("注册"):
            if reg_password != reg_password2:
                st.error("❌ 两次输入的密码不一致")
            else:
                success, message = register_user(reg_username, reg_password)
                if success:
                    st.success(f"✅ {message}")
                    st.info("请返回登录标签页进行登录")
                else:
                    st.error(f"❌ {message}")
    
    st.markdown("---")
    st.markdown("第二小组 | 最后更新时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

else:
    # 自定义CSS样式
    st.markdown("""
        <style>
            .stApp {
                max-width: 1200px;
                margin: 0 auto;
            }
            .main {
                padding: 2rem;
            }
            .stButton>button {
                width: 100%;
                background-color: #ff4b4b;
                color: white;
            }
            .stButton>button:hover {
                background-color: #ff6b6b;
            }
            .reportview-container {
                margin-top: 2rem;
            }
            .css-1d391kg {
                padding: 1rem;
                border-radius: 0.5rem;
                background-color: #f0f2f6;
            }
            .stMetricValue {
                font-size: 2rem !important;
            }
            h1 {
                color: #ff4b4b;
            }
            h2 {
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # 保存检测记录
    def save_detection_record(image_name, num_detections, confidence_scores):
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_name': image_name,
            'num_detections': num_detections,
            'confidence_scores': confidence_scores,
            'model_used': st.session_state.model_path,
            'username': st.session_state.username
        }
        st.session_state.detection_history.append(record)
        
        # 保存到JSON文件
        with open('detection_history.json', 'w') as f:
            json.dump(st.session_state.detection_history, f)
    
    # 侧边栏配置
    with st.sidebar:
        st.title("⚙️ 系统配置")
        
        # 显示当前用户
        st.markdown(f"### 👤 当前用户: {st.session_state.username}")
        if st.button("登出"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        
        # 模型选择
        available_models = [f for f in os.listdir('.') if f.endswith('.pt')]
        st.session_state.model_path = st.selectbox(
            "选择检测模型",
            available_models,
            index=available_models.index('best.pt') if 'best.pt' in available_models else 0
        )
        
        confidence = st.slider("置信度阈值", 0.0, 1.0, 0.01, 0.01)
        
        # 高级设置折叠面板
        with st.expander("🛠️ 高级设置"):
            iou_threshold = st.slider("IOU阈值", 0.0, 1.0, 0.45, 0.01)
            max_det = st.number_input("最大检测数量", 1, 100, 20)
        
        st.markdown("---")
        st.subheader("📊 实时统计")
        
        # 实时统计数据
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总检测数", st.session_state.total_detections)
        with col2:
            st.metric("处理图片", st.session_state.processed_images)
        
        # 检测历史图表
        if st.session_state.detection_history:
            df = pd.DataFrame(st.session_state.detection_history)
            fig = px.line(df, x=df.index, y='num_detections', 
                         title='检测历史趋势',
                         labels={'index': '检测序号', 'num_detections': '检测数量'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        ### 💡 使用说明
        1. 选择合适的检测模型
        2. 调整置信度阈值（值越小检出率越高）
        3. 选择检测模式（图片/视频）
        4. 查看实时统计和检测历史
        """)
    
    # 加载模型
    @st.cache_resource
    def load_model(model_path):
        model = YOLO(model_path)
        return model
    
    model = load_model(st.session_state.model_path)
    
    # 主界面
    st.title("🪖 安全头盔检测系统")
    st.markdown(f"当前使用模型: `{st.session_state.model_path}`")
    
    # 检测模式选择
    detection_mode = st.radio("选择检测模式", 
                             ["📸 单张图片检测", "📁 批量图片检测", "📂 数据集文件夹检测", 
                              "🎥 实时视频检测", "📹 视频文件检测"], 
                             horizontal=True)
    
    if detection_mode == "📸 单张图片检测":
        uploaded_file = st.file_uploader("选择图片", type=['png', 'jpg', 'jpeg'])
    
        if uploaded_file is not None:
            try:
                # 读取图片
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # 创建两列布局
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📸 原始图片")
                    # 确保图片是RGB格式
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    st.image(image, caption=f"文件名: {uploaded_file.name}")
                
                with col2:
                    st.subheader("🎯 检测结果")
                    start_time = time.time()
                    # 进行预测
                    results = model(img_array, conf=confidence, iou=iou_threshold, max_det=max_det)
                    end_time = time.time()
                    
                    # 在图片上绘制检测结果
                    for result in results:
                        plotted = result.plot()
                        st.image(plotted, caption="检测结果")
                        
                        # 更新统计信息
                        st.session_state.processed_images += 1
                        boxes = result.boxes
                        num_detections = len(boxes)
                        st.session_state.total_detections += num_detections
                        
                        # 保存检测记录
                        confidence_scores = [box.conf.item() for box in boxes]
                        save_detection_record(uploaded_file.name, num_detections, confidence_scores)
                        
                        # 显示检测结果
                        if num_detections > 0:
                            st.success(f"✅ 检测到 {num_detections} 个目标")
                            
                            # 创建检测结果表格
                            results_data = []
                            for i, box in enumerate(boxes):
                                confidence = box.conf.item()
                                results_data.append({
                                    "目标编号": i + 1,
                                    "置信度": f"{confidence:.2%}",
                                    "坐标": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}, {int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})"
                                })
                            
                            st.table(results_data)
                            
                            # 显示置信度分布
                            fig = px.histogram(confidence_scores, 
                                             title='置信度分布',
                                             labels={'value': '置信度', 'count': '数量'},
                                             nbins=10)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ 未检测到目标")
                        
                        st.info(f"⚡ 检测用时: {(end_time - start_time):.3f} 秒")
            except Exception as e:
                st.error(f"处理图片时出错: {str(e)}")
    
    elif detection_mode == "📁 批量图片检测":
        uploaded_files = st.file_uploader("选择多张图片", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        if uploaded_files:
            st.info(f"已上传 {len(uploaded_files)} 张图片")
            
            # 批量处理进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 创建图片网格布局
            cols = st.columns(3)
            
            # 批量处理图片
            for idx, uploaded_file in enumerate(uploaded_files):
                # 更新进度
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"正在处理: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
                
                # 读取图片
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # 进行检测
                start_time = time.time()
                results = model(img_array, conf=confidence, iou=iou_threshold, max_det=max_det)
                end_time = time.time()
                
                # 在对应的列中显示结果
                col_idx = idx % 3
                with cols[col_idx]:
                    # 创建一个expander来显示每张图片的详细信息
                    with st.expander(f"图片 {idx + 1}: {uploaded_file.name}", expanded=True):
                        # 显示原图和检测结果的对比
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="原图")
                        with col2:
                            for result in results:
                                plotted = result.plot()
                                st.image(plotted, caption="检测结果")
                                
                                # 更新统计信息
                                boxes = result.boxes
                                num_detections = len(boxes)
                                st.session_state.total_detections += num_detections
                                st.session_state.processed_images += 1
                                
                                # 保存检测记录
                                confidence_scores = [box.conf.item() for box in boxes]
                                save_detection_record(uploaded_file.name, num_detections, confidence_scores)
                                
                                # 显示检测结果
                                if num_detections > 0:
                                    st.success(f"检测到 {num_detections} 个目标")
                                    # 显示置信度
                                    for i, conf in enumerate(confidence_scores):
                                        st.text(f"目标 {i+1} 置信度: {conf:.2%}")
                                else:
                                    st.warning("未检测到目标")
                                
                                st.text(f"处理时间: {(end_time - start_time):.3f} 秒")
            
            # 完成处理
            progress_bar.empty()
            status_text.success("✅ 所有图片处理完成！")
            
            # 显示批量处理统计
            st.markdown("---")
            st.subheader("📊 批量处理统计")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("处理图片总数", len(uploaded_files))
            with col2:
                st.metric("检测目标总数", st.session_state.total_detections)
            with col3:
                avg_time = sum(st.session_state.detection_times) / len(st.session_state.detection_times) if st.session_state.detection_times else 0
                st.metric("平均处理时间", f"{avg_time:.3f}秒")
    
    elif detection_mode == "📂 数据集文件夹检测":
        st.markdown("### 📂 数据集文件夹检测")
        
        # 输入数据集路径
        dataset_path = st.text_input("输入数据集文件夹路径", "")
        
        # 高级设置
        with st.expander("🛠️ 高级设置"):
            batch_size = st.number_input("批处理大小", min_value=1, max_value=32, value=16)
            save_results = st.checkbox("保存检测结果", value=True)
            save_path = st.text_input("结果保存路径", "detection_results") if save_results else None
            recursive_search = st.checkbox("递归搜索子文件夹", value=True)
            
        if st.button("开始处理数据集") and dataset_path:
            if not os.path.exists(dataset_path):
                st.error("❌ 数据集路径不存在！")
            else:
                # 获取所有图片文件
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
                image_files = []
                
                if recursive_search:
                    for root, _, files in os.walk(dataset_path):
                        for file in files:
                            if file.lower().endswith(image_extensions):
                                image_files.append(os.path.join(root, file))
                else:
                    image_files = [f for f in os.listdir(dataset_path) 
                                 if f.lower().endswith(image_extensions)]
                    image_files = [os.path.join(dataset_path, f) for f in image_files]
                
                if not image_files:
                    st.warning("⚠️ 未找到图片文件！")
                else:
                    st.info(f"找到 {len(image_files)} 个图片文件")
                    
                    # 创建保存目录
                    if save_results and save_path:
                        os.makedirs(save_path, exist_ok=True)
                    
                    # 初始化进度条和计数器
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats_container = st.empty()
                    
                    # 处理统计
                    processed_count = 0
                    detection_count = 0
                    processing_times = []
                    detection_results = []
                    
                    # 批量处理图片
                    for i in range(0, len(image_files), batch_size):
                        batch_files = image_files[i:i + batch_size]
                        batch_images = []
                        
                        # 读取批次图片
                        for img_path in batch_files:
                            try:
                                img = cv2.imread(img_path)
                                if img is not None:
                                    batch_images.append((img_path, img))
                            except Exception as e:
                                st.error(f"处理图片 {img_path} 时出错: {str(e)}")
                        
                        # 批量检测
                        start_time = time.time()
                        results = model([img for _, img in batch_images], 
                                     conf=confidence, 
                                     iou=iou_threshold, 
                                     max_det=max_det)
                        batch_time = time.time() - start_time
                        processing_times.append(batch_time)
                        
                        # 处理每张图片的结果
                        for (img_path, _), result in zip(batch_images, results):
                            processed_count += 1
                            
                            # 获取检测结果
                            boxes = result.boxes
                            num_detections = len(boxes)
                            detection_count += num_detections
                            
                            # 记录结果
                            detection_results.append({
                                'image': os.path.basename(img_path),
                                'detections': num_detections,
                                'confidences': [box.conf.item() for box in boxes]
                            })
                            
                            # 保存结果
                            if save_results and save_path:
                                # 保存标注后的图片
                                output_path = os.path.join(save_path, os.path.basename(img_path))
                                cv2.imwrite(output_path, result.plot())
                                
                                # 保存检测结果到txt
                                txt_path = os.path.splitext(output_path)[0] + '.txt'
                                with open(txt_path, 'w') as f:
                                    for box in boxes:
                                        coords = box.xyxy[0].tolist()
                                        conf = box.conf.item()
                                        f.write(f"helmet {conf:.4f} {' '.join(map(str, coords))}\n")
                        
                        # 更新进度和统计
                        progress = (i + len(batch_files)) / len(image_files)
                        progress_bar.progress(progress)
                        
                        # 更新状态信息
                        status_text.text(f"处理进度: {processed_count}/{len(image_files)} 图片")
                        
                        # 实时统计信息
                        stats_container.markdown(f"""
                        ### 📊 实时处理统计
                        - 已处理图片: {processed_count}/{len(image_files)}
                        - 检测到的目标: {detection_count}
                        - 平均每图目标数: {detection_count/max(1, processed_count):.2f}
                        - 批处理用时: {batch_time:.3f}秒
                        - 每张图片平均用时: {batch_time/len(batch_files):.3f}秒
                        """)
                    
                    # 处理完成，显示最终统计
                    progress_bar.empty()
                    status_text.success("✅ 数据集处理完成！")
                    
                    # 创建详细的统计报告
                    st.markdown("### 📊 处理结果统计")
                    
                    # 基础统计
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("处理图片总数", processed_count)
                    with col2:
                        st.metric("检测目标总数", detection_count)
                    with col3:
                        st.metric("平均每图目标数", f"{detection_count/max(1, processed_count):.2f}")
                    
                    # 创建检测结果DataFrame
                    df = pd.DataFrame(detection_results)
                    
                    # 显示检测分布图表
                    fig1 = px.histogram(df, x='detections',
                                      title='每张图片检测目标数量分布',
                                      labels={'detections': '检测目标数', 'count': '图片数量'})
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # 显示置信度分布
                    all_confidences = [conf for result in detection_results 
                                     for conf in result['confidences']]
                    if all_confidences:
                        fig2 = px.histogram(all_confidences,
                                          title='检测置信度分布',
                                          labels={'value': '置信度', 'count': '目标数量'},
                                          nbins=50)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # 保存统计结果
                    if save_results and save_path:
                        # 保存统计报告
                        report_path = os.path.join(save_path, 'detection_report.csv')
                        df.to_csv(report_path, index=False)
                        
                        # 保存详细结果
                        detailed_results = {
                            'summary': {
                                'total_images': processed_count,
                                'total_detections': detection_count,
                                'avg_detections_per_image': detection_count/max(1, processed_count),
                                'processing_time': sum(processing_times),
                                'avg_time_per_image': sum(processing_times)/max(1, len(processing_times))
                            },
                            'detailed_results': detection_results
                        }
                        
                        with open(os.path.join(save_path, 'detection_results.json'), 'w') as f:
                            json.dump(detailed_results, f, indent=4)
                        
                        st.success(f"✅ 检测结果已保存到: {save_path}")
                        
                        # 提供下载链接
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载检测报告(CSV)",
                                data=f,
                                file_name="detection_report.csv",
                                mime="text/csv"
                            )
    
    elif detection_mode == "🎥 实时视频检测":
        st.warning("⚠️ 请确保已授权摄像头访问权限")
        
        # 开始/停止按钮
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        
        if st.button('🎥 开始检测' if not st.session_state.camera_running else '⏹️ 停止检测'):
            st.session_state.camera_running = not st.session_state.camera_running
        
        # 创建占位符
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("无法访问摄像头")
                    break
                
                # 进行检测
                start_time = time.time()
                results = model(frame, conf=confidence, iou=iou_threshold, max_det=max_det)
                end_time = time.time()
                
                # 绘制结果
                for result in results:
                    frame = result.plot()
                    
                    # 更新统计信息
                    boxes = result.boxes
                    current_detections = len(boxes)
                    st.session_state.total_detections += current_detections
                    
                    # 显示实时统计
                    stats_placeholder.markdown(f"""
                    ### 📊 实时监测数据
                    - 当前检测目标数: {current_detections}
                    - FPS: {1/(end_time - start_time):.1f}
                    """)
                
                # 显示帧
                frame_placeholder.image(frame, channels="BGR", caption="实时检测")
                
                # 控制帧率
                time.sleep(0.01)
                
            cap.release()
        
        st.markdown("---")
        st.info("📝 提示：点击'开始检测'按钮启动摄像头检测，再次点击停止检测。")
    
    else:  # 视频文件检测
        uploaded_video = st.file_uploader("选择视频文件", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            # 保存上传的视频文件
            video_path = f"temp_video_{int(time.time())}.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            # 视频处理控制
            if st.button("开始处理视频"):
                # 打开视频文件
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # 创建进度条和状态显示
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                stats_placeholder = st.empty()
                
                frame_count = 0
                detection_results = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 更新进度
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"处理帧: {frame_count}/{total_frames}")
                    
                    # 每隔几帧进行一次检测
                    if frame_count % 3 == 0:  # 可以调整检测频率
                        # 进行检测
                        results = model(frame, conf=confidence, iou=iou_threshold, max_det=max_det)
                        
                        for result in results:
                            frame = result.plot()
                            boxes = result.boxes
                            num_detections = len(boxes)
                            
                            # 记录检测结果
                            detection_results.append({
                                'frame': frame_count,
                                'detections': num_detections,
                                'confidences': [box.conf.item() for box in boxes]
                            })
                    
                    # 显示处理后的帧
                    frame_placeholder.image(frame, channels="BGR", caption=f"Frame {frame_count}")
                    
                    # 显示实时统计
                    if detection_results:
                        avg_detections = np.mean([r['detections'] for r in detection_results])
                        stats_placeholder.markdown(f"""
                        ### 📊 视频处理统计
                        - 当前帧: {frame_count}/{total_frames}
                        - 平均检测数: {avg_detections:.2f}
                        - 处理进度: {(frame_count/total_frames)*100:.1f}%
                        """)
                
                # 释放资源
                cap.release()
                os.remove(video_path)  # 删除临时视频文件
                
                # 显示最终统计
                st.success("✅ 视频处理完成！")
                
                # 创建检测结果图表
                if detection_results:
                    df = pd.DataFrame(detection_results)
                    fig = px.line(df, x='frame', y='detections',
                                title='检测目标数量随时间变化',
                                labels={'frame': '帧数', 'detections': '检测目标数'})
                    st.plotly_chart(fig, use_container_width=True)
    
    # 统计分析部分
    st.markdown("---")
    st.subheader("📊 检测统计分析")
    
    # 创建三列布局显示主要指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总处理图片数", st.session_state.processed_images)
    with col2:
        st.metric("总检测目标数", st.session_state.total_detections)
    with col3:
        avg_detections = st.session_state.total_detections/max(1, st.session_state.processed_images)
        st.metric("平均每张检测数", f"{avg_detections:.2f}")
    with col4:
        if st.session_state.detection_history:
            max_detections = max(record['num_detections'] for record in st.session_state.detection_history)
            st.metric("单张最多检测数", max_detections)
    
    # 显示检测历史详情
    if st.session_state.detection_history:
        with st.expander("📈 查看详细检测历史"):
            history_df = pd.DataFrame(st.session_state.detection_history)
            st.dataframe(history_df)
            
            # 下载检测历史
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 下载检测历史",
                data=csv,
                file_name="detection_history.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    st.markdown("第二小组 | 最后更新时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
