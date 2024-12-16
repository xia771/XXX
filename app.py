import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
import pandas as pd

# 设置页面配置
st.set_page_config(
    page_title="安全头盔检测系统",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
        }
        .stButton>button:hover {
            background-color: #ff6b6b;
        }
    </style>
""", unsafe_allow_html=True)

# 加载模型
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# 模型选择
available_models = [f for f in os.listdir('.') if f.endswith('.pt')]
model_path = 'best.pt' if 'best.pt' in available_models else available_models[0] if available_models else None

if not model_path:
    st.error("未找到可用的模型文件！")
    st.stop()

# 主界面
st.title("🪖 安全头盔检测系统")
st.markdown(f"当前使用模型: `{model_path}`")

# 侧边栏配置
with st.sidebar:
    st.title("⚙️ 系统配置")
    
    st.markdown("---")
    
    # 检测模式选择
    detection_mode = st.selectbox(
        "选择检测模式",
        ["📸 单张图片检测", "📁 批量图片检测", "📂 数据集文件夹检测", "🎥 实时视频检测", "📹 视频文件检测"],
        help="选择要使用的检测模式"
    )
    
    st.markdown("---")
    
    # 模型选择
    model_path = st.selectbox(
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
    st.markdown("""
    ### 💡 使用说明
    1. 选择合适的检测模型
    2. 调整置信度阈值（值越小检出率越高）
    3. 选择检测模式（图片/视频）
    """)

# 加载选择的模型
model = load_model(model_path)

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
                    
                    # 显示检测结果
                    if len(result.boxes) > 0:
                        st.success(f"✅ 检测到 {len(result.boxes)} 个目标")
                        
                        # 创建检测结果表格
                        results_data = []
                        for i, box in enumerate(result.boxes):
                            confidence = box.conf.item()
                            results_data.append({
                                "目标编号": i + 1,
                                "置信度": f"{confidence:.2%}",
                                "坐标": f"({int(box.xyxy[0][0])}, {int(box.xyxy[0][1])}, {int(box.xyxy[0][2])}, {int(box.xyxy[0][3])})"
                            })
                        
                        st.table(results_data)
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
                            
                            # 显示检测结果
                            if len(result.boxes) > 0:
                                st.success(f"检测到 {len(result.boxes)} 个目标")
                                # 显示置信度
                                for i, conf in enumerate([box.conf.item() for box in result.boxes]):
                                    st.text(f"目标 {i+1} 置信度: {conf:.2%}")
                            else:
                                st.warning("未检测到目标")
                            
                            st.text(f"处理时间: {(end_time - start_time):.3f} 秒")
        
        # 完成处理
        progress_bar.empty()
        status_text.success("✅ 所有图片处理完成！")
        
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
                    - 每张图片平均用时: {batch_time/len(batch_files):.3f}秒
                    """)
                
                # 处理完成，显示最终统计
                progress_bar.empty()
                status_text.success("✅ 数据集处理完成！")
                
                # 创建检测结果DataFrame
                df = pd.DataFrame(detection_results)

                # 显示统计信息
                st.markdown("### 📊 检测统计分析")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总处理图片数", processed_count)
                with col2:
                    st.metric("总检测目标数", detection_count)
                with col3:
                    avg_detections = detection_count/max(1, processed_count)
                    st.metric("平均每张检测数", f"{avg_detections:.2f}")
                    
                # 保存统计结果
                if save_results and save_path:
                    # 保存统计报告
                    report_path = os.path.join(save_path, 'detection_report.csv')
                    df.to_csv(report_path, index=False)
                    
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
    if st.button('🎥 开始检测'):
        cap = cv2.VideoCapture(0)
        
        while True:
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
                
                # 显示实时统计
                st.markdown(f"""
                ### 📊 实时监测数据
                - 当前检测目标数: {len(result.boxes)}
                - FPS: {1/(end_time - start_time):.1f}
                """)
            
            # 显示帧
            st.image(frame, channels="BGR", caption="实时检测")
            
            # 控制帧率
            time.sleep(0.01)
            
        cap.release()
        
    st.markdown("---")
    st.info("📝 提示：点击'开始检测'按钮启动摄像头检测。")
    
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
            
            # 创建检测结果DataFrame
            df = pd.DataFrame(detection_results)

st.markdown("---")
st.markdown("第二小组 | 最后更新时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
