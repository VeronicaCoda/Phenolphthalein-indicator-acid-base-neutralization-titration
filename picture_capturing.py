import cv2
import os
import time


def get_picture(cap, save_dir):  # 获取照片
    # 捕获一帧的数据
    ret, frame = cap.read()
    if not ret or frame is None:
        print("未能捕获到帧")
        return None  # 返回 None 表示没有捕获到图像

    # 显示捕获的图像
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(100)  # 显示图像1毫秒

    # 生成基于当前时间戳的文件名
    image_name = str(int(time.time())) + ".jpg"
    # 照片存储位置
    filepath = os.path.join(save_dir, image_name)

    # 将照片保存起来
    cv2.imwrite(filepath, frame)

    return image_name


def capture_training_images(save_dir, num_images=100, interval=1):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 表示第一个摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        return


    print(f"开始捕获图像，保存到目录: {save_dir}")

    for i in range(num_images):
        image_name = get_picture(cap, save_dir)
        if image_name:
            print(f"已保存图像: {image_name}")
        else:
            print("捕获图像失败")

        time.sleep(interval)  # 等待指定的间隔时间再捕获下一张图像

    # 释放摄像头
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

    print("图像捕获完成")


if __name__ == '__main__':
    save_directory = "original data/Phenolphthalein"  # 修改为你想要保存图像的目录
    capture_training_images(save_directory, num_images=10000, interval=1)