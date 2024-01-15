import face_recognition
import cv2
import sys

def recongnition(image_path, face_path):
    # 读取图像并将其转换为RGB格式
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取面部位置
    face_locations = face_recognition.face_locations(image)

    # 获取面部编码
    image_encodings = face_recognition.face_encodings(image, face_locations)

    if len(image_encodings) < 1:
        return 3  # 未检测到面部

    # 读取第二张图像并将其转换为RGB格式
    face_image = cv2.imread(face_path)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # 获取第二张图像的面部位置
    face_locations = face_recognition.face_locations(face_image)

    # 获取第二张图像的面部编码
    face_encoding = face_recognition.face_encodings(face_image, face_locations)

    if len(face_encoding) < 1:
        return 3  # 未检测到面部

    # 比较面部编码
    matches = face_recognition.compare_faces([face_encoding[0]], image_encodings[0], tolerance=0.49)
    if True in matches:
        return 2  # 人脸匹配
    return 1  # 人脸不匹配

if __name__ == '__main__':
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    #path1 = "C:/Users/cheny/Desktop/project-interface-main/src/main/resources/static/face/1.png"
    #path2 = "C:/Users/cheny/Desktop/project-interface-main/src/main/resources/static/check/1_45.png"
    result = recongnition(path1, path2)
    print(result)
