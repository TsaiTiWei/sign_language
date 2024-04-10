import cv2  # 導入 OpenCV 库
import mediapipe as mp  # 導入 Mediapipe 库
import numpy as np  # 導入 NumPy 库
import time  # 導入時間模組
import os  # 導入檔案操作模組

# 初始化 Mediapipe Hands
mp_hands = mp.solutions.hands # 偵測手掌方法
hands = mp_hands.Hands() # 初始化手部追蹤


len_hand = 1

def compute_distances(landmarks):
    distances = []
    
    # 定義用於距離計算的配對
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8),
             (0, 9), (0, 10), (0, 11), (0, 12),
             (0, 13), (0, 14), (0, 15), (0, 16),
             (0, 17), (0, 18), (0, 19), (0, 20),
             (4, 8), (8, 12), (12, 16), (16, 20)]

    reference_pair = (0, 9)
    p_ref1 = np.array([landmarks.landmark[reference_pair[0]].x, landmarks.landmark[reference_pair[0]].y])
    p_ref2 = np.array([landmarks.landmark[reference_pair[1]].x, landmarks.landmark[reference_pair[1]].y])
    reference_distance = np.linalg.norm(p_ref1 - p_ref2)

    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2) / reference_distance
        distances.append(distance)

    return distances

def twohand_collect(landmarks):
    hand_data = []
    
    for i in range(21):
        x1 = np.array([landmarks.landmark[i].x, landmarks.landmark[i].y])
        print("x1:", x1)
        hand_data.append(x1)

    return hand_data

def compute_twohand_distances(hand1, hand2):
    twohand_distance = []
    pairs = [(0, 0), (1, 1), (2, 2), (3, 3),
             (4, 4), (5, 5), (6, 6), (7, 7),
             (8, 8), (9, 9), (10, 10), (11, 11),
             (12, 12), (13, 13), (14, 14), (15, 15),
             (16, 16), (17, 17), (18, 18), (19, 19), 
             (20, 20), (0, 8), (0, 12), (0, 16)]
    
    for pair in pairs:
        p1 = np.array(hand1[0][pair[0]])
        p2 = np.array(hand2[0][pair[1]])
        print("p1-p2:", p1-p2)
        distance = np.linalg.norm(p1 - p2)
        twohand_distance.append(distance)
    
    return twohand_distance

# 詢問使用者檔案名稱
if len_hand == 1:
    file = "dataset_onehand_V3/"
elif len_hand == 2:
    file = "dataset_twohand_V3/"

filename = input("請輸入資料的檔案名稱: ")
save_path = file + filename

# 檢查 'dataset' 目錄是否存在，若不存在則創建
if len_hand == 1 and not os.path.exists("dataset_onehand_V3"):
    os.makedirs("dataset_onehand_V3")
elif len_hand == 2 and not os.path.exists("dataset_twohand_V3"):
    os.makedirs("dataset_twohand_V3")

# 鏡頭
cap = cv2.VideoCapture(0)
data_collection = []
hand_1_data = []
hand_2_data = []
tem = 1

collecting = False # 是否按下空白鍵
ready_time = False
start_time = None

while True:
    ret, frame = cap.read()
    # 翻轉左右鏡頭
    frame = cv2.flip(frame, 1)
    if not ret:
        continue

    if not collecting:
        cv2.putText(frame, "Press SPACE to start data collection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        if not ready_time:
            elapsed_time_key = int(time.time() - start_time_key)
            remaining_time_key = 3 - elapsed_time_key
            cv2.putText(frame, f"Ready: {remaining_time_key} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 183, 50), 2, cv2.LINE_AA)
            if elapsed_time_key >= 3:
                ready_time = True
                start_time = time.time()
        else:
            elapsed_time = int(time.time() - start_time)
            remaining_time = 10 - elapsed_time
            cv2.putText(frame, f"Time left: {remaining_time} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if elapsed_time >= 10:
                break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and collecting and ready_time:
        if len_hand == 1:
            for landmarks in results.multi_hand_landmarks:
                distances = compute_distances(landmarks)
                data_collection.append(distances)
            
        elif len_hand == 2:
            hand_1_data = []
            hand_2_data = []
            for landmarks in results.multi_hand_landmarks:
                hand_data = twohand_collect(landmarks)
                if tem == 1:
                    hand_1_data.append(hand_data)
                    print("hand_1_data :", hand_1_data)
                else:
                    hand_2_data.append(hand_data)
                    print("hand_2_data :", hand_2_data)
                tem += 1    
            
            if hand_1_data and hand_2_data:
                twohand_distance = compute_twohand_distances(hand_1_data, hand_2_data)
                print("twohand_distance:", twohand_distance)
                data_collection.append(twohand_distance)
            tem = 1

    cv2.imshow("資料收集", frame)
    key = cv2.waitKey(1)
    
    if key == 32 and not collecting:
        collecting = True
        start_time_key = time.time()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("data_collect:\n", data_collection)

# 將 data_collection 列表轉換為 NumPy 陣列並儲存
np.save(save_path, np.array(data_collection))
print(data_collection)
print(f"資料已儲存至 {save_path}")
