import cv2
import mediapipe as mp
import numpy as np
import joblib

# Initialize mediapipe Hands Detection
mp_hands = mp.solutions.hands  
hands = mp_hands.Hands()

# 載入 SVM 模型和標準化器
model_filename_1 = "svm_model_V3_len_hand_1.pkl"
clf_1 = joblib.load(model_filename_1)
scaler_filename_1 = "scaler_V3_len_hand_1.pkl"
scaler_1 = joblib.load(scaler_filename_1)

model_filename_2 = "svm_model_V3_len_hand_2.pkl"
clf_2 = joblib.load(model_filename_2)
scaler_filename_2 = "scaler_V3_len_hand_2.pkl"
scaler_2 = joblib.load(scaler_filename_2)

# 載入標籤
label1_file = "labels1_V3.txt"
with open(label1_file, 'r') as f:
    labels = f.readlines()
labels1 = [label.strip() for label in labels]

label2_file = "labels2_V3.txt"
with open(label2_file, 'r') as f:
    labels = f.readlines()
labels2 = [label.strip() for label in labels]


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
        p1 = np.array(hand1[pair[0]])
        p2 = np.array(hand2[pair[1]])

        distance = np.linalg.norm(p1-p2)
        twohand_distance.append(distance)
    
    return twohand_distance

cap = cv2.VideoCapture(0)
hand_1_data = []
hand_2_data = []
tem = 1

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for index, landmarks in enumerate(results.multi_hand_landmarks):
            # Distinguish between left and right hand
            if len(results.multi_hand_landmarks) == 1:
                hand_label = "Left" if results.multi_handedness[index].classification[0].label == "Left" else "Right"

                distances = compute_distances(landmarks)
                distances = scaler_1.transform([distances])

                prediction = clf_1.predict(distances)
                confidence = np.max(clf_1.predict_proba(distances))

                label = labels1[prediction[0]]
                display_text = f"{hand_label} Hand: {label} ({confidence*100:.2f}%)"

                cv2.putText(frame, display_text, (10, 30 + (index * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif len(results.multi_hand_landmarks) == 2:
                hand_data = twohand_collect(landmarks)
                if tem == 1:
                    hand_1_data = hand_data
                elif tem == 2:
                    hand_2_data = hand_data
                tem += 1
                if hand_1_data and hand_2_data:
                    twohand_distance = compute_twohand_distances(hand_1_data, hand_2_data)
                    distances = scaler_2.transform([twohand_distance])

                    prediction = clf_2.predict(distances)
                    confidence = np.max(clf_2.predict_proba(distances))

                    label = labels2[prediction[0]]
                    display_text = f"Hand: {label} ({confidence*100:.2f}%)"
                    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        tem = 1
        hand_1_data = []
        hand_2_data = []

    cv2.imshow('Demo', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()