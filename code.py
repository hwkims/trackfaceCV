import cv2
import mediapipe as mp
import os

# TensorFlow 경고 메시지 제어 (선택 사항)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # WARNING, ERROR, FATAL 메시지만 표시


# MediaPipe 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 조준선 그리기 함수
def draw_crosshair(image, center, size, color, thickness):
    x, y = center
    cv2.line(image, (x - size, y), (x + size, y), color, thickness)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(image, center, size // 2, color, thickness)

# 웹캠 초기화 및 설정 확인
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠

# 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다. 웹캠 연결 상태 및 사용 권한을 확인하세요.")
    exit()

# 웹캠 설정 (선택 사항 - 해상도, 프레임 속도 등)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 높이 설정
cap.set(cv2.CAP_PROP_FPS, 30)  # 프레임 속도 설정 (카메라가 지원하는 경우)


while True:
    # 프레임 읽기 및 오류 처리 강화
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 웹캠 연결을 다시 확인하거나 다른 웹캠을 시도해 보세요.")
        #  cv2.waitKey(3000) # 3초 대기 후 다시 시도 (선택 사항)
        #  continue        # 다음 프레임으로 넘어감 (선택 사항)
        break             #  while loop 종료.  상황에 맞게 continue/break 선택

    # BGR -> RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Mesh 처리
    results = face_mesh.process(frame_rgb)

    # 얼굴 랜드마크가 검출되면
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[4]
            image_height, image_width, _ = frame.shape
            center_x = int(nose_tip.x * image_width)
            center_y = int(nose_tip.y * image_height)

            crosshair_size = 30
            crosshair_color = (0, 255, 0)  # Green
            crosshair_thickness = 2
            draw_crosshair(frame, (center_x, center_y), crosshair_size, crosshair_color, crosshair_thickness)

    # 결과 화면 표시 (전체 화면 모드 - 선택 사항)
    cv2.namedWindow('Head Tracking', cv2.WINDOW_NORMAL)  # 창 크기 조절 가능하도록 설정
    # cv2.setWindowProperty('Head Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # 전체화면

    cv2.imshow('Head Tracking', frame)


    # 'q' 또는 'esc' 키를 누르면 종료
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 27 is the ESC key
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
