import obsws_python as obs
import cv2
import mediapipe as mp
from collections import deque
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into environment
# Replace MY_VAR with your variable name


# ------------------- OBS SETUP -------------------
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = os.getenv("OBS_KEY")

DEFAULT_SCENE = "Desktop W/ Cam"
FOCUS_SCENE = "Raw_Cam"

ws = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD, timeout=3)
ws.set_current_program_scene(DEFAULT_SCENE)

# ------------------- Mediapipe SETUP -------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(1)  # OBS Virtual Camera index

# ------------------- Smoothing & thresholds -------------------
SMOOTHING_FRAMES = 5
proj_history_left = deque(maxlen=SMOOTHING_FRAMES)
proj_history_right = deque(maxlen=SMOOTHING_FRAMES)  # future use

current_state = False  

ENTER_THRESHOLD = 0.6  
EXIT_THRESHOLD = 0.5 

# ------------------- Eye projection functions -------------------
def compute_left_eye_proj(landmarks, width, height):
    """Camera-facing left eye (currently active)"""
    iris = landmarks[473]
    outer = landmarks[263] 
    inner = landmarks[362]  

    ix, iy = int(iris.x * width), int(iris.y * height)
    ox, oy = int(outer.x * width), int(outer.y * height)
    ixr, iyr = int(inner.x * width), int(inner.y * height)

    eye_vec_x = ox - ixr
    eye_vec_y = oy - iyr
    iris_vec_x = ix - ixr
    iris_vec_y = iy - iyr

    eye_len_sq = eye_vec_x**2 + eye_vec_y**2
    proj = (iris_vec_x * eye_vec_x + iris_vec_y * eye_vec_y) / eye_len_sq

    return proj, (ix, iy), (ixr, iyr), (ox, oy)

def compute_right_eye_proj(landmarks, width, height):
    """Camera-facing right eye (optional, not active yet)"""
    iris = landmarks[468]  # left eye from Mediapipe view
    outer = landmarks[33]  
    inner = landmarks[133]  

    ix, iy = int(iris.x * width), int(iris.y * height)
    ox, oy = int(outer.x * width), int(outer.y * height)
    ixr, iyr = int(inner.x * width), int(inner.y * height)

    eye_vec_x = ox - ixr
    eye_vec_y = oy - iyr
    iris_vec_x = ix - ixr
    iris_vec_y = iy - iyr

    eye_len_sq = eye_vec_x**2 + eye_vec_y**2
    proj = (iris_vec_x * eye_vec_x + iris_vec_y * eye_vec_y) / eye_len_sq

    return proj, (ix, iy), (ixr, iyr), (ox, oy)

def update_looking_state(proj, current_state):
    if current_state:
        return proj > EXIT_THRESHOLD
    else:
        return proj > ENTER_THRESHOLD

# ------------------- MAIN LOOP -------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    proj_left = 0.5  
    debug_coords_left = {}
    proj_right = 0.5
    debug_coords_right = {}

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Left eye
            proj_left, iris_coord_l, inner_coord_l, outer_coord_l = compute_left_eye_proj(face_landmarks.landmark, w, h)
            debug_coords_left = {
                "iris": iris_coord_l,
                "inner": inner_coord_l,
                "outer": outer_coord_l
            }


            proj_right, iris_coord_r, inner_coord_r, outer_coord_r = compute_right_eye_proj(face_landmarks.landmark, w, h)
            debug_coords_right = {
                "iris": iris_coord_r,
                "inner": inner_coord_r,
                "outer": outer_coord_r
            }
            break


    proj_history_left.append(proj_left)
    smoothed_proj_left = sum(proj_history_left) / len(proj_history_left)

    proj_history_right.append(proj_right)  # for future use
    smoothed_proj_right = sum(proj_history_right) / len(proj_history_right)


    current_state = update_looking_state(smoothed_proj_left, current_state)


    try:
        if current_state and ws.get_current_program_scene().current_program_scene_name != FOCUS_SCENE:
            ws.set_current_program_scene(FOCUS_SCENE)
        elif not current_state and ws.get_current_program_scene().current_program_scene_name != DEFAULT_SCENE:
            ws.set_current_program_scene(DEFAULT_SCENE)
    except Exception as e:
        print("OBS Error:", e)


    if debug_coords_left:
        lx, ly = debug_coords_left["iris"]
        li_x, li_y = debug_coords_left["inner"]
        lo_x, lo_y = debug_coords_left["outer"]

        cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)
        cv2.circle(frame, (li_x, li_y), 2, (0, 0, 255), -1)
        cv2.circle(frame, (lo_x, lo_y), 2, (0, 0, 255), -1)
        cv2.line(frame, (li_x, li_y), (lo_x, lo_y), (255, 0, 0), 1)

 
    if debug_coords_right:
        rx, ry = debug_coords_right["iris"]
        ri_x, ri_y = debug_coords_right["inner"]
        ro_x, ro_y = debug_coords_right["outer"]

      
        cv2.circle(frame, (rx, ry), 3, (0, 255, 255), -1)  # cyan iris
        cv2.circle(frame, (ri_x, ri_y), 2, (255, 0, 255), -1)  # magenta corners
        cv2.circle(frame, (ro_x, ro_y), 2, (255, 0, 255), -1)
        cv2.line(frame, (ri_x, ri_y), (ro_x, ro_y), (255, 255, 0), 1)

    # Status
    status_text = "LOOKING" if current_state else "NOT LOOKING"
    color = (0, 255, 0) if current_state else (0, 0, 255)
    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Eye Tracker Debug", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ws.set_current_program_scene(DEFAULT_SCENE)
ws.disconnect()
