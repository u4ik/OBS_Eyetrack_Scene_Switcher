import obsws_python as obs
import cv2
import mediapipe as mp
from dotenv import load_dotenv
import os
import warnings
import logging

# Suppress TensorFlow messages (0 = all, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress Python warnings (like protobuf deprecation)
warnings.filterwarnings("ignore")

# Optional: suppress absl logging too
logging.getLogger("absl").setLevel(logging.ERROR)

load_dotenv()  # Loads variables from .env into environment

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
alpha = 0.2  # EMA smoothing factor (lower = smoother, higher = more responsive)
smoothed_proj_left = 0.5  # initialize
smoothed_proj_right = 0.5

current_state = False

ENTER_THRESHOLD = 0.6  # When pupil moves toward the camera, enter LOOKING state
EXIT_THRESHOLD = 0.5  # Hysteresis prevents flickering


# ------------------- Eye projection functions -------------------
def compute_left_eye_proj(landmarks, width, height):
    """Camera-facing left eye"""
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
    """Camera-facing right eye (optional)"""
    iris = landmarks[468]
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
try:
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
                proj_left, iris_coord_l, inner_coord_l, outer_coord_l = (
                    compute_left_eye_proj(face_landmarks.landmark, w, h)
                )
                debug_coords_left = {
                    "iris": iris_coord_l,
                    "inner": inner_coord_l,
                    "outer": outer_coord_l,
                }

                proj_right, iris_coord_r, inner_coord_r, outer_coord_r = (
                    compute_right_eye_proj(face_landmarks.landmark, w, h)
                )
                debug_coords_right = {
                    "iris": iris_coord_r,
                    "inner": inner_coord_r,
                    "outer": outer_coord_r,
                }
                break

        # ------------------- Apply EMA smoothing -------------------
        smoothed_proj_left = alpha * proj_left + (1 - alpha) * smoothed_proj_left
        smoothed_proj_right = alpha * proj_right + (1 - alpha) * smoothed_proj_right

        # ------------------- State update -------------------
        current_state = update_looking_state(smoothed_proj_left, current_state)

        try:
            if (
                current_state
                and ws.get_current_program_scene().current_program_scene_name
                != FOCUS_SCENE
            ):
                ws.set_current_program_scene(FOCUS_SCENE)
            elif (
                not current_state
                and ws.get_current_program_scene().current_program_scene_name
                != DEFAULT_SCENE
            ):
                ws.set_current_program_scene(DEFAULT_SCENE)
        except Exception as e:
            print("OBS Error:", e)

        # ------------------- Debug drawing -------------------
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

            cv2.circle(frame, (rx, ry), 3, (0, 255, 255), -1)
            cv2.circle(frame, (ri_x, ri_y), 2, (255, 0, 255), -1)
            cv2.circle(frame, (ro_x, ro_y), 2, (255, 0, 255), -1)
            cv2.line(frame, (ri_x, ri_y), (ro_x, ro_y), (255, 255, 0), 1)

        # ------------------- Status Text -------------------
        status_text = (
            f"LOOKING ({smoothed_proj_left:.2f})"
            if current_state
            else f"NOT LOOKING ({smoothed_proj_left:.2f})"
        )
        color = (0, 255, 0) if current_state else (0, 0, 255)
        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Eye Tracker Debug", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
except KeyboardInterrupt:
    print("Exiting")
finally:

    cap.release()
    cv2.destroyAllWindows()
    ws.set_current_program_scene(DEFAULT_SCENE)
    ws.disconnect()
