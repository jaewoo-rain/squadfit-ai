# ai_local.py
import cv2
import time
import json
import logging
from typing import Callable, Awaitable, Optional

import mediapipe as mp

log = logging.getLogger("ai_local")
logging.basicConfig(level=logging.INFO)

# ===== 직렬화 (orjson 우선) =====
try:
    import orjson
    def dumps(obj) -> str:
        return orjson.dumps(obj).decode()
except Exception:
    def dumps(obj) -> str:
        return json.dumps(obj, separators=(",", ":"))

# ===== MediaPipe 준비 =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def default_send_json(txt: str):
    """샘플: JSON을 그냥 콘솔로 출력"""
    print(txt, flush=True)


def run_local_camera(
    cam_index: int = 0,
    send_json: Optional[Callable[[str], None]] = None,
    show_window: bool = True,
):
    """
    노트북 카메라에서 프레임을 읽어 실시간 손 좌표(JSON) 출력.
    - cam_index: 기본 카메라 0
    - send_json: JSON 문자열을 전송하는 콜백 (기본: 콘솔 출력)
    - show_window: 창에 랜드마크 그려서 보여줄지 여부
    """
    if send_json is None:
        send_json = default_send_json

    # 카메라 열기 (해상도 힌트 주기 - 실제 적용 여부는 드라이버/OS에 따라 다름)
    cap = cv2.VideoCapture(cam_index)
    # 권장: 낮은 해상도 힌트
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. 다른 앱이 점유 중인지 확인하세요.")

    # 처리용 목표 FPS (너무 높이면 CPU만 바쁨)
    target_fps = 45.0
    min_interval = 1.0 / target_fps
    last_sent = 0.0

    # 처리용 다운스케일(짧은 변 기준)
    PROCESS_SHORT_SIDE = 320
    mirror_x = True  # 좌표만 미러링

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    log.info("[AI] local camera hands analyzer started")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                log.warning("프레임을 읽지 못했습니다.")
                continue

            H, W = frame_bgr.shape[:2]

            # FPS 상한 (드롭 방식)
            now = time.perf_counter()
            if now - last_sent < min_interval:
                # 최신 프레임만 처리하기 위해 드롭
                continue
            last_sent = now

            # 다운스케일 사이즈 계산(비율 유지)
            if W <= H:
                proc_w = PROCESS_SHORT_SIDE
                proc_h = int(round(H * (PROCESS_SHORT_SIDE / W)))
            else:
                proc_h = PROCESS_SHORT_SIDE
                proc_w = int(round(W * (PROCESS_SHORT_SIDE / H)))

            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 다운스케일 (CPU 리사이즈)
            if (proc_w, proc_h) != (W, H):
                frame_rgb_small = cv2.resize(frame_rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            else:
                frame_rgb_small = frame_rgb

            frame_rgb_small.flags.writeable = False
            results = hands.process(frame_rgb_small)

            points = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        # MediaPipe는 정규화 좌표(0~1) → 원본 크기로 환산
                        x = int(lm.x * W)
                        y = int(lm.y * H)
                        if mirror_x:
                            x = (W - 1) - x
                        points.append({"x": x, "y": y})

            payload = {"size": {"w": W, "h": H}, "points": points}
            try:
                send_json(dumps(payload))
            except Exception as e:
                log.warning("send_json 실패: %s", e)

            # 화면 표시(옵션)
            if show_window:
                # 그리기용으로 좌우 반전된 프리뷰를 보고 싶다면:
                draw_bgr = cv2.flip(frame_bgr, 1) if mirror_x else frame_bgr.copy()

                if results.multi_hand_landmarks:
                    # 그리기는 다운스케일 좌표라, 그릴 때는 다운스케일 이미지 위에 그리는 게 정확
                    # 여기서는 간단히 원본 위에 그릴 때 좌표만 이용해 원형 찍기
                    for p in points:
                        cv2.circle(draw_bgr, (p["x"], p["y"]), 3, (0, 255, 0), -1)

                cv2.imshow("Hands (press 'q' to quit)", draw_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        hands.close()
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        log.info("[AI] local camera hands analyzer stopped")


if __name__ == "__main__":
    # 그냥 실행하면 콘솔로 JSON 쏘면서 창에 랜드마크 보여줌
    run_local_camera()
