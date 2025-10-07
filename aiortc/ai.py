# ai.py
import asyncio
import json
import logging
import time
import math
import contextlib
from typing import Awaitable, Callable

import mediapipe as mp
from av import VideoFrame
from aiortc import MediaStreamTrack

log = logging.getLogger("ai")

SendJson = Callable[[str], asyncio.Future | None]
mp_hands = mp.solutions.hands

try:
    import orjson
    def dumps(obj) -> str:
        return orjson.dumps(obj).decode()
except Exception:
    def dumps(obj) -> str:
        return json.dumps(obj, separators=(",", ":"))

def _distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def start_hands_analyzer(video_track: MediaStreamTrack, send_json: SendJson) -> asyncio.Task:
    async def _runner():
        log.info("[AI] hands analyzer started")

        target_fps = 30.0
        min_interval = 1.0 / target_fps
        last_sent = 0.0

        PROCESS_SHORT_SIDE = 320
        mirror_x = True
        MAX_HANDS = 1

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        send_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=3)
        async def _sender():
            while True:
                msg = await send_queue.get()
                try:
                    await send_json(msg) if asyncio.iscoroutinefunction(send_json) else send_json(msg)
                except Exception as e:
                    log.debug("[AI] send_json error: %s", e)
                finally:
                    send_queue.task_done()
        sender_task = asyncio.create_task(_sender())

        score = 0
        prev_closed = False

        try:
            while True:
                frame: VideoFrame = await video_track.recv()
                W, H = frame.width, frame.height

                now = time.perf_counter()
                if now - last_sent < min_interval:
                    continue
                last_sent = now

                if W <= H:
                    proc_w = PROCESS_SHORT_SIDE
                    proc_h = int(round(H * (PROCESS_SHORT_SIDE / W)))
                else:
                    proc_h = PROCESS_SHORT_SIDE
                    proc_w = int(round(W * (PROCESS_SHORT_SIDE / H)))

                img_rgb = frame.to_ndarray(format="rgb24", width=proc_w, height=proc_h)
                img_rgb.flags.writeable = False
                results = hands.process(img_rgb)

                pts = []
                fist_closed = False

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:
                        for lm in hand_landmarks.landmark:
                            x = int(lm.x * W)
                            y = int(lm.y * H)
                            if mirror_x:
                                x = (W - 1) - x
                            pts.append({"x": x, "y": y})

                        # ----------- 주먹 판별 -------------
                        tip_ids = [4, 8, 12, 16, 20]
                        wrist = hand_landmarks.landmark[0]
                        avg_tip_dist = sum(
                            _distance(hand_landmarks.landmark[i], wrist) for i in tip_ids
                        ) / len(tip_ids)
                        # 경험적으로 0.15 이하이면 주먹
                        if avg_tip_dist < 0.3:
                            fist_closed = True

                # 주먹 → 펴짐 이벤트로 점수 증가
                if prev_closed and not fist_closed:
                    score += 1
                    log.info(f"[AI] Hand opened! Score={score}")
                prev_closed = fist_closed

                payload = {"size": {"w": W, "h": H}, "points": pts, "score": score}
                msg = dumps(payload)

                if send_queue.full():
                    with contextlib.suppress(Exception):
                        _ = send_queue.get_nowait()
                        send_queue.task_done()
                await send_queue.put(msg)

        except asyncio.CancelledError:
            log.info("[AI] hands analyzer cancelled")
        except Exception as e:
            log.exception("[AI] analyzer error: %s", e)
        finally:
            hands.close()
            sender_task.cancel()
            with contextlib.suppress(Exception):
                await sender_task

    return asyncio.create_task(_runner())
