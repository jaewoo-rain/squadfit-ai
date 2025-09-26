# ai.py
import os
import time
import logging
from typing import Tuple

import cv2
import numpy as np
from av import VideoFrame
from aiortc import MediaStreamTrack

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai")

DRAW_COLOR = tuple(int(x) for x in os.getenv("AI_DRAW_COLOR", "0,255,0").split(","))  # B,G,R
DRAW_THICK = int(os.getenv("AI_DRAW_THICK", "3"))

class AnalyzedVideoTrack(MediaStreamTrack):
    """
    입력 영상(track)을 받아 간단한 '머리(얼굴) 추정 원'을 그린 뒤 다시 내보내는 예제 트랙.
    성능/안정성을 위해 프레임 레이트를 약간 제한할 수도 있음.
    """
    kind = "video"

    def __init__(self, source: MediaStreamTrack):
        super().__init__()  # base class 초기화
        self.source = source
        self._t0 = time.time()
        self._cnt = 0

    async def recv(self) -> VideoFrame:
        frame: VideoFrame = await self.source.recv()

        # av.VideoFrame -> numpy (BGR)
        img = frame.to_ndarray(format="bgr24")

        # === 매우 간단한 '머리 위치' 가짜 추정: 가운데 얼굴 높이 정도에 원을 그림 ===
        h, w, _ = img.shape
        cx, cy = w // 2, int(h * 0.6)
        radius = max(30, min(w, h) // 8)

        cv2.circle(img, (cx, cy), radius, DRAW_COLOR, DRAW_THICK)
        cv2.putText(img, "AI: head-tracking demo", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, DRAW_COLOR, 2, cv2.LINE_AA)

        # FPS 로그(1초 주기)
        self._cnt += 1
        now = time.time()
        if now - self._t0 >= 1.0:
            log.info("[AI] frames out/s = %d", self._cnt)
            self._cnt = 0
            self._t0 = now

        # numpy -> av.VideoFrame (pts/time_base 유지)
        out = VideoFrame.from_ndarray(img, format="bgr24")
        out.pts = frame.pts
        out.time_base = frame.time_base
        return out

def build_analyzed_track(source: MediaStreamTrack) -> MediaStreamTrack:
    """
    외부에서 호출하는 팩토리 함수.
    필요 시 여기에서 얼굴/포즈/손가락 등 실제 모델 로딩을 하고,
    그 핸들(예: tflite 인터프리터)을 AnalyzedVideoTrack에 전달해도 됨.
    """
    return AnalyzedVideoTrack(source)
