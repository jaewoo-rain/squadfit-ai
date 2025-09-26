# server.py
import logging
from typing import Set

from fastapi import FastAPI
from pydantic import BaseModel

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCRtpSender
from aiortc.contrib.media import MediaBlackhole, MediaRelay  # ★ relay 추가

# from ai import build_analyzed_track   # ★ 잠시 주석 처리하여 AI 파이프라인 우회

app = FastAPI()
pcs: Set[RTCPeerConnection] = set()
relay = MediaRelay()  # ★ 들어온 트랙을 재전송할 때 사용
logger = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

class SDP(BaseModel):
    sdp: str
    type: str

def _prefer_vp8_for(pc: RTCPeerConnection) -> None:
    """aiortc 1.7.0 호환: Sender capability로 codec preference 세팅"""
    try:
        caps = RTCRtpSender.getCapabilities("video")
        vp8 = [c for c in caps.codecs if c.mimeType.lower() == "video/vp8"]
        others = [c for c in caps.codecs if c not in vp8]
        prefs = vp8 + others if vp8 else caps.codecs
        for tr in pc.getTransceivers():
            if tr.kind == "video" and hasattr(tr, "setCodecPreferences"):
                tr.setCodecPreferences(prefs)
    except Exception as e:
        logger.warning(f"setCodecPreferences skip: {e}")

@app.post("/offer")
async def offer(sdp: SDP):
    try:
        rtc_config = RTCConfiguration(iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            # 필요 시 TURN 서버를 추가 (Spring에서 내려준 creds를 Flutter가 씀)
        ])
        pc = RTCPeerConnection(rtc_config)
        pcs.add(pc)
        logger.info(f"PC {id(pc)} created (total={len(pcs)})")

        @pc.on("connectionstatechange")
        async def on_state():
            logger.info(f"PC {id(pc)} state={pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            logger.info(f"PC {id(pc)} TRACK kind={track.kind} id={track.id}")
            if track.kind == "video":
                _prefer_vp8_for(pc)

                # ★ AI 우회: 들어온 트랙을 그대로 되돌려줌
                # analyzed = build_analyzed_track(track)    # (일단 주석)
                # pc.addTrack(analyzed)
                pc.addTrack(relay.subscribe(track))

            else:
                MediaBlackhole().addTrack(track)

        remote = RTCSessionDescription(sdp=sdp.sdp, type=sdp.type)
        await pc.setRemoteDescription(remote)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    except Exception as e:
        # ★ 500 원인 로그를 확실히 남김
        logger.exception("offer() failed")
        # 클라이언트가 읽기 쉬운 메시지로 돌려줌
        return app.responses.JSONResponse(
            status_code=500,
            content={"error": "offer_failed", "detail": str(e)},
        )

@app.get("/health")
def health():
    return {"ok": True, "pcs": len(pcs)}
