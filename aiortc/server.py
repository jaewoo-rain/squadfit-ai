# server.py
import os
import time
import hmac
import base64
import hashlib
import logging
import uuid
import json
import asyncio
from typing import Set, Dict

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    RTCRtpSender,
    RTCIceCandidate,
    RTCDataChannel,
)
from aiortc.contrib.media import MediaBlackhole, MediaRelay

from ai import start_hands_analyzer  # ✅ 변경된 import

# ===== 기본 설정 =====
AI_MODE = os.getenv("AI_MODE", "points").lower()
TURN_HOST = os.getenv("TURN_HOST", "coturn")
TURN_SECRET = os.getenv("TURN_AUTH_SECRET", "devsecret")
TURN_TTL = int(os.getenv("TURN_TTL", "3600"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server")

app = FastAPI()
pcs: Set[RTCPeerConnection] = set()
relay = MediaRelay()

# 세션/채널/작업 맵
sessions: Dict[str, RTCPeerConnection] = {}
channels: Dict[str, RTCDataChannel] = {}
tasks: Dict[str, asyncio.Task] = {}


class SDP(BaseModel):
    sdp: str
    type: str


def _turn_credentials(ttl: int = TURN_TTL):
    exp = int(time.time()) + ttl
    username = str(exp)
    digest = hmac.new(TURN_SECRET.encode(), username.encode(), hashlib.sha1).digest()
    credential = base64.b64encode(digest).decode()
    return username, credential


def _prefer_vp8(pc: RTCPeerConnection) -> None:
    try:
        caps = RTCRtpSender.getCapabilities("video")
        vp8 = [c for c in caps.codecs if c.mimeType.lower() == "video/vp8"]
        others = [c for c in caps.codecs if c not in vp8]
        prefs = (vp8 + others) if vp8 else caps.codecs
        for tr in pc.getTransceivers():
            if tr.kind == "video" and hasattr(tr, "setCodecPreferences"):
                tr.setCodecPreferences(prefs)
    except Exception as e:
        log.warning("setCodecPreferences skipped: %s", e)


# ============== /offer ==============
@app.post("/offer")
async def offer(sdp: SDP):
    """
    클라이언트가 Offer를 보내면:
      - DataChannel("coords")를 서버가 '먼저' 생성(협상 포함)
      - answer SDP 반환 + sid 반환
      - 이후 /candidate 로 트릭클
    """
    try:
        user, passw = _turn_credentials()
        rtc_config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(
                    urls=[
                        f"turn:{TURN_HOST}:3478?transport=udp",
                        f"turn:{TURN_HOST}:3478?transport=tcp",
                        f"turns:{TURN_HOST}:5349?transport=tcp", 
                    ],
                    username=user,
                    credential=passw,
                ),
            ]
        )

        pc = RTCPeerConnection(rtc_config)
        pcs.add(pc)

        sid = str(uuid.uuid4())
        sessions[sid] = pc
        log.info("PC %s created sid=%s (total=%d)", id(pc), sid, len(pcs))

        # --- 서버가 먼저 DataChannel 생성: SDP에 m=application 보장
        dc = pc.createDataChannel("coords")
        channels[sid] = dc

        @dc.on("open")
        def on_dc_open():
            log.info("DC open sid=%s label=%s", sid, dc.label)
            try:
                dc.send(json.dumps({"hello": True, "ts": time.time()}))
            except Exception as e:
                log.warning("DC hello send failed sid=%s: %s", sid, e)

        @dc.on("close")
        def on_dc_close():
            log.info("DC close sid=%s", sid)

        # --- 혹시 클라이언트가 DataChannel을 만들 경우도 수신
        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            log.info("on_datachannel sid=%s label=%s", sid, channel.label)
            # 이미 서버가 만든 채널이 있으므로, 교체하지 않는다!
            # (서버는 server-created 채널로만 계속 전송)
            if sid not in channels:
                channels[sid] = channel

            @channel.on("open")
            def on_open():
                log.info("DC(open from client) sid=%s label=%s", sid, channel.label)
                try:
                    channel.send(json.dumps({"hello": True, "ts": time.time()}))
                except Exception as e:
                    log.warning("DC hello send failed sid=%s: %s", sid, e)

        @pc.on("connectionstatechange")
        async def on_state_change():
            log.info("PC %s state=%s sid=%s", id(pc), pc.connectionState, sid)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await _cleanup_session(sid)

        @pc.on("track")
        def on_track(track):
            log.info("PC %s TRACK kind=%s id=%s sid=%s",
                     id(pc), track.kind, track.id, sid)
            if track.kind == "video":
                subscribed = relay.subscribe(track)

                async def _send_json(txt: str):
                    ch = channels.get(sid)
                    if not ch:
                        return
                    if ch.readyState != "open":
                        return
                    try:
                        ch.send(txt)
                    except Exception as e:
                        log.warning("DC send failed sid=%s: %s", sid, e)
                        
                # 좌표 분석 워커 시작
                t = start_hands_analyzer(subscribed, _send_json)
                tasks[sid] = t
            else:
                MediaBlackhole().addTrack(track)

        # SDP 처리
        remote = RTCSessionDescription(sdp=sdp.sdp, type=sdp.type)
        t1 = time.time()
        await pc.setRemoteDescription(remote)
        log.info("setRemote done (sid=%s, %.1f ms)", sid, (time.time() - t1) * 1000)

        t2 = time.time()
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        log.info("create+setLocal answer done (sid=%s, %.1f ms)", sid, (time.time() - t2) * 1000)

        local = pc.localDescription
        assert local is not None

        # 디버깅: DataChannel m=application 포함 여부 확인
        log.info("ANSWER SDP (truncated 2k chars):\n%s", local.sdp[:2048])

        return {"sdp": local.sdp, "type": local.type, "sid": sid}

    except Exception as e:
        log.exception("offer() failed")
        return JSONResponse(
            status_code=500,
            content={"error": "offer_failed", "detail": str(e)},
        )


# ============== /candidate (trickle) ==============
def _is_loopback_candidate(cand_str: str) -> bool:
    if not cand_str:
        return False
    s = f" {cand_str} "
    return (" 127.0.0.1 " in s) or (" ::1 " in s) or (" 0:0:0:0:0:0:0:1 " in s)


def _normalize_candidate(c: dict) -> RTCIceCandidate:
    sdp_mid = c.get("sdpMid") or c.get("sdp_mid") or c.get("mid")
    sdp_mline_index = c.get("sdpMLineIndex") or c.get("sdp_mline_index") or c.get("mlineindex")
    cand_str = c.get("candidate") or c.get("candidateDescription") or ""
    return RTCIceCandidate(sdpMid=sdp_mid, sdpMLineIndex=sdp_mline_index, candidate=cand_str)


@app.post("/candidate")
async def add_candidate(payload: dict = Body(...)):
    try:
        sid = payload.get("sid")
        if not sid:
            return JSONResponse(status_code=400, content={"error": "missing_sid"})
        pc = sessions.get(sid)
        if not pc:
            return JSONResponse(status_code=404, content={"error": "invalid_sid"})

        if payload.get("endOfCandidates"):
            log.info("sid=%s endOfCandidates received", sid)
            return {"ok": True}

        cand = payload.get("candidate")
        if not cand:
            return JSONResponse(status_code=400, content={"error": "missing_candidate"})

        cand_str = cand.get("candidate") or ""
        if _is_loopback_candidate(cand_str):
            log.info("sid=%s skip loopback candidate: %s", sid, cand_str)
            return {"ok": True}

        await pc.addIceCandidate(_normalize_candidate(cand))
        return {"ok": True}
    except Exception as e:
        log.exception("add_candidate failed")
        return JSONResponse(status_code=500, content={"error": "candidate_failed", "detail": str(e)})


# ============== 기타 ==============
@app.get("/health")
def health():
    return {
        "ok": True,
        "pcs": len(pcs),
        "sessions": len(sessions),
        "tasks": len(tasks),
        "mode": AI_MODE,
    }


async def _cleanup_pc(pc: RTCPeerConnection):
    if pc in pcs:
        pcs.discard(pc)
    try:
        await pc.close()
    except Exception:
        pass


async def _cleanup_session(sid: str):
    # 워커 종료
    t = tasks.pop(sid, None)
    if t:
        t.cancel()
        try:
            await t
        except Exception:
            pass
    # 채널 종료
    ch = channels.pop(sid, None)
    if ch:
        try:
            ch.close()
        except Exception:
            pass
    # PC 종료
    pc = sessions.pop(sid, None)
    if pc:
        await _cleanup_pc(pc)


@app.on_event("shutdown")
async def on_shutdown():
    for sid in list(sessions.keys()):
        try:
            await _cleanup_session(sid)
        except Exception:
            pass
    pcs.clear()
