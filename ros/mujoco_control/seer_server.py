#!/usr/bin/env python3
"""
A server script for the Seer model.

Usage: python seer_server.py \
        --model_path <checkpoint_path> \
        --vit_checkpoint_path <mae_pretrain_vit_checkpoint_path> \
        --port <port number>

Protocol (both directions: 4-byte big-endian length + pickle payload)
  request : {"cmd": "predict", "primary": <jpeg bytes>, "wrist": <jpeg bytes>, "state": [7 floats], "instruction": str}
            {"cmd": "reset"}
            {"cmd": "ping"}
  reply   : {"ok": True, "delta": [7 floats]}   # normalized delta action
            {"ok": True}                          # reset / ping
            {"ok": False, "error": str}
"""

import io
import os
import sys
import torch
import numpy as np
import pickle
import socket
import struct
import argparse
import functools
import traceback

from collections import deque
from PIL import Image as PILImage


# ==================== Wire Helpers ====================
def send_msg(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Peer closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack(">I", recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, n))


# ==================== Model Wrapper ====================
class Seer:
    def __init__(
        self,
        model_path,
        vit_checkpoint_path,
        seer_root,
        device="cuda",
        sequence_length=7,
        action_pred_steps=3,
        num_resampler_query=6,
        transformer_layers=24,
        use_ensembling=True,
        ensembling_temp=0.01,
        max_steps=600,
        bf16_vision_encoder=True,
    ):
        sys.path.insert(0, seer_root)
        import clip
        from models.seer_model import SeerAgent
        from utils.data_utils import preprocess_image, preprocess_text_calvin

        self.device = torch.device(device)
        self.history_len = sequence_length
        self.action_pred_steps = action_pred_steps
        self.use_ensembling = use_ensembling
        self.ensembling_temp = ensembling_temp
        self.max_steps = max_steps

        self.model = SeerAgent(
            finetune_type="real",
            clip_device=self.device,
            vit_checkpoint_path=vit_checkpoint_path,
            sequence_length=sequence_length,
            num_resampler_query=num_resampler_query,
            num_obs_token_per_image=9,
            calvin_input_image_size=224,
            patch_size=16,
            action_pred_steps=action_pred_steps,
            obs_pred=True,
            atten_only_obs=False,
            attn_robot_proprio_state=False,
            atten_goal=0,
            atten_goal_state=False,
            mask_l_obs_ratio=0.0,
            transformer_layers=transformer_layers,
            hidden_dim=384,
            transformer_heads=12,
            phase="evaluate",
            gripper_width=False,
        ).float()
        if bf16_vision_encoder and self.device.type == "cuda":
            self.model.vision_encoder.bfloat16()
        self.model.clip_model.requires_grad_(False)
        self.model.vision_encoder.requires_grad_(False)
        self.model = self.model.to(self.device)
        self.model._init_model_type()

        ckpt = torch.load(model_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[Model] missing keys: {len(missing)} e.g. {missing[:3]}")
        if unexpected:
            print(f"[Model] unexpected keys: {len(unexpected)} e.g. {unexpected[:3]}")
        self.model.eval()

        self.image_fn = functools.partial(preprocess_image, image_processor=self.model.image_processor)
        self.text_fn = functools.partial(preprocess_text_calvin, tokenizer=clip)
        self.reset()

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.wrist_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.timestep = 0
        self.all_time_actions = torch.zeros(
            self.max_steps, self.max_steps + self.action_pred_steps, 7, device=self.device
        )

    @torch.no_grad()
    def predict(self, primary_pil, wrist_pil, state7, instruction):
        img = self.image_fn([primary_pil.convert("RGB")]).unsqueeze(1)
        wrist = self.image_fn([wrist_pil.convert("RGB")]).unsqueeze(1)
        state = torch.as_tensor(np.asarray(state7, dtype=np.float32)).view(1, 1, 7)
        text = self.text_fn([instruction]).unsqueeze(1)

        self.img_queue.append(img.to(self.device))
        self.wrist_queue.append(wrist.to(self.device))
        self.state_queue.append(state.to(self.device))
        if len(self.text_queue) == 0:
            for _ in range(self.history_len):
                self.text_queue.append(text.to(self.device))

        image_primary = torch.cat(list(self.img_queue), dim=1)
        image_wrist = torch.cat(list(self.wrist_queue), dim=1)
        state_hist = torch.cat(list(self.state_queue), dim=1)
        text_tokens = torch.cat(list(self.text_queue), dim=1)
        num_step = image_primary.shape[1]

        if num_step < self.history_len:
            pad = self.history_len - num_step
            image_primary = torch.cat([image_primary, image_primary[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
            image_wrist = torch.cat([image_wrist, image_wrist[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
            state_hist = torch.cat([state_hist, state_hist[:, -1:].repeat(1, pad, 1)], dim=1)

        arm_action, gripper_action, _, _, _, _ = self.model(
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state_hist,
            text_token=text_tokens,
            action=torch.zeros(1, self.history_len, 7, device=self.device),
        )
        sel = num_step - 1 if num_step < self.history_len else -1

        if not self.use_ensembling:
            act = torch.cat([arm_action[0, sel, 0, :], gripper_action[0, sel, 0, :]], dim=-1)
        else:
            t = min(self.timestep, self.max_steps - 1)
            chunk = torch.cat([arm_action[:, sel], gripper_action[:, sel]], dim=-1)
            self.all_time_actions[t:t + 1, t:t + self.action_pred_steps] = chunk
            cands = self.all_time_actions[:, t]
            cands = cands[torch.all(cands != 0, dim=1)]
            w = np.exp(-self.ensembling_temp * np.arange(len(cands)))
            w = torch.from_numpy(w / w.sum()).to(self.device, dtype=cands.dtype).unsqueeze(1)
            act = (cands * w).sum(dim=0)

        act = act.float().cpu().numpy()
        act[6] = 1.0 if act[6] > 0.5 else -1.0
        self.timestep += 1
        return act.tolist()


# ===================== Server Loop =====================
def serve(model, host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[server] listening on {host}:{port}")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server] client connected from {addr}; resetting history")
        model.reset()
        try:
            while True:
                req = recv_msg(conn)
                cmd = req.get("cmd")
                try:
                    if cmd == "predict":
                        delta = model.predict(
                            PILImage.open(io.BytesIO(req["primary"])),
                            PILImage.open(io.BytesIO(req["wrist"])),
                            req["state"],
                            req["instruction"],
                        )
                        send_msg(conn, {"ok": True, "delta": delta})
                    elif cmd == "reset":
                        model.reset()
                        send_msg(conn, {"ok": True})
                    elif cmd == "ping":
                        send_msg(conn, {"ok": True})
                    else:
                        send_msg(conn, {"ok": False, "error": f"unknown cmd {cmd!r}"})
                except Exception as e:  # keep serving after a bad request
                    traceback.print_exc()
                    send_msg(conn, {"ok": False, "error": repr(e)})
        except (ConnectionError, OSError, EOFError):
            print("[server] client disconnected")
        finally:
            conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vit_checkpoint_path", required=True)
    ap.add_argument("--seer_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")  # loopback only; reached via SSH
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--no_ensembling", action="store_true")
    a = ap.parse_args()

    model = Seer(a.model_path, a.vit_checkpoint_path, a.seer_root, device=a.device,
              use_ensembling=not a.no_ensembling)
    print("[server] model loaded")
    serve(model, a.host, a.port)