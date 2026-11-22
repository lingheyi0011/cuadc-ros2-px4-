# -*- coding: utf-8 -*-
"""
yolov5_shim.py
--------------
兼容垫片：模拟 pip 包 `yolov5` 的 `YOLOv5` 接口，但内部使用 YOLOv5 v7.0 源码
（/mnt/sda/yolov5-7.0）的 DetectMultiBackend 加载模型。

背景：
  - best.pt 是用 YOLOv5 v7.0 训练的（含 models.yolo.DetectionModel），
    pip 包 yolov5(6.0.7) 无法反序列化该 ckpt。
  - 原节点代码用 `from yolov5 import YOLOv5` + `self.yolov5.predict(image)`，
    并访问返回对象的 `.pred[0]` 和 `.names`。
  - 本 shim 提供完全相同的接口，节点业务代码无需改动。

用法：
  把节点里的
      from yolov5 import YOLOv5
  改成
      from yolov5_shim import YOLOv5

依赖：
  - /mnt/sda/yolov5-7.0 源码目录（含 models/ utils/）
  - torch / torchvision（已在 /mnt/sda/pylibs）
"""
import os
import sys
import warnings

import numpy as np
import torch

# ---- 定位 YOLOv5 v7.0 源码目录 ----
_YOLO_SRC_CANDIDATES = [
    os.environ.get("YOLOV5_V7_DIR", ""),
    "/mnt/sda/yolov5-7.0",
]


def _ensure_yolo_src():
    for p in _YOLO_SRC_CANDIDATES:
        if p and os.path.isdir(os.path.join(p, "models")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p
    raise RuntimeError(
        "找不到 YOLOv5 v7.0 源码目录（需要含 models/ utils/）。"
        "请设置环境变量 YOLOV5_V7_DIR 或放到 /mnt/sda/yolov5-7.0"
    )


# ---- 某些 v7.0 文件 import IPython.display，但没有 IPython 时装个桩 ----
def _stub_ipython_if_needed():
    try:
        import IPython.display  # noqa: F401
    except Exception:
        import types
        mod_ip = types.ModuleType("IPython")
        mod_disp = types.ModuleType("IPython.display")
        mod_disp.display = lambda *a, **k: None
        mod_ip.display = mod_disp
        sys.modules["IPython"] = mod_ip
        sys.modules["IPython.display"] = mod_disp


class _Detections:
    """模拟 pip yolov5 的 Detections 返回对象（只实现节点用到的字段）。"""

    def __init__(self, pred, names, imgs):
        self.pred = pred          # list[Tensor]，每个元素 N x 6: x1,y1,x2,y2,conf,cls
        self.names = names        # dict{int: str}
        self.imgs = imgs          # list[np.ndarray] 原图（BGR）

    def __len__(self):
        return len(self.pred)

    def __iter__(self):
        return iter(self.pred)


class YOLOv5:
    """与 pip 包 yolov5.YOLOv5 兼容的封装，内部用 v7.0 DetectMultiBackend。"""

    def __init__(self, model_path, device=None, load_on_init=True,
                 imgsz=640, conf_thres=0.25, iou_thres=0.45, fp16=True,
                 max_det=1000):
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)

        # device 解析
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            dev = torch.device(device if device != "cuda" else "cuda:0")
        else:
            dev = device
        self.device = dev
        self.fp16 = bool(fp16) and dev.type == "cuda"

        self.model = None
        self.stride = 32
        self.names = {}

        if load_on_init:
            self.load_model()

    def load_model(self):
        _ensure_yolo_src()
        _stub_ipython_if_needed()

        # 延迟 import，确保 sys.path 已包含 v7 源码
        from models.common import DetectMultiBackend  # type: ignore

        if not os.path.exists(self.model_path):
            raise FileNotFoundError("模型文件不存在: %s" % self.model_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DetectMultiBackend(self.model_path, device=self.device,
                                       dnn=False, data=None, fp16=self.fp16)
        model.eval()
        try:
            model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))
        except Exception:
            pass
        self.model = model
        self.stride = int(model.stride)

        # ---- 类别名处理 ----
        # TensorRT engine 不携带类别名，DetectMultiBackend 会给出假的 class0..classN。
        # 策略：若是 .engine，优先从同目录的 best.pt 读取真实 names；否则用模型自带。
        self.names = {}
        is_engine = str(self.model_path).lower().endswith((".engine", ".trt"))
        if is_engine:
            self.names = self._load_names_from_pt()
        if not self.names:
            raw = model.names
            self.names = raw if isinstance(raw, dict) else {i: n for i, n in enumerate(raw)}
        return self.model

    def _load_names_from_pt(self):
        """从同目录 best.pt 读取真实类别名（engine 无类名时用）。"""
        cand = os.path.join(os.path.dirname(self.model_path), "best.pt")
        if not os.path.exists(cand):
            return {}
        try:
            import torch as _torch
            ckpt = _torch.load(cand, map_location="cpu")
            m = ckpt.get("model") if isinstance(ckpt, dict) else None
            nm = getattr(m, "names", None) if m is not None else None
            if nm is None and isinstance(ckpt, dict):
                nm = ckpt.get("names")
            if isinstance(nm, dict):
                return {int(k): str(v) for k, v in nm.items()}
            if isinstance(nm, (list, tuple)):
                return {i: str(n) for i, n in enumerate(nm)}
        except Exception:
            pass
        return {}

    @staticmethod
    def _letterbox(im, new_shape=640, stride=32, auto=True):
        # 内联 letterbox，避免依赖 utils 版本差异
        import cv2
        shape = im.shape[:2]  # h, w
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not auto:
            r = min(r, 1.0)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im

    def predict(self, image, size=None, augment=False):
        """兼容 pip 包签名：image 可为单张 ndarray(BGR) 或 list。"""
        assert self.model is not None, "模型未加载"
        from utils.general import non_max_suppression  # type: ignore

        sz = int(size) if size else self.imgsz
        single = isinstance(image, np.ndarray)
        imgs = [image] if single else list(image)

        tensors = []
        for im in imgs:
            im0 = self._letterbox(im, sz, stride=self.stride, auto=True)
            im0 = im0[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
            im0 = np.ascontiguousarray(im0)
            t = torch.from_numpy(im0).to(self.device)
            t = t.float() / 255.0
            tensors.append(t)
        batch = torch.stack(tensors, 0)

        preds = []
        with torch.no_grad():
            out = self.model(batch)
            out = non_max_suppression(out, self.conf_thres, self.iou_thres,
                                      max_det=self.max_det)
        for det in out:
            if det is None or len(det) == 0:
                preds.append(torch.zeros((0, 6), device=self.device))
            else:
                preds.append(det)  # 已是 N x 6: x1,y1,x2,y2,conf,cls

        return _Detections(preds, self.names, imgs)
