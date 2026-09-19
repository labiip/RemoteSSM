import os
import cv2
import glob
import re
import hashlib
import numpy as np
import torch
import logging
from torch.utils.data import Dataset

from face_tracker import RobustFaceTracker
from config import COMMON_CONFIG, PHYSIO_CONFIG, TRAIN_CONFIG, PROJECT_ROOT
from utils import DifferentiablePhysioExtractor, safe_torch_load


class TitanrPPGDataset(Dataset):
    def __init__(self, root_dir, seq_len=300, step=150, subject_ids=None, split_name="all"):
        self.root_dir = root_dir
        self.seq_len = int(seq_len)
        self.step = int(step)
        self.subject_ids = None if subject_ids is None else set(int(s) for s in subject_ids)
        self.split_name = split_name

        self.enable_time_alignment = bool(TRAIN_CONFIG.get("ENABLE_TIME_ALIGNMENT", True))
        self.use_bvp_derived_hr_label = bool(TRAIN_CONFIG.get("USE_BVP_DERIVED_HR_LABEL", True))
        self.enable_quality_filter = bool(TRAIN_CONFIG.get("ENABLE_QUALITY_FILTER", True))

        self.min_valid_hr_ratio = float(TRAIN_CONFIG.get("MIN_VALID_HR_RATIO", 0.60))
        self.min_bvp_band_energy_ratio = float(TRAIN_CONFIG.get("MIN_BVP_BAND_ENERGY_RATIO", 0.20))
        self.min_bvp_peak_ratio = float(TRAIN_CONFIG.get("MIN_BVP_PEAK_RATIO", 0.12))
        self.min_rgb_std = float(TRAIN_CONFIG.get("MIN_RGB_STD", 0.30))
        self.roi_size = int(TRAIN_CONFIG.get("FACE_ROI_SIZE", 32))
        # 缓存版本后追加 _real 以区分复数缓存
        self.cache_version = str(TRAIN_CONFIG.get("CACHE_VERSION", "v1")) + "_real"

        if self.subject_ids is None and self.split_name == "all":
            cache_dir_name = f"offline_cache_dir_{self.cache_version}_seq{seq_len}_step{step}"
        else:
            subject_tag = self._build_subject_tag()
            cache_dir_name = (
                f"offline_cache_dir_{self.cache_version}_"
                f"{self.split_name}_{subject_tag}_seq{seq_len}_step{step}"
            )

        # 缓存放项目目录下(避免数据集目录写权限问题)
        self.cache_dir = os.path.join(str(PROJECT_ROOT), "cache", cache_dir_name)

        self.cache_files = []
        if os.path.exists(self.cache_dir):
            self.cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "*.pt")))

        if len(self.cache_files) > 0:
            logging.info(
                f"🚀 检测到已构建的实数缓存目录 [{self.cache_dir}]，共找到 {len(self.cache_files)} 个样本切片，极速准备就绪！"
            )
        else:
            logging.info(f"⏳ 未检测到实数缓存 [{self.cache_dir}]，开始执行特征预处理 (仅需运行一次)...")
            os.makedirs(self.cache_dir, exist_ok=True)
            self._load_and_preprocess_all()
            self.cache_files = sorted(glob.glob(os.path.join(self.cache_dir, "*.pt")))

        # 构建人脸 ROI 内存映射: {folder_name: uint8 array (N_frames, H, W, 3)}
        # 注意: MMPD/PURE 每个受试者含多段视频(多个文件夹), 必须按 folder_name 索引,
        # 否则同一受试者的多段视频会互相覆盖, 导致 face_roi 与 BGR 信号错位.
        self.face_roi_dict = {}
        # 构建 BGR 信号内存映射: {folder_name: float32 array (N_frames, 3)}
        self.bgr_dict = {}
        folders = [
            f for f in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, f)) and not self._is_cache_dir(f)
        ]
        for folder_name in folders:
            subject_id = self._extract_subject_id(folder_name)
            if subject_id is None:
                continue
            if self.subject_ids is not None and subject_id not in self.subject_ids:
                continue
            roi_path = os.path.join(self.root_dir, folder_name, f"auto_face_roi_{self.roi_size}x{self.roi_size}.npy")
            if os.path.exists(roi_path):
                self.face_roi_dict[folder_name] = np.load(roi_path, allow_pickle=False)  # (N, H, W, 3) uint8
            # 加载 BGR 5-ROI 信号
            bgr_path = os.path.join(self.root_dir, folder_name, "auto_bgr_feat_5roi.npy")
            if os.path.exists(bgr_path):
                self.bgr_dict[folder_name] = np.load(bgr_path, allow_pickle=False).astype(np.float32)  # (N, 3)

    def _build_subject_tag(self):
        if not self.subject_ids:
            return "allsubjects"

        sorted_ids = sorted(self.subject_ids)
        subject_str = ",".join(str(x) for x in sorted_ids)
        short_hash = hashlib.md5(subject_str.encode("utf-8")).hexdigest()[:8]

        if len(sorted_ids) == 1:
            return f"s{sorted_ids[0]:02d}_{short_hash}"

        return f"s{sorted_ids[0]:02d}_to_s{sorted_ids[-1]:02d}_n{len(sorted_ids)}_{short_hash}"

    def _extract_subject_id(self, folder_name):
        match = re.search(r"(\d+)", folder_name)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _is_cache_dir(self, folder_name):
        return folder_name.startswith("offline_cache_dir")

    def _extract_features(self, video_path, feat_path, face_roi_path, roi_size=32):
        logging.info(f"⏳ 提取对齐 BGR 信号 + 人脸 ROI: {os.path.basename(video_path)} ...")
        cap = cv2.VideoCapture(video_path)
        tracker = RobustFaceTracker()
        bgr_signal = []
        face_rois = []

        last_valid_fused = np.array([127.0, 127.0, 127.0], dtype=np.float32)
        last_valid_roi = np.full((roi_size, roi_size, 3), 128, dtype=np.uint8)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tracker_result = tracker.process_frame(frame)
            roi_tuple = tracker_result[0] if tracker_result is not None else None
            face_box = tracker.get_face_box() if roi_tuple is not None else None

            if roi_tuple is not None:
                fh_c, fh_l, fh_r, ch_l, ch_r = roi_tuple
                fused = (
                    0.4 * np.array(fh_c[:3], dtype=np.float32)
                    + 0.1 * np.array(fh_l[:3], dtype=np.float32)
                    + 0.1 * np.array(fh_r[:3], dtype=np.float32)
                    + 0.2 * np.array(ch_l[:3], dtype=np.float32)
                    + 0.2 * np.array(ch_r[:3], dtype=np.float32)
                )
                last_valid_fused = fused

            if face_box is not None:
                fx, fy, fw, fh_box = face_box
                fx, fy = max(0, fx), max(0, fy)
                crop = frame[fy:fy+fh_box, fx:fx+fw]
                if crop.size > 0:
                    roi = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
                    last_valid_roi = roi

            bgr_signal.append(last_valid_fused)
            face_rois.append(last_valid_roi)

        cap.release()

        signal = np.array(bgr_signal, dtype=np.float32)
        face_roi_arr = np.array(face_rois, dtype=np.uint8)
        np.save(feat_path, signal)
        np.save(face_roi_path, face_roi_arr)
        return signal, face_roi_arr

    @staticmethod
    def _compute_chrom(rgb):
        """CHROM (de Haan & Jeanne 2013): 肤色正交投影提取 rPPG。

        对静态数据 (UBFC/PURE) 能从 5-ROI RGB 提取 r=0.68~0.90 的脉搏信号,
        远优于单 G 通道 (r=0.22~0.52)。rgb: (T,3) [0,1] 按 RGB 顺序。
        """
        x = rgb / (rgb.mean(0, keepdims=True) + 1e-8)
        xs = 3 * x[:, 0] - 2 * x[:, 1]
        ys = 1.5 * x[:, 0] + x[:, 1] - 1.5 * x[:, 2]
        xs = xs - xs.mean()
        ys = ys - ys.mean()
        alpha = np.std(xs) / (np.std(ys) + 1e-8)
        return xs - alpha * ys

    @staticmethod
    def _compute_pos(rgb):
        """POS (Wang et al. 2017): 平面正交肌肤投影 (规范版, Plane-Orthogonal-to-Skin)。

        在肤色正交平面 (垂直于 [1,1,1] 光照方向) 上投影, 同时抑制镜面反射与运动伪影:
          S1 = G - B
          S2 = -2R + G + B
          h  = S1 + alpha*S2,  alpha = std(S1)/std(S2)

        rgb: (T,3) [0,1] 按 RGB 顺序。
        """
        x = rgb / (rgb.mean(0, keepdims=True) + 1e-8)
        s1 = x[:, 1] - x[:, 2]                       # G - B
        s2 = -2.0 * x[:, 0] + x[:, 1] + x[:, 2]      # -2R + G + B
        s1 = s1 - s1.mean()
        s2 = s2 - s2.mean()
        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        return s1 + alpha * s2

    def _safe_interp(self, x_old, y_old, x_new):
        x_old = np.asarray(x_old, dtype=np.float64).reshape(-1)
        y_old = np.asarray(y_old, dtype=np.float64).reshape(-1)
        x_new = np.asarray(x_new, dtype=np.float64).reshape(-1)

        finite_mask = np.isfinite(x_old) & np.isfinite(y_old)
        x_old = x_old[finite_mask]
        y_old = y_old[finite_mask]

        if len(x_old) < 2:
            if len(y_old) == 0:
                return np.zeros_like(x_new, dtype=np.float32)
            return np.full_like(x_new, float(y_old[0]), dtype=np.float32)

        order = np.argsort(x_old)
        x_old = x_old[order]
        y_old = y_old[order]

        unique_x, unique_idx = np.unique(x_old, return_index=True)
        unique_y = y_old[unique_idx]

        if len(unique_x) < 2:
            return np.full_like(x_new, float(unique_y[0]), dtype=np.float32)

        y_new = np.interp(x_new, unique_x, unique_y)
        return y_new.astype(np.float32)

    def _fallback_resample_by_index(self, arr, target_len, fill_value=0.0):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        target_len = int(target_len)

        if target_len <= 0:
            return arr[:0].astype(np.float32)

        if arr.size == 0:
            return np.full(target_len, fill_value, dtype=np.float32)

        finite_mask = np.isfinite(arr)
        finite_count = int(np.sum(finite_mask))

        if finite_count == 0:
            return np.full(target_len, fill_value, dtype=np.float32)

        if finite_count == 1:
            return np.full(target_len, float(arr[finite_mask][0]), dtype=np.float32)

        src_axis = np.linspace(0.0, 1.0, len(arr), dtype=np.float64)
        dst_axis = np.linspace(0.0, 1.0, target_len, dtype=np.float64)

        valid_idx = np.where(finite_mask)[0].astype(np.float64)
        valid_pos = valid_idx / max(1.0, float(len(arr) - 1))
        valid_val = arr[finite_mask].astype(np.float64)

        repaired = np.interp(src_axis, valid_pos, valid_val)
        out = np.interp(dst_axis, src_axis, repaired)
        return out.astype(np.float32)

    def _align_gt_to_video_timeline(self, gt_bvp, gt_hr, gt_time, target_len):
        gt_bvp = np.asarray(gt_bvp, dtype=np.float32).reshape(-1)
        gt_hr = np.asarray(gt_hr, dtype=np.float32).reshape(-1)
        gt_time = np.asarray(gt_time, dtype=np.float32).reshape(-1)

        target_len = int(target_len)
        if target_len <= 0:
            return gt_bvp[:0], gt_hr[:0], gt_time[:0]

        min_raw_len = min(len(gt_bvp), len(gt_hr), len(gt_time))
        if min_raw_len < 2:
            aligned_bvp = self._fallback_resample_by_index(gt_bvp, target_len, fill_value=0.0)
            aligned_hr = self._fallback_resample_by_index(gt_hr, target_len, fill_value=0.0)
            aligned_time = np.arange(target_len, dtype=np.float32) / float(COMMON_CONFIG["FS"])
            return aligned_bvp, aligned_hr, aligned_time


        gt_bvp = gt_bvp[:min_raw_len]
        gt_hr = gt_hr[:min_raw_len]
        gt_time = gt_time[:min_raw_len]

        joint_finite_mask = (
            np.isfinite(gt_time)
            & np.isfinite(gt_bvp)
            & np.isfinite(gt_hr)
        )

        if int(np.sum(joint_finite_mask)) < 2:
            aligned_bvp = self._fallback_resample_by_index(gt_bvp, target_len, fill_value=0.0)
            aligned_hr = self._fallback_resample_by_index(gt_hr, target_len, fill_value=0.0)
            aligned_time = np.arange(target_len, dtype=np.float32) / float(COMMON_CONFIG["FS"])
            return aligned_bvp, aligned_hr, aligned_time

        gt_bvp = gt_bvp[joint_finite_mask]
        gt_hr = gt_hr[joint_finite_mask]
        gt_time = gt_time[joint_finite_mask]

        if len(gt_time) < 2:
            aligned_bvp = self._fallback_resample_by_index(gt_bvp, target_len, fill_value=0.0)
            aligned_hr = self._fallback_resample_by_index(gt_hr, target_len, fill_value=0.0)
            aligned_time = np.arange(target_len, dtype=np.float32) / float(COMMON_CONFIG["FS"])
            return aligned_bvp, aligned_hr, aligned_time

        order = np.argsort(gt_time.astype(np.float64))
        gt_time = gt_time[order]
        gt_bvp = gt_bvp[order]
        gt_hr = gt_hr[order]

        gt_time = gt_time - gt_time[0]
        gt_time = np.maximum.accumulate(gt_time)

        unique_time, unique_idx = np.unique(gt_time, return_index=True)
        gt_time = unique_time.astype(np.float32)
        gt_bvp = gt_bvp[unique_idx]
        gt_hr = gt_hr[unique_idx]

        if len(gt_time) < 2:
            aligned_bvp = self._fallback_resample_by_index(gt_bvp, target_len, fill_value=0.0)
            aligned_hr = self._fallback_resample_by_index(gt_hr, target_len, fill_value=0.0)
            aligned_time = np.arange(target_len, dtype=np.float32) / float(COMMON_CONFIG["FS"])
            return aligned_bvp, aligned_hr, aligned_time

        valid_duration = float(gt_time[-1])
        if not np.isfinite(valid_duration) or valid_duration <= 1e-6:
            aligned_bvp = self._fallback_resample_by_index(gt_bvp, target_len, fill_value=0.0)
            aligned_hr = self._fallback_resample_by_index(gt_hr, target_len, fill_value=0.0)
            aligned_time = np.arange(target_len, dtype=np.float32) / float(COMMON_CONFIG["FS"])
            return aligned_bvp, aligned_hr, aligned_time

        dst_time = np.linspace(0.0, valid_duration, target_len, dtype=np.float64)

        aligned_bvp = self._safe_interp(gt_time, gt_bvp, dst_time)
        aligned_hr = self._safe_interp(gt_time, gt_hr, dst_time)
        aligned_time = dst_time.astype(np.float32)

        return aligned_bvp, aligned_hr, aligned_time

    def _compute_bvp_quality(self, bvp_slice, fs):
        sig = np.asarray(bvp_slice, dtype=np.float32).reshape(-1)
        if len(sig) < 8 or not np.all(np.isfinite(sig)):
            return 0.0, 0.0

        sig = sig - np.mean(sig)
        sig_std = float(np.std(sig))
        if sig_std < 1e-8:
            return 0.0, 0.0

        sig = sig / (sig_std + 1e-8)
        window = np.hanning(len(sig)).astype(np.float32)
        sig = sig * window

        n_fft = max(1024, 1 << int(np.ceil(np.log2(len(sig)))))
        spec = np.abs(np.fft.rfft(sig, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))

        total_energy = float(np.sum(spec[1:])) + 1e-8
        band = PHYSIO_CONFIG["HR_BAND"]
        band_mask = (freqs >= band[0]) & (freqs <= band[1])

        if not np.any(band_mask):
            return 0.0, 0.0

        band_power = spec[band_mask]
        band_energy = float(np.sum(band_power))
        band_energy_ratio = band_energy / total_energy

        peak_ratio = float(np.max(band_power) / (band_energy + 1e-8)) if band_energy > 0 else 0.0
        return band_energy_ratio, peak_ratio

    def _is_valid_window(self, feat_slice, bvp_slice, hr_slice):
        if not self.enable_quality_filter:
            return True

        feat_slice = np.asarray(feat_slice, dtype=np.float32)
        bvp_slice = np.asarray(bvp_slice, dtype=np.float32)
        hr_slice = np.asarray(hr_slice, dtype=np.float32)

        if feat_slice.ndim != 2 or feat_slice.shape[0] != self.seq_len:
            return False
        if len(bvp_slice) != self.seq_len or len(hr_slice) != self.seq_len:
            return False

        if not (np.all(np.isfinite(feat_slice)) and np.all(np.isfinite(bvp_slice)) and np.all(np.isfinite(hr_slice))):
            return False

        rgb_std = float(np.mean(np.std(feat_slice, axis=0)))
        if rgb_std < self.min_rgb_std:
            return False

        valid_hr_mask = (hr_slice >= 40.0) & (hr_slice <= 180.0)
        valid_hr_ratio = float(np.mean(valid_hr_mask))
        if valid_hr_ratio < self.min_valid_hr_ratio:
            return False

        band_energy_ratio, peak_ratio = self._compute_bvp_quality(
            bvp_slice,
            fs=float(COMMON_CONFIG["FS"])
        )
        if band_energy_ratio < self.min_bvp_band_energy_ratio:
            return False
        if peak_ratio < self.min_bvp_peak_ratio:
            return False

        return True

    def _estimate_hr_from_bvp(self, bvp_slice, extractor, device, temperature=0.05):
        t_bvp = torch.tensor(bvp_slice, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            est_hr = extractor(
                t_bvp,
                freq_range=PHYSIO_CONFIG["HR_BAND"],
                temperature=temperature
            ).detach().cpu().squeeze(0)
        return est_hr

    def _load_and_preprocess_all(self):
        folders = [
            f for f in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, f)) and not self._is_cache_dir(f)
        ]
        folders.sort()
        selected_folders = []
        for folder_name in folders:
            subject_id = self._extract_subject_id(folder_name)
            if self.subject_ids is None or subject_id in self.subject_ids:
                selected_folders.append(folder_name)

        logging.info(f"[Dataset-{self.split_name}] 正在处理 {len(selected_folders)} 个受试者...")
        temp_data = []
        for folder_name in selected_folders:
            self._process_video(os.path.join(self.root_dir, folder_name), folder_name, temp_data)

        device = torch.device(COMMON_CONFIG["DEVICE"] if torch.cuda.is_available() else "cpu")
        extractor = DifferentiablePhysioExtractor(fs=COMMON_CONFIG["FS"]).to(device)
        saved_count, skipped_count = 0, 0

        for i, item in enumerate(temp_data):
            if i % 500 == 0 and i > 0:
                logging.info(f"[{self.split_name}] 预处理进度: {i}/{len(temp_data)}")

            feat_slice = item["bgr_features"].copy().astype(np.float32)
            bvp_slice = item["bvp"].copy().astype(np.float32)
            hr_slice = item["hr"].copy().astype(np.float32)

            if not self._is_valid_window(feat_slice, bvp_slice, hr_slice):
                skipped_count += 1
                continue

            # 复数信号输入重构: real=CHROM, imag=POS
            #   CHROM (de Haan 2013): 肤色正交投影, 抑制镜面反射
            #   POS (Wang 2017): 平面正交肌肤投影, 抑制运动伪影
            # 二者在肤色正交平面上互补, 取代原始的 G / (B-R)。实测带内 Pearson |r|
            # 从 ~0.24 提升到 ~0.65 (见 _diag_chrom_pos.py)
            rgb01 = feat_slice[:, ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
            real_part = self._compute_chrom(rgb01).astype(np.float32)   # CHROM
            imag_part = self._compute_pos(rgb01).astype(np.float32)     # POS

            # 逐窗标准化到单位方差: CHROM/POS 已去均值, 标准化使其尺度与数据增强
            # (scale=0.1, noise=0.01) 匹配, 避免弱脉冲被噪声淹没
            real_part = real_part / (float(np.std(real_part)) + 1e-8)
            imag_part = imag_part / (float(np.std(imag_part)) + 1e-8)

            t_real = torch.from_numpy(real_part).unsqueeze(-1)
            t_imag = torch.from_numpy(imag_part).unsqueeze(-1)
            t_bvp = torch.tensor(bvp_slice, dtype=torch.float32)

            if self.use_bvp_derived_hr_label:
                try:
                    robust_hr = self._estimate_hr_from_bvp(bvp_slice, extractor, device)
                except Exception:
                    valid_mask = (hr_slice >= 40.0) & (hr_slice <= 180.0)
                    valid_vals = hr_slice[valid_mask]
                    robust_hr = torch.tensor(np.median(valid_vals) if valid_vals.size > 0 else 0.0, dtype=torch.float32)
            else:
                valid_mask = (hr_slice >= 40.0) & (hr_slice <= 180.0)
                valid_vals = hr_slice[valid_mask]
                robust_hr = torch.tensor(np.median(valid_vals) if valid_vals.size > 0 else 0.0, dtype=torch.float32)
            robust_hr = torch.clamp(robust_hr.float(), min=40.0, max=180.0)

            sample_dict = {
                "input_real": t_real,
                "input_imag": t_imag,
                "target_bvp": t_bvp,
                "target_hr": robust_hr,
                "meta_subject": int(item["subject_id"]),
                "meta_folder": str(item["folder_name"]),
                "meta_start_idx": int(item["start_idx"]),
            }
            save_path = os.path.join(self.cache_dir, f"sample_{saved_count:06d}.pt")
            torch.save(sample_dict, save_path)
            saved_count += 1

        logging.info(f"💾 [{self.split_name}] 实数缓存完成！保留: {saved_count}, 过滤: {skipped_count}")


    def _process_video(self, folder_path, folder_name, temp_data):
        txt_files = [f for f in os.listdir(folder_path) if f.lower() == "ground_truth.txt"]
        if not txt_files:
            logging.warning(f"⚠️ 未找到 ground_truth.txt: {folder_name}，跳过。")
            return

        gt_path = os.path.join(folder_path, txt_files[0])
        feat_path = os.path.join(folder_path, "auto_bgr_feat_5roi.npy")
        face_roi_path = os.path.join(folder_path,
                                     f"auto_face_roi_{self.roi_size}x{self.roi_size}.npy")

        if not os.path.exists(feat_path) or not os.path.exists(face_roi_path):
            vid_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".avi")]
            if not vid_files:
                logging.warning(f"⚠️ 未找到 .avi 视频文件: {folder_name}，跳过。")
                return
            bgr_feat, face_roi_arr = self._extract_features(
                os.path.join(folder_path, vid_files[0]), feat_path, face_roi_path,
                roi_size=self.roi_size)
        else:
            bgr_feat = np.load(feat_path, allow_pickle=False)
            face_roi_arr = np.load(face_roi_path, allow_pickle=False)

        try:
            gt_data = np.loadtxt(gt_path)
            if gt_data.size == 0 or gt_data.ndim != 2:
                logging.warning(f"⚠️ {folder_name} 的 GT 数据格式不符合 UBFC 标准，已跳过。")
                return

            if gt_data.shape[0] == 3:
                gt_bvp = gt_data[0, :].flatten().astype(np.float32)
                gt_hr = gt_data[1, :].flatten().astype(np.float32)
                gt_time = gt_data[2, :].flatten().astype(np.float32)
            elif gt_data.shape[1] == 3:
                gt_bvp = gt_data[:, 0].flatten().astype(np.float32)
                gt_hr = gt_data[:, 1].flatten().astype(np.float32)
                gt_time = gt_data[:, 2].flatten().astype(np.float32)
            else:
                logging.warning(f"⚠️ {folder_name} 的 GT 数据维度异常: {gt_data.shape}，已跳过。")
                return

            if len(bgr_feat) < self.seq_len:
                return

            if self.enable_time_alignment:
                gt_bvp, gt_hr, gt_time = self._align_gt_to_video_timeline(
                    gt_bvp=gt_bvp,
                    gt_hr=gt_hr,
                    gt_time=gt_time,
                    target_len=len(bgr_feat),
                )
                min_len = min(len(gt_bvp), len(gt_hr), len(gt_time), len(bgr_feat))
            else:
                min_len = min(len(gt_bvp), len(bgr_feat), len(gt_hr), len(gt_time))

            if min_len < self.seq_len:
                return

            bgr_feat = bgr_feat[:min_len]
            gt_bvp = gt_bvp[:min_len]
            gt_hr = gt_hr[:min_len]
            gt_time = gt_time[:min_len]

            subject_id = self._extract_subject_id(folder_name)
            if subject_id is None:
                subject_id = -1

            for start_idx in range(0, min_len - self.seq_len + 1, self.step):
                end_idx = start_idx + self.seq_len
                temp_data.append({
                    "bgr_features": bgr_feat[start_idx:end_idx],
                    "bvp": gt_bvp[start_idx:end_idx],
                    "hr": gt_hr[start_idx:end_idx],
                    "time": gt_time[start_idx:end_idx],
                    "subject_id": subject_id,
                    "folder_name": folder_name,
                    "start_idx": start_idx,
                })

        except Exception as e:
            logging.error(f"❌ 解析视频夹 {folder_name} 数据失败: {e}")

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        import warnings
        file_path = self.cache_files[idx]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", category=UserWarning)
            sample = safe_torch_load(file_path, map_location="cpu")

        # 从内存映射中切片人脸 ROI: (seq_len, H, W, 3) uint8 → float32 / 255.0
        subj = int(sample.get("meta_subject", -1))
        folder = str(sample.get("meta_folder", ""))
        start = int(sample.get("meta_start_idx", 0))
        if folder in self.face_roi_dict:
            roi_arr = self.face_roi_dict[folder]
            end = start + self.seq_len
            if start >= 0 and end <= len(roi_arr):
                face_roi = torch.from_numpy(roi_arr[start:end].astype(np.float32) / 255.0)
            else:
                face_roi = torch.zeros(self.seq_len, self.roi_size, self.roi_size, 3)
        else:
            face_roi = torch.zeros(self.seq_len, self.roi_size, self.roi_size, 3)
        sample["face_roi"] = face_roi
        sample["subject_id"] = subj - 1 if subj >= 1 else 0  # map 1-34 → 0-33 for discriminator

        # 从内存映射中切片 BGR 3通道信号: (seq_len, 3) float32, 归一化 /128-1
        if folder in self.bgr_dict:
            bgr_arr = self.bgr_dict[folder]
            end = start + self.seq_len
            if start >= 0 and end <= len(bgr_arr):
                bgr_signal = torch.from_numpy(bgr_arr[start:end] / 128.0 - 1.0)
            else:
                bgr_signal = torch.zeros(self.seq_len, 3)
        else:
            bgr_signal = torch.zeros(self.seq_len, 3)
        sample["bgr_signal"] = bgr_signal
        return sample