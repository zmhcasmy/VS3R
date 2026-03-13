import argparse
import os
import cv2
import numpy as np
import glob
import csv
import fcntl
from tqdm import tqdm
from scipy.fftpack import fft
import torch

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[提示] 未检测到 lpips 库，LPIPS 指标将跳过。")

class GeoMetricAnalyzer:
    def __init__(self):
        self.MIN_MATCH_COUNT = 10
        self.ratio = 0.7
        self.thresh = 5.0
        self.detector = cv2.SIFT_create() 
        self.matcher = cv2.BFMatcher()

    def get_homography_stab_to_orig(self, img_orig, img_stab):
        if img_stab.shape != img_orig.shape:
            img_stab = cv2.resize(img_stab, (img_orig.shape[1], img_orig.shape[0]))

        kp1, des1 = self.detector.detectAndCompute(img_orig, None)
        kp2, des2 = self.detector.detectAndCompute(img_stab, None)

        if des1 is None or des2 is None: return None

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) > 1:
                m, n = m_n[0], m_n[1]
                if m.distance < self.ratio * n.distance:
                    good.append(m)

        if len(good) <= self.MIN_MATCH_COUNT: return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        try:
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.thresh)
            return M
        except:
            return None

    def get_frame_motion(self, prev_gray, curr_gray):
        kp1, des1 = self.detector.detectAndCompute(prev_gray, None)
        kp2, des2 = self.detector.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None: return None

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) > 1:
                m, n = m_n[0], m_n[1]
                if m.distance < self.ratio * n.distance:
                    good.append(m)

        if len(good) <= self.MIN_MATCH_COUNT: return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        try:
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.thresh)
            return M
        except:
            return None

    def generate_random_jitter_matrix(self):
        dx = np.random.uniform(-5, 5)
        dy = np.random.uniform(-5, 5)
        da = np.random.uniform(-1, 1) * np.pi / 180.0
        
        cos_a = np.cos(da)
        sin_a = np.sin(da)
        
        H = np.eye(3)
        H[0, 0] = cos_a
        H[0, 1] = -sin_a
        H[0, 2] = dx
        H[1, 0] = sin_a
        H[1, 1] = cos_a
        H[1, 2] = dy
        return H

    def findAreaPersentAfterWarp(self, homo, width, height):
        if homo is None: return None
        try:
            scaleRecovered = np.sqrt(homo[0, 1]**2 + homo[0, 0]**2)
            if scaleRecovered == 0: return 0.0
            return 1.0 / scaleRecovered
        except:
            return 0.0

    def findLowFrequencyPersentAfterWarp(self, trajectory):
        if len(trajectory) < 2: return 0.0 
        
        P_seq_t = []
        P_seq_r = []
        
        for M in trajectory:
            trans = np.sqrt(M[0, 2]**2 + M[1, 2]**2)
            theta = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            
            P_seq_t.append(trans)
            P_seq_r.append(theta)

        def calc_fft_ratio(seq):
            seq = np.array(seq)
            fft_res = np.fft.fft(seq)
            energy = np.abs(fft_res)**2
            
            if len(energy) < 2: return 0.0
            
            energy = np.delete(energy, 0)
            
            half_len = len(energy) // 2
            energy = energy[:half_len]
            
            total_e = np.sum(energy)
            if total_e == 0: return 1.0
            
            idx_limit = min(5, len(energy)) 
            low_freq_e = np.sum(energy[:idx_limit])
            
            return low_freq_e / total_e

        SS_t = calc_fft_ratio(P_seq_t)
        SS_r = calc_fft_ratio(P_seq_r)
        
        return (SS_t + SS_r) / 2.0

    def calculate_valid_ratio(self, img):
        if img is None: return 0.0
        
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            return 0.0
        
        c = max(cnts, key=cv2.contourArea)
        
        mask_final = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_final, [c], -1, 255, thickness=-1)
        valid_pixels = cv2.countNonZero(mask_final)
        total_pixels = h * w
        
        return valid_pixels / total_pixels

    def calc_epipolar_error(self, img1_bgr, img2_bgr):
        if img1_bgr is None or img2_bgr is None: return None
        kp1, des1 = self.detector.detectAndCompute(img1_bgr, None)
        kp2, des2 = self.detector.detectAndCompute(img2_bgr, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8: return None
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance: good.append(m)
        if len(good) < 8: return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        try:
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 1.0, 0.99)
            if F is None or mask is None: return None
            mask = mask.ravel().astype(bool)
            ones = np.ones((src_pts.shape[0], 1), dtype=np.float32)
            x1 = np.concatenate([src_pts, ones], axis=1)
            x2 = np.concatenate([dst_pts, ones], axis=1)
            Fx1 = (F @ x1.T).T
            Ftx2 = (F.T @ x2.T).T
            x2tFx1 = np.sum(x2 * Fx1, axis=1)
            denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
            denom = np.maximum(denom, 1e-8)
            sampson = (x2tFx1 ** 2) / denom
            all_mean = float(np.mean(sampson))
            if np.any(~mask):
                out_mean = float(np.mean(sampson[~mask]))
            else:
                out_mean = 0.0
            out_ratio = float(np.sum(~mask)) / float(len(mask))
            return all_mean, out_mean, out_ratio
        except: return None

class DeepMetricAnalyzer:
    def __init__(self, device='cuda'):
        self.device = device
        self.has_lpips = HAS_LPIPS
        if self.has_lpips:
            self.lpips_metric = lpips.LPIPS(net='alex').to(device)

    def preprocess_lpips(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_t = img_t * 2.0 - 1.0 
        return img_t.unsqueeze(0).to(self.device)

    def calc_lpips(self, img_orig_bgr, img_stab_bgr):
        if not self.has_lpips: return None
        try:
            with torch.no_grad(): return self.lpips_metric(self.preprocess_lpips(img_orig_bgr), self.preprocess_lpips(img_stab_bgr)).item()
        except: return None

    def calc_flow_warp_error(self, prev_bgr, curr_bgr):
        if prev_bgr is None or curr_bgr is None: return None
        h, w = prev_bgr.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        if scale < 1.0:
            new_w, new_h = int(w*scale), int(h*scale)
            prev_small = cv2.resize(prev_bgr, (new_w, new_h))
            curr_small = cv2.resize(curr_bgr, (new_w, new_h))
        else:
            prev_small, curr_small = prev_bgr, curr_bgr
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fh, fw = flow.shape[:2]
            grid_x, grid_y = np.meshgrid(np.arange(fw), np.arange(fh))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped_prev = cv2.remap(prev_small, map_x, map_y, cv2.INTER_LINEAR)
            diff = (warped_prev.astype(np.float32) - curr_small.astype(np.float32)) ** 2
            return np.mean(diff)
        except: return None

def evaluate_video_pair(orig_path, stab_path, geo_analyzer, deep_analyzer):
    cap_orig = cv2.VideoCapture(orig_path)
    cap_stab = cv2.VideoCapture(stab_path)

    if not cap_orig.isOpened() or not cap_stab.isOpened():
        print(f"[错误] 无法打开视频: {os.path.basename(orig_path)}")
        return None

    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_stab.get(cv2.CAP_PROP_FRAME_COUNT)))

    crop_scores = []
    warp_error_scores = []
    lpips_scores = [] 
    epipolar_all_scores = []
    epipolar_out_scores = []
    epipolar_out_ratio_scores = []
    
    trajectory = []
    current_H = np.eye(3)
    trajectory.append(current_H)
    
    prev_stab_gray = None
    prev_stab_frame = None 
    
    for i in range(total_frames):
        ret1, frame_orig = cap_orig.read()
        ret2, frame_stab = cap_stab.read()
        
        if not ret1 or not ret2: break

        frame_stab_resized = cv2.resize(frame_stab, (width, height))
        
        gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        gray_stab = cv2.cvtColor(frame_stab_resized, cv2.COLOR_BGR2GRAY)
        
        if prev_stab_gray is not None:
            H_inter = geo_analyzer.get_frame_motion(prev_stab_gray, gray_stab)
            if H_inter is not None:
                current_H = np.matmul(current_H, H_inter)
                trajectory.append(current_H.copy())
            
            if prev_stab_frame is not None:
                w_err = deep_analyzer.calc_flow_warp_error(prev_stab_frame, frame_stab_resized)
                if w_err is not None: warp_error_scores.append(w_err)
                
                epi_res = geo_analyzer.calc_epipolar_error(prev_stab_frame, frame_stab_resized)
                if epi_res is not None:
                    epi_all, epi_out, epi_out_ratio = epi_res
                    epipolar_all_scores.append(epi_all)
                    epipolar_out_scores.append(epi_out)
                    epipolar_out_ratio_scores.append(epi_out_ratio)

        prev_stab_gray = gray_stab
        prev_stab_frame = frame_stab_resized 

        H_so = geo_analyzer.get_homography_stab_to_orig(frame_orig, frame_stab_resized)
        
        if H_so is not None:
            c = geo_analyzer.findAreaPersentAfterWarp(H_so, width, height)
            
            if c is not None and c > 0:
                crop_scores.append(c)
        
        if i % 5 == 0:
            if np.mean(frame_stab_resized) > 5.0: 
                if deep_analyzer.has_lpips and H_so is not None:
                    try:
                        H_inv = np.linalg.inv(H_so)
                        
                        aligned_stab = cv2.warpPerspective(frame_stab_resized, H_inv, (width, height))
                        
                        mask = np.ones((height, width), dtype=np.uint8) * 255
                        mask_warped = cv2.warpPerspective(mask, H_inv, (width, height))
                        
                        frame_orig_masked = cv2.bitwise_and(frame_orig, frame_orig, mask=mask_warped)
                        aligned_stab_masked = cv2.bitwise_and(aligned_stab, aligned_stab, mask=mask_warped)
                        
                        l_score = deep_analyzer.calc_lpips(frame_orig_masked, aligned_stab_masked)
                        if l_score is not None: lpips_scores.append(l_score)
                    except Exception:
                        pass

    cap_orig.release()
    cap_stab.release()

    if not crop_scores: return None

    avg_crop = np.min([np.mean(crop_scores), 1.0]) 
    
    stab_score_pen = geo_analyzer.findLowFrequencyPersentAfterWarp(trajectory)
    
    avg_warp = np.mean(warp_error_scores) if warp_error_scores else None
    avg_lpips = np.mean(lpips_scores) if lpips_scores else None
    avg_epi_all = np.mean(epipolar_all_scores) if epipolar_all_scores else None
    avg_epi_out = np.mean(epipolar_out_scores) if epipolar_out_scores else None
    avg_epi_out_ratio = np.mean(epipolar_out_ratio_scores) if epipolar_out_ratio_scores else None
    
    return avg_crop, stab_score_pen, avg_warp, avg_lpips, avg_epi_all, avg_epi_out, avg_epi_out_ratio

def get_file_map(directory, extensions=['.mp4', '.avi', '.mov', '.mkv']):
    file_map = {}
    if not os.path.exists(directory):
        print(f"[错误] 目录不存在: {directory}")
        return file_map
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(directory, "**", f"*{ext.upper()}"), recursive=True))
    for f_path in files:
        rel_path = os.path.relpath(f_path, directory)
        file_id = os.path.splitext(rel_path)[0]
        file_map[file_id] = f_path
    return file_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_dir', type=str, default="/home/zmh/program/video_stablization/dataset/test_data/NUS_unstable_video_classified", help='原始视频根目录')
    parser.add_argument('--stab_dir', type=str, default="/home/zmh/program/video_stablization/dataset/test_data/stable_myself_vggt4d_stab-10/video", help='稳定视频根目录')
    parser.add_argument('--output_csv', type=str, default="/home/zmh/program/video_stablization/dataset/test_data/stable_myself_vggt4d_stab-10/eval.csv", help='结果追加输出的CSV文件路径')
    parser.add_argument('--method_name', type=str, default='Unknown', help='方法名称(用于CSV记录)')
    args = parser.parse_args()

    orig_map = get_file_map(args.orig_dir)
    stab_map = get_file_map(args.stab_dir)
    orig_ids = sorted(list(orig_map.keys()))
    matched_ids = sorted(set(orig_map.keys()) & set(stab_map.keys()))

    print(f"找到视频: 原视频 {len(orig_map)} 个, 稳定视频 {len(stab_map)} 个")
    print(f"严格ID匹配后可评估视频: {len(matched_ids)} 个")

    geo_analyzer = GeoMetricAnalyzer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deep_analyzer = DeepMetricAnalyzer(device=device)
    
    scores = {
        'crop_avg': [], 'stab_pen': [], 
        'warp': [], 'lpips': [], 'epi_all': [], 'epi_out': [], 'epi_out_ratio': []
    }
    category_scores = {}
    valid_videos = 0

    for vid_id in tqdm(matched_ids):
        target_path = stab_map[vid_id]
        
        category = os.path.dirname(vid_id)
        if not category: category = "Root"
        
        if category not in category_scores:
            category_scores[category] = {
                'crop_avg': [], 'stab_pen': [], 
                'warp': [], 'lpips': [], 'epi_all': [], 'epi_out': [], 'epi_out_ratio': []
            }
        
        orig_path = orig_map[vid_id]
        
        try:
            res = evaluate_video_pair(orig_path, target_path, geo_analyzer, deep_analyzer)
            if res is None: continue
            
            c_avg, s_pen, w, l, e_all, e_out, e_out_ratio = res
            
            def append_score(dct, key, val):
                if val is not None: dct[key].append(val)

            append_score(scores, 'crop_avg', c_avg)
            append_score(scores, 'stab_pen', s_pen)
            append_score(scores, 'warp', w)
            append_score(scores, 'lpips', l)
            append_score(scores, 'epi_all', e_all)
            append_score(scores, 'epi_out', e_out)
            append_score(scores, 'epi_out_ratio', e_out_ratio)
            
            append_score(category_scores[category], 'crop_avg', c_avg)
            append_score(category_scores[category], 'stab_pen', s_pen)
            append_score(category_scores[category], 'warp', w)
            append_score(category_scores[category], 'lpips', l)
            append_score(category_scores[category], 'epi_all', e_all)
            append_score(category_scores[category], 'epi_out', e_out)
            append_score(category_scores[category], 'epi_out_ratio', e_out_ratio)
            
            valid_videos += 1
            
        except Exception as e:
            print(f"[Err] {vid_id}: {e}")

    if valid_videos > 0:
        def print_m(dct, name, key, desc=""):
            vals = dct[key]
            if len(vals) > 0:
                print(f"{name:<30} : {np.mean(vals):.4f}  {desc}")
            else:
                print(f"{name:<30} : N/A")

        print("\n" + "="*80)
        print(f"【最终评估报告】 (共 {valid_videos} 个有效视频):")
        print("="*80)
        print_m(scores, "1. Stability (FFT)", 'stab_pen', "(越高越好, Target=1.0)")
        print_m(scores, "2. Crop Ratio (Inv Scale)", 'crop_avg', "(越高越好, Max=1.0)")
        print_m(scores, "3. Warp Error", 'warp', "(越低越好)")
        print_m(scores, "4. Epipolar Sampson (All)", 'epi_all', "(越低越好)")
        print_m(scores, "5. Epipolar Sampson (Outlier)", 'epi_out', "(越低越好)")
        print_m(scores, "6. Outlier Ratio", 'epi_out_ratio', "(越低越好)")
        print_m(scores, "7. LPIPS", 'lpips', "(越低越好)")
        print("="*80)
        
        sorted_cats = sorted(category_scores.keys())
        for cat in sorted_cats:
            print(f"\n>> 📂 Category: {cat}")
            print("-" * 60)
            print_m(category_scores[cat], "  Stability", 'stab_pen')
            print_m(category_scores[cat], "  Crop Ratio", 'crop_avg')
            print_m(category_scores[cat], "  Warp Error", 'warp')
            print_m(category_scores[cat], "  Epipolar Sampson (All)", 'epi_all')
            print_m(category_scores[cat], "  Epipolar Sampson (Outlier)", 'epi_out')
            print_m(category_scores[cat], "  Outlier Ratio", 'epi_out_ratio')
            print_m(category_scores[cat], "  LPIPS", 'lpips')
            print("-" * 60)

        if args.output_csv:
            file_exists = os.path.isfile(args.output_csv)
            
            def get_mean(dct, key):
                vals = dct.get(key, [])
                return np.mean(vals) if vals else ""

            header = [
                'Method', 'Category',
                'Stability', 'Crop Ratio', 
                'Warp Error', 'Epi (All)', 'Epi (Out)', 'Outlier Ratio', 
                'LPIPS'
            ]

            rows_to_write = []

            row_global = [
                args.method_name, 'Global',
                get_mean(scores, 'stab_pen'),
                get_mean(scores, 'crop_avg'),
                get_mean(scores, 'warp'),
                get_mean(scores, 'epi_all'),
                get_mean(scores, 'epi_out'),
                get_mean(scores, 'epi_out_ratio'),
                get_mean(scores, 'lpips')
            ]
            rows_to_write.append(row_global)

            for cat in sorted_cats:
                cat_scores = category_scores[cat]
                row_cat = [
                    args.method_name, cat,
                    get_mean(cat_scores, 'stab_pen'),
                    get_mean(cat_scores, 'crop_avg'),
                    get_mean(cat_scores, 'warp'),
                    get_mean(cat_scores, 'epi_all'),
                    get_mean(cat_scores, 'epi_out'),
                    get_mean(cat_scores, 'epi_out_ratio'),
                    get_mean(cat_scores, 'lpips')
                ]
                rows_to_write.append(row_cat)

            try:
                with open(args.output_csv, 'a', newline='', encoding='utf-8-sig') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    
                    is_empty = (f.tell() == 0)
                    
                    writer = csv.writer(f)
                    if is_empty:
                        writer.writerow(header)
                    writer.writerows(rows_to_write)
                    
                    fcntl.flock(f, fcntl.LOCK_UN)
                    
                print(f"\n[Success] 结果(含分类)已追加到 CSV 文件: {args.output_csv}")
            except Exception as e:
                print(f"\n[Error] 写入 CSV 失败: {e}")
    else:
        print("无有效结果。")
