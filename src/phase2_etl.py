import os, cv2, dlib, json, time, warnings, sys
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from scipy.fftpack import dct
from datetime import datetime, timedelta
from tqdm import tqdm
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore", category=FutureWarning)

# ── PATHS ─────────────────────────────────────────────────────────────────────
META_CSV       = "/home/hduser/deepfake-data/metadata/metadata_ffpp.csv"
LANDMARK_MODEL = "/home/hduser/deepfake-data/shape_predictor_68_face_landmarks.dat"
PARQUET_OUT    = "/home/hduser/deepfake-data/processed/parquet/features"
FEATURE_CSV    = "/home/hduser/deepfake-data/processed/features/feature_table.csv"
CHECKPOINT_CSV = "/home/hduser/deepfake-data/processed/features/checkpoint.csv"
LOG_PATH       = "/home/hduser/deepfake-thesis/logs/phase2_etl_log.json"
RANDOM_SEED    = 42
FRAME_STEP     = 5
MAX_FRAMES     = 100
BATCH_SIZE     = 50

LEFT_EYE  = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# ── ANSI COLOURS (safe on Ubuntu terminal) ────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def p(msg, flush=True):
    print(msg, flush=flush)

def banner(text):
    line = "─" * 60
    p(f"\n{CYAN}{BOLD}{line}{RESET}")
    p(f"{CYAN}{BOLD}  {text}{RESET}")
    p(f"{CYAN}{BOLD}{line}{RESET}")

# ── LOCAL FS HELPER ───────────────────────────────────────────────────────────
def local(path):
    return "file://" + path

# ── SPARK ─────────────────────────────────────────────────────────────────────
def build_spark():
    spark = SparkSession.builder \
        .appName("deepfake-phase2-etl") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .config("spark.hadoop.fs.default.name", "file:///") \
        .config("spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.hadoop.fs.hdfs.impl",
                "org.apache.hadoop.hdfs.DistributedFileSystem") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    p(f"{GREEN}Spark {spark.version} ready | "
      f"cores: {spark.sparkContext.defaultParallelism}{RESET}")
    return spark

# ── EAR ───────────────────────────────────────────────────────────────────────
def _ear(pts):
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_video_features(video_path, video_id, manipulation_type,
                            class_label, detector, predictor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_vals = []
    flow_mags       = []
    ear_vals        = []
    dct_low_vals    = []
    dct_high_vals   = []
    lm_disp_vals    = []

    prev_gray      = None
    prev_landmarks = None
    frames_read    = 0
    frames_detected = 0
    frame_idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_STEP != 0:
            continue
        if frames_read >= MAX_FRAMES:
            break
        frames_read += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(float(np.mean(gray)))

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_mags.append(float(np.mean(mag)))
        prev_gray = gray.copy()

        small = cv2.resize(gray, (64, 64)).astype(np.float32)
        D = dct(dct(small, axis=0), axis=1)
        dct_low_vals.append(float(np.mean(np.abs(D[:8,  :8]))))
        dct_high_vals.append(float(np.mean(np.abs(D[32:, 32:]))))

        faces = detector(gray, 0)
        if len(faces) > 0:
            frames_detected += 1
            shape = predictor(gray, faces[0])
            pts = np.array([[shape.part(i).x, shape.part(i).y]
                            for i in range(68)])
            ear_vals.append(float(
                (_ear(pts[LEFT_EYE]) + _ear(pts[RIGHT_EYE])) / 2.0))
            if prev_landmarks is not None:
                disp = np.mean(np.sqrt(
                    np.sum((pts - prev_landmarks)**2, axis=1)))
                lm_disp_vals.append(float(disp))
            prev_landmarks = pts.copy()

    cap.release()
    if frames_read == 0:
        return None

    def stats(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0, 0.0
        a = np.array(arr)
        return (float(np.mean(a)), float(np.std(a)),
                float(np.min(a)),  float(np.max(a)))

    bm,  bs,  bmin,  bmax  = stats(brightness_vals)
    fm,  fs,  fmin,  fmax  = stats(flow_mags)
    em,  es,  emin,  emax  = stats(ear_vals)
    dm,  ds,  dmin,  dmax  = stats(lm_disp_vals)
    dclm, dcls, _, _       = stats(dct_low_vals)
    dchm, dchs, _, _       = stats(dct_high_vals)

    return {
        "video_id":            video_id,
        "manipulation_type":   manipulation_type,
        "class_label":         class_label,
        "total_frames":        total_frames,
        "frames_sampled":      frames_read,
        "frames_with_face":    frames_detected,
        "face_detection_rate": round(frames_detected / frames_read, 4),
        "brightness_mean":     round(bm,   4),
        "brightness_std":      round(bs,   4),
        "brightness_min":      round(bmin, 4),
        "brightness_max":      round(bmax, 4),
        "optical_flow_mean":   round(fm,   4),
        "optical_flow_std":    round(fs,   4),
        "optical_flow_min":    round(fmin, 4),
        "optical_flow_max":    round(fmax, 4),
        "ear_mean":            round(em,   4),
        "ear_std":             round(es,   4),
        "ear_min":             round(emin, 4),
        "ear_max":             round(emax, 4),
        "landmark_disp_mean":  round(dm,   4),
        "landmark_disp_std":   round(ds,   4),
        "landmark_disp_min":   round(dmin, 4),
        "landmark_disp_max":   round(dmax, 4),
        "dct_low_mean":        round(dclm, 4),
        "dct_low_std":         round(dcls, 4),
        "dct_high_mean":       round(dchm, 4),
        "dct_high_std":        round(dchs, 4),
    }

# ── SMOKE TEST ────────────────────────────────────────────────────────────────
def run_smoke(df_meta, detector, predictor):
    banner("PHASE 2 SMOKE TEST  (1 video per manipulation type)")
    sample = df_meta.groupby("manipulation_type", group_keys=False).apply(
        lambda x: x.sample(n=1, random_state=RANDOM_SEED),
        include_groups=False)
    sample = df_meta.loc[sample.index]
    p(f"Videos selected: {len(sample)}\n")

    results = []
    for _, row in sample.iterrows():
        t = time.time()
        feats = extract_video_features(
            row["file_path"], row["video_id"],
            row["manipulation_type"], row["class_label"],
            detector, predictor)
        elapsed = round(time.time() - t, 2)

        if feats:
            col = GREEN
            tag = "OK  "
            results.append(feats)
            detail = (f"face_rate={feats['face_detection_rate']:.2f}  "
                      f"ear={feats['ear_mean']:.3f}  "
                      f"flow={feats['optical_flow_mean']:.3f}  "
                      f"dct_hi={feats['dct_high_mean']:.1f}")
        else:
            col = RED
            tag = "FAIL"
            detail = "could not open or decode video"

        p(f"{col}[{tag}]{RESET} "
          f"{row['video_id']:20s} | "
          f"{row['manipulation_type']:15s} | "
          f"{detail} | {elapsed}s")

    p(f"\n{GREEN if len(results)==len(sample) else RED}"
      f"Smoke test: {len(results)}/{len(sample)} passed{RESET}")
    return results

# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
def run_full(df_meta, detector, predictor):
    banner("PHASE 2 FULL EXTRACTION")

    # Resume support
    already_done = set()
    if os.path.exists(CHECKPOINT_CSV):
        ck = pd.read_csv(CHECKPOINT_CSV)
        already_done = set(ck["video_id"].tolist())
        p(f"{YELLOW}Resuming: {len(already_done)} videos already done, "
          f"skipping.{RESET}")

    remaining = df_meta[~df_meta["video_id"].isin(already_done)].reset_index(drop=True)
    total_remaining = len(remaining)
    p(f"Videos to process: {total_remaining} / {len(df_meta)}\n")

    results  = []
    failed   = []
    counts   = {}   # per-manipulation-type live count
    t_start  = time.time()

    # tqdm bar: shows percentage, count, elapsed, ETA, speed
    pbar = tqdm(
        total=total_remaining,
        desc="Extracting",
        unit="vid",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
        file=sys.stdout
    )

    for i, row in remaining.iterrows():
        t_vid = time.time()
        feats = extract_video_features(
            row["file_path"], row["video_id"],
            row["manipulation_type"], row["class_label"],
            detector, predictor)
        elapsed_vid = round(time.time() - t_vid, 1)

        mt = row["manipulation_type"]
        counts[mt] = counts.get(mt, 0) + 1

        if feats is None:
            failed.append(row["video_id"])
            pbar.set_postfix_str(
                f"{RED}FAIL{RESET} {row['video_id']} | "
                f"failed_total={len(failed)}", refresh=True)
        else:
            results.append(feats)
            pbar.set_postfix_str(
                f"face={feats['face_detection_rate']:.2f} "
                f"ear={feats['ear_mean']:.3f} "
                f"flow={feats['optical_flow_mean']:.3f} "
                f"failed={len(failed)}",
                refresh=True)

        pbar.update(1)

        # Detailed line every 25 videos (outside tqdm to not corrupt bar)
        if (pbar.n) % 25 == 0 and pbar.n > 0:
            elapsed_total = time.time() - t_start
            rate = pbar.n / elapsed_total
            eta_sec = (total_remaining - pbar.n) / rate if rate > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_sec)))
            done_pct = round(100 * pbar.n / total_remaining, 1)
            per_type = "  ".join(f"{k}:{v}" for k, v in counts.items())
            tqdm.write(
                f"  {CYAN}[{done_pct:5.1f}%]{RESET} "
                f"{pbar.n}/{total_remaining}  "
                f"ETA: {eta_str}  "
                f"failed: {len(failed)}  "
                f"per-type: {per_type}"
            )

        # Checkpoint flush every BATCH_SIZE videos
        if len(results) > 0 and len(results) % BATCH_SIZE == 0:
            _flush_checkpoint(results)
            results = []

    pbar.close()

    # Final flush
    if results:
        _flush_checkpoint(results)

    # Read full result
    df_all = pd.read_csv(CHECKPOINT_CSV)
    p(f"\n{GREEN}Extraction complete{RESET}: "
      f"{len(df_all)} succeeded | {len(failed)} failed")
    if failed:
        p(f"{RED}Failed video IDs: {failed}{RESET}")

    return df_all.to_dict("records"), failed

# ── CHECKPOINT HELPER ─────────────────────────────────────────────────────────
def _flush_checkpoint(new_rows):
    df_new = pd.DataFrame(new_rows)
    if os.path.exists(CHECKPOINT_CSV):
        df_prev = pd.read_csv(CHECKPOINT_CSV)
        df_new = pd.concat([df_prev, df_new], ignore_index=True)
    os.makedirs(os.path.dirname(CHECKPOINT_CSV), exist_ok=True)
    df_new.to_csv(CHECKPOINT_CSV, index=False)
    tqdm.write(f"  {YELLOW}Checkpoint: {len(df_new)} videos saved to disk{RESET}")

# ── WRITE OUTPUTS ─────────────────────────────────────────────────────────────
def write_outputs(results, df_meta, spark):
    banner("WRITING OUTPUTS")
    df_feats = pd.DataFrame(results)

    rate_map = dict(zip(df_feats["video_id"], df_feats["face_detection_rate"]))
    df_meta["face_detection_success_rate"] = df_meta["video_id"].map(rate_map)
    df_meta.to_csv(META_CSV, index=False)
    p(f"  Metadata updated  : {META_CSV}")

    os.makedirs(os.path.dirname(FEATURE_CSV), exist_ok=True)
    df_feats.to_csv(FEATURE_CSV, index=False)
    p(f"  Feature CSV       : {FEATURE_CSV}  shape={df_feats.shape}")

    p("\n  Class distribution:")
    dist_df = (df_feats
               .groupby(["manipulation_type", "class_label"])
               .size()
               .reset_index(name="count"))
    p(dist_df.to_string(index=False))

    p("\n  Feature statistics (mean ± std across all videos):")
    num_cols = [c for c in df_feats.columns
                if c not in ["video_id", "manipulation_type", "class_label"]]
    stats_df = df_feats[num_cols].describe().loc[["mean", "std"]]
    p(stats_df.T.to_string())

    os.makedirs(PARQUET_OUT, exist_ok=True)
    spark_df = spark.createDataFrame(df_feats)
    spark_df.write.mode("overwrite").parquet(local(PARQUET_OUT))
    row_check = spark.read.parquet(local(PARQUET_OUT)).count()
    p(f"\n  Parquet written   : {PARQUET_OUT}")
    p(f"  Parquet row check : {row_check} rows")

    p(f"\n{GREEN}Outputs written successfully.{RESET}")
    return df_feats

# ── LOG ───────────────────────────────────────────────────────────────────────
def write_log(n_success, failed, elapsed, mode):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log = {
        "timestamp":          datetime.now().isoformat(),
        "mode":               mode,
        "random_seed":        RANDOM_SEED,
        "frame_step":         FRAME_STEP,
        "max_frames_per_vid": MAX_FRAMES,
        "features_per_video": 27,
        "videos_succeeded":   n_success,
        "videos_failed":      len(failed) if isinstance(failed, list) else 0,
        "failed_ids":         failed if isinstance(failed, list) else [],
        "elapsed_seconds":    elapsed,
        "elapsed_minutes":    round(elapsed / 60, 2),
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    p(f"  Log saved         : {LOG_PATH}")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if not os.path.exists(LANDMARK_MODEL):
        raise FileNotFoundError(
            f"Landmark model not found: {LANDMARK_MODEL}\n"
            "Fix: wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            " && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")

    banner("PHASE 2 ETL  —  Deepfake Detection Thesis")
    p(f"Mode       : {BOLD}{mode.upper()}{RESET}")
    p(f"Frame step : every {FRAME_STEP}th frame")
    p(f"Max frames : {MAX_FRAMES} per video")
    p(f"Batch size : checkpoint every {BATCH_SIZE} videos")

    p("\nLoading dlib detector + predictor...")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL)
    p(f"{GREEN}dlib ready.{RESET}")

    df_meta = pd.read_csv(META_CSV)
    p(f"Metadata loaded: {len(df_meta)} videos")

    spark = build_spark()

    spark_meta = spark.read.csv(local(META_CSV), header=True, inferSchema=False)
    assert spark_meta.count() == len(df_meta), "Spark/pandas row count mismatch"
    p(f"Spark validation: {spark_meta.count()} rows confirmed")

    t0 = time.time()

    if mode == "smoke":
        results = run_smoke(df_meta, detector, predictor)
        failed  = []
    else:
        results, failed = run_full(df_meta, detector, predictor)

    elapsed = round(time.time() - t0, 1)

    if results:
        df_feats = write_outputs(results, df_meta, spark)
        write_log(len(results), failed, elapsed, mode)

        banner("SUMMARY")
        p(f"  Mode             : {mode.upper()}")
        p(f"  Videos processed : {GREEN}{len(results)}{RESET}")
        p(f"  Videos failed    : {RED if failed else GREEN}{len(failed)}{RESET}")
        p(f"  Total time       : {round(elapsed/60, 1)} minutes")
        p(f"  Feature CSV      : {FEATURE_CSV}")
        p(f"  Parquet          : {PARQUET_OUT}")
        p(f"  Log              : {LOG_PATH}")
        p(f"\n{GREEN}{BOLD}Phase 2 {mode.upper()} COMPLETE{RESET}\n")
    else:
        p(f"{RED}No results produced. Check video paths and landmark model.{RESET}")

    spark.stop()
