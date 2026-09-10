import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent
MMW_ROOT = ROOT / "MMW-HAT" / "MMW-HAT-Release"
EXAMPLE2_ROOT = MMW_ROOT / "example_2_track"
if str(MMW_ROOT) not in sys.path:
    sys.path.insert(0, str(MMW_ROOT))
if str(EXAMPLE2_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE2_ROOT))

from signal_proc import SigProc
from utility.helper import find_setting_in_directory, parse_full_frame, parse_radar_cfg, read_uint12, split_samples
from utility.mmw_cube_proc_v0 import CubeProcessor

RD_CONFIG_DIR = MMW_ROOT / "radar_config" / "config_3rx_3m"
TRACK_CONFIG_DIR = MMW_ROOT / "radar_config" / "config_track"
TRACK_PROCESSING_CONFIG = EXAMPLE2_ROOT / "config" / "processing_config.json"
MAP_NAMES = ("rd", "ar", "ad", "xytrack")
THREE_MAP_NAMES = ("rd", "ar", "ad")


def load_setting(config_dir):
    with open(find_setting_in_directory(str(config_dir)), "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_mmw_frames(bin_path):
    with open(bin_path, "rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated MMW-HAT frame header in {bin_path}")
            data_length = int.from_bytes(header[8:12], byteorder="little", signed=False)
            payload = handle.read(data_length)
            if len(payload) != data_length:
                raise ValueError(f"Truncated MMW-HAT frame payload in {bin_path}")
            parsed = parse_full_frame(header + payload)
            if parsed is None:
                raise ValueError(f"Invalid MMW-HAT frame in {bin_path}")
            yield parsed


def _payload_sample_count(setting):
    params = parse_radar_cfg(setting)
    return params["num_chirps_per_frame"] * params["num_samples_per_chirp"] * params["num_antennas"]


def _payload_to_track_frame(payload, radar_params):
    adc = read_uint12(payload)
    samples = split_samples(
        adc,
        1,
        radar_params["num_chirps_per_frame"],
        radar_params["num_samples_per_chirp"],
        radar_params["num_antennas"],
    )
    return np.transpose(samples[0], (2, 0, 1))


def acquire_four_maps(bin_path, rd_config_dir=RD_CONFIG_DIR, track_config_dir=TRACK_CONFIG_DIR):
    rd_setting = load_setting(rd_config_dir)
    track_setting = load_setting(track_config_dir)
    rd_params = parse_radar_cfg(rd_setting)
    track_params = parse_radar_cfg(track_setting)
    cube = CubeProcessor(rd_setting, num_azimuth_bin=16, num_elevation_bin=16)
    expected_rd = _payload_sample_count(rd_setting)
    expected_track = _payload_sample_count(track_setting)
    xy_params = track_params if expected_track == expected_rd else rd_params
    tracker = SigProc(str(TRACK_PROCESSING_CONFIG), xy_params)
    maps = {name: [] for name in MAP_NAMES}

    for _, _, _, payload in iter_mmw_frames(bin_path):
        payload_samples = len(payload) * 2 // 3
        if payload_samples != expected_rd:
            raise ValueError(
                f"Radar payload has {payload_samples} samples; config_3rx_3m expects {expected_rd}"
            )
        cube.process_raw_data(list(payload))
        maps["rd"].append(cube.vis_2d("Doppler", "Range").astype(np.float32))
        maps["ar"].append(cube.vis_2d("Azimuth", "Range").astype(np.float32))
        maps["ad"].append(cube.vis_2d("Azimuth", "Doppler").astype(np.float32))
        track_frame = _payload_to_track_frame(payload, xy_params)
        _, _, plot = tracker.update(track_frame)
        maps["xytrack"].append(np.asarray(plot["map"], dtype=np.float32))

    if not maps["rd"]:
        raise ValueError(f"No MMW-HAT frames found in {bin_path}")
    return {name: np.stack(values) for name, values in maps.items()}, float(rd_params["frame_rate"])


def acquire_three_maps(bin_path, rd_config_dir=RD_CONFIG_DIR):
    rd_setting = load_setting(rd_config_dir)
    rd_params = parse_radar_cfg(rd_setting)
    cube = CubeProcessor(rd_setting, num_azimuth_bin=16, num_elevation_bin=16)
    expected = _payload_sample_count(rd_setting)
    maps = {name: [] for name in THREE_MAP_NAMES}

    for _, _, _, payload in iter_mmw_frames(bin_path):
        payload_samples = len(payload) * 2 // 3
        if payload_samples != expected:
            raise ValueError(
                f"Radar payload has {payload_samples} samples; config_3rx_3m expects {expected}"
            )
        cube.process_raw_data(list(payload))
        maps["rd"].append(cube.vis_2d("Doppler", "Range").astype(np.float32))
        maps["ar"].append(cube.vis_2d("Azimuth", "Range").astype(np.float32))
        maps["ad"].append(cube.vis_2d("Azimuth", "Doppler").astype(np.float32))

    if not maps["rd"]:
        raise ValueError(f"No MMW-HAT frames found in {bin_path}")
    return {name: np.stack(values) for name, values in maps.items()}, float(rd_params["frame_rate"])


def resize_map_sequence(sequence, output_size=(64, 64)):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3:
        raise ValueError(f"Expected map sequence shaped (time, height, width), got {sequence.shape}")
    height_scale = output_size[0] / sequence.shape[1]
    width_scale = output_size[1] / sequence.shape[2]
    return zoom(sequence, (1.0, height_scale, width_scale), order=1).astype(np.float32)


def resample_map_sequence(sequence, source_rate, target_rate):
    sequence = np.asarray(sequence, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sampling rates must be positive")
    if len(sequence) < 2:
        return sequence.copy()
    duration = (len(sequence) - 1) / source_rate
    output_count = max(2, int(round(duration * target_rate)) + 1)
    source_positions = np.linspace(0.0, len(sequence) - 1, output_count)
    lower = np.floor(source_positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(sequence) - 1)
    weights = (source_positions - lower).reshape((-1,) + (1,) * (sequence.ndim - 1))
    return ((1.0 - weights) * sequence[lower] + weights * sequence[upper]).astype(np.float32)


def postprocess_maps(maps, source_rate, target_rate, output_size=(64, 64)):
    channels = []
    for name in MAP_NAMES:
        if name not in maps:
            raise KeyError(f"Missing radar map: {name}")
        resized = resize_map_sequence(maps[name], output_size)
        resampled = resample_map_sequence(resized, source_rate, target_rate)
        channels.append(np.log1p(np.maximum(resampled, 0.0)))
    common_length = min(len(channel) for channel in channels)
    stacked = np.stack([channel[:common_length] for channel in channels], axis=1)
    mean = stacked.mean(axis=(0, 2, 3), keepdims=True)
    std = stacked.std(axis=(0, 2, 3), keepdims=True)
    return ((stacked - mean) / (std + 1e-6)).astype(np.float32)


def postprocess_three_maps(maps, output_size=(64, 64)):
    channels = []
    for name in THREE_MAP_NAMES:
        if name not in maps:
            raise KeyError(f"Missing radar map: {name}")
        resized = resize_map_sequence(maps[name], output_size)
        channels.append(np.log1p(np.maximum(resized, 0.0)))
    common_length = min(len(channel) for channel in channels)
    stacked = np.stack([channel[:common_length] for channel in channels], axis=-1)
    mean = stacked.mean(axis=(0, 1, 2), keepdims=True)
    std = stacked.std(axis=(0, 1, 2), keepdims=True)
    normalized = (stacked - mean) / (std + 1e-6)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32)


def load_radar_chunk(bin_path, output_size=(64, 64)):
    maps, _ = acquire_three_maps(bin_path)
    return postprocess_three_maps(maps, output_size)


def load_radar_tensor(minute_dir, target_rate, output_size=(64, 64)):
    minute_dir = Path(minute_dir)
    radar_files = sorted(minute_dir.glob("mmw_radar_raw_*.bin"))
    if not radar_files:
        raise FileNotFoundError(f"No mmw_radar_raw_*.bin in {minute_dir}")
    maps, source_rate = acquire_four_maps(radar_files[0])
    return postprocess_maps(maps, source_rate, target_rate, output_size)


def fit_temporal_length(array, length):
    array = np.asarray(array)
    if length <= 0:
        raise ValueError("length must be positive")
    if len(array) == length:
        return array
    if len(array) > length:
        return array[:length]
    if len(array) == 0:
        return np.zeros((length,) + array.shape[1:], dtype=np.float32)
    padding = np.repeat(array[-1:], length - len(array), axis=0)
    return np.concatenate((array, padding), axis=0)
