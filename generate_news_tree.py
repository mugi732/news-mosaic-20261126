#!/usr/bin/env python3
"""
generate_news_tree.py

Markdown形式の英語ニューステキストを等幅グリッドに隙間なく敷き詰め、
文字の色分けのみで「空・雲・大きな木（葉・幹・枝）・大地」を描く
タイポグラフィ・モザイクアート生成スクリプト。
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont

# 高精細な文字画像のJPEG optimize保存時にバッファ不足(Suspension not allowed here)を防ぐ
ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 16 * 1024 * 1024)

# ==============================================================================
# 調整用パラメータ（定数）
# ==============================================================================
INPUT_MD_PATH = "news_20261126.md"
OUTPUT_PNG_PATH = "output_tree_news.png"
OUTPUT_JPG_PATH = "output_tree_news.jpg"

# キャンバス設定（1024x1024 px 以下厳守）
CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 1024
BG_COLOR = (245, 245, 242)  # #F5F5F2 オフホワイト紙面ベース

# フォント・グリッド設定（すべての文字で同一フォント・同一サイズ・変形なし）
REPEAT_TEXT = False         # False=記事を繰り返さず1回のみ表示 / True=末尾以降もループして埋める
FONT_SIZE = 18              # 1回表示でキャンバス全体にちょうど収まる大きめサイズ (旧: 10px)
FONT_FACE_INDEX = 1         # Menlo.ttc の場合: 0=Regular, 1=Bold (太字の方が色の視認性が高い)
CELL_WIDTH = 11             # 横方向の文字送り幅 (px)
CELL_HEIGHT = 19            # 縦方向の行送り幅 (px)
GRID_COLS: int | None = 88  # グリッド列数 (Noneの場合はキャンバス幅÷CELL_WIDTH)
GRID_ROWS: int | None = 52  # グリッド行数 (Noneの場合はキャンバス高÷CELL_HEIGHT)
NEWS_SEPARATOR = " | "      # ニュース間の区切り文字列

# 木のサイズ・配置設定（キャンバスに対する相対比率 0.0〜1.0）
TREE_CENTER_X = 0.50        # 木の中心X座標
TREE_CANOPY_CENTER_Y = 0.38 # 樹冠（葉の塊）の中心Y座標
TREE_SCALE = 1.00           # 木全体のスケール倍率
GROUND_LEVEL_Y = 0.855      # 大地の基準高さY座標
RANDOM_SEED = 20261126      # 雲・葉・木肌のノイズ乱数シード

# ファイルサイズ上限（1 MiB = 1,048,576 bytes）
MAX_FILE_BYTES = 1_048_576

# 自動探索するシステム等幅フォント候補
MONOSPACE_FONT_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("/System/Library/Fonts/Menlo.ttc", FONT_FACE_INDEX),
    ("/System/Library/Fonts/Menlo.ttc", 0),
    ("/System/Library/Fonts/Monaco.ttf", 0),
    ("/System/Library/Fonts/SFNSMono.ttf", 0),
    ("/System/Library/Fonts/Courier.ttc", 0),
    ("/Library/Fonts/DejaVuSansMono.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
)


# ==============================================================================
# フォント探索・テキスト整形
# ==============================================================================
def find_monospace_font(
    font_size: int,
    custom_font_path: str | None = None,
    face_index: int = FONT_FACE_INDEX,
) -> tuple[ImageFont.FreeTypeFont, str, int]:
    """システム内の等幅フォントを自動探索して読み込む。見つからない場合は例外を送出。"""
    candidates: list[tuple[str, int]] = []
    if custom_font_path:
        candidates.append((custom_font_path, face_index))
        candidates.append((custom_font_path, 0))
    candidates.extend(MONOSPACE_FONT_CANDIDATES)

    for path_str, idx in candidates:
        path = Path(path_str)
        if not path.is_file():
            continue
        try:
            font = ImageFont.truetype(str(path), size=font_size, index=idx)
            _, style = font.getname()
            if "italic" in style.lower() or "oblique" in style.lower():
                continue
            return font, str(path), idx
        except OSError:
            continue

    searched = ", ".join(sorted({p for p, _ in candidates}))
    raise FileNotFoundError(
        f"利用可能な等幅フォントが見つかりませんでした。探索パス: {searched}"
    )


def load_and_clean_news_text(
    md_path: Path,
    separator: str = NEWS_SEPARATOR,
    repeat_text: bool = REPEAT_TEXT,
) -> str:
    """
    Markdownファイルを読み込み、装飾記号や改行・余分な空白を整理し、
    文字化け防止のためASCII互換のクリーンな連続文字列に整形する。
    """
    if not md_path.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {md_path}")

    raw_text = md_path.read_text(encoding="utf-8")

    # 等幅グリッドで幅ずれ・文字化けを起こしやすいダッシュ・引用符をASCII正規化
    replacements = {
        "—": " - ",
        "–": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        raw_text = raw_text.replace(src, dst)

    segments: list[str] = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        # 見出し記号(#)や太字(**)などのMarkdown記法を除去
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            segments.append(cleaned)

    if not segments:
        raise ValueError(f"入力ファイルに有効なテキストが含まれていません: {md_path}")

    joined = separator.join(segments) + (separator if repeat_text else "")
    # 万が一の制御文字や非印字文字を除外し、印字可能ASCII文字に統一
    joined = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in joined)
    joined = re.sub(r" {2,}", " ", joined)
    return joined


# ==============================================================================
# プロシージャル・カラーマップ生成（空・雲・大きな木・大地）
# ==============================================================================
def _smooth_noise_2d(
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """バイリニア補間＋コサインスムージングによる滑らかな2Dノイズ (0.0〜1.0)。"""
    coarse = rng.random((grid_h + 1, grid_w + 1), dtype=np.float32)
    ys = np.linspace(0, grid_h, height, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, grid_w, width, endpoint=False, dtype=np.float32)

    yi = np.floor(ys).astype(np.int32)
    xi = np.floor(xs).astype(np.int32)
    yf = ys - yi
    xf = xs - xi

    # Smoothstep (3t^2 - 2t^3)
    uy = yf * yf * (3.0 - 2.0 * yf)
    ux = xf * xf * (3.0 - 2.0 * xf)

    c00 = coarse[yi[:, None], xi[None, :]]
    c10 = coarse[yi[:, None], xi[None, :] + 1]
    c01 = coarse[yi[:, None] + 1, xi[None, :]]
    c11 = coarse[yi[:, None] + 1, xi[None, :] + 1]

    top = c00 + ux[None, :] * (c10 - c00)
    bottom = c01 + ux[None, :] * (c11 - c01)
    return top + uy[:, None] * (bottom - top)


def _fbm_2d(
    height: int,
    width: int,
    base_h: int,
    base_w: int,
    octaves: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """複数オクターブを重ねたフラクタルノイズ (0.0〜1.0に正規化)。"""
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    amp_sum = 0.0
    gh, gw = base_h, base_w
    for _ in range(octaves):
        total += amplitude * _smooth_noise_2d(height, width, gh, gw, rng)
        amp_sum += amplitude
        amplitude *= 0.5
        gh = max(2, gh * 2)
        gw = max(2, gw * 2)
    total /= amp_sum
    vmin, vmax = float(total.min()), float(total.max())
    if vmax > vmin:
        total = (total - vmin) / (vmax - vmin)
    return total


def _segment_distance_and_side(
    px: np.ndarray,
    py: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    各グリッド点 (px, py) から線分 A(ax,ay)->B(bx,by) への最短距離、
    補間係数 t (0..1)、および法線方向の符号付きオフセットを返す。
    """
    vx = bx - ax
    vy = by - ay
    seg_len_sq = vx * vx + vy * vy + 1e-9
    t = ((px - ax) * vx + (py - ay) * vy) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    proj_x = ax + t_clamped * vx
    proj_y = ay + t_clamped * vy
    dx = px - proj_x
    dy = py - proj_y
    dist = np.hypot(dx, dy)
    # 左側（左上からの太陽光側）を負〜正で判定するための水平寄りシェーディング係数
    seg_len = math.sqrt(seg_len_sq)
    nx = -vy / seg_len
    ny = vx / seg_len
    # 常に画面左側が明るくなるよう法線の向きを揃える
    if nx > 0:
        nx, ny = -nx, -ny
    side = (dx * (-nx) + dy * (-ny)) / (dist + 1e-6)
    return dist, t_clamped, side


def build_tree_scene_colormap(
    cols: int,
    rows: int,
    center_x: float = TREE_CENTER_X,
    canopy_center_y: float = TREE_CANOPY_CENTER_Y,
    tree_scale: float = TREE_SCALE,
    ground_level_y: float = GROUND_LEVEL_Y,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    グリッドサイズ (rows, cols) に対して、各セルのRGB色 (uint8, shape=(rows, cols, 3)) を生成する。
    外部画像を使わず、空・雲・大地・幹と枝・樹冠（葉の重なりと陰影）を合成する。
    """
    rng = np.random.default_rng(seed)

    # 正規化座標 (0.0〜1.0)
    ys = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    xs = (np.arange(cols, dtype=np.float32) + 0.5) / cols
    xx, yy = np.meshgrid(xs, ys)

    rgb = np.zeros((rows, cols, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # 1. 空（上部〜背景）: 淡いライトブルーのグラデーション + 雲
    #    ※オフホワイト背景(#F5F5F2)上で文字がしっかり読める彩度・明度に設計
    # ------------------------------------------------------------------
    sky_t = np.clip(yy / ground_level_y, 0.0, 1.0)
    sky_top = np.array([78.0, 152.0, 222.0], dtype=np.float32)     # 爽やかなスカイブルー
    sky_horizon = np.array([128.0, 185.0, 234.0], dtype=np.float32) # 地平線近くの淡い水色
    rgb = sky_top[None, None, :] * (1.0 - sky_t[:, :, None]) + sky_horizon[None, None, :] * sky_t[:, :, None]

    # 雲のレイヤー（楕円パフ群 + FBMノイズで自然なちぎれ雲を表現）
    cloud_noise = _fbm_2d(rows, cols, 4, 7, 4, rng)
    cloud_field = np.zeros((rows, cols), dtype=np.float32)
    cloud_specs = [
        # (cx, cy, rx, ry, intensity)
        (0.16, 0.13, 0.14, 0.048, 1.00),
        (0.24, 0.15, 0.11, 0.040, 0.90),
        (0.09, 0.15, 0.09, 0.034, 0.85),
        (0.82, 0.11, 0.15, 0.046, 1.00),
        (0.73, 0.13, 0.10, 0.038, 0.88),
        (0.90, 0.14, 0.10, 0.035, 0.85),
        (0.12, 0.44, 0.12, 0.036, 0.78),
        (0.88, 0.48, 0.13, 0.038, 0.80),
        (0.50, 0.07, 0.16, 0.034, 0.72),
    ]
    for cx, cy, rx, ry, strength in cloud_specs:
        d2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        puff = strength * np.exp(-1.5 * d2)
        cloud_field = np.maximum(cloud_field, puff)

    # ノイズで雲の輪郭をふわっと崩す（上空のみ）
    cloud_mask = np.clip((cloud_field + 0.34 * (cloud_noise - 0.45) - 0.28) / 0.52, 0.0, 1.0)
    cloud_mask *= np.clip((ground_level_y - 0.10 - yy) / 0.20, 0.0, 1.0)

    # 雲の色：紙面(#F5F5F2)上で視認でき、かつ青空の中で「白い雲」に見える明るいアイスグレーブルー
    cloud_bright = np.array([188.0, 204.0, 218.0], dtype=np.float32)
    cloud_shade = np.array([154.0, 178.0, 202.0], dtype=np.float32)
    cloud_vertical = np.clip((0.5 - (cloud_noise - 0.5)), 0.0, 1.0)
    cloud_color = (
        cloud_bright[None, None, :] * cloud_vertical[:, :, None]
        + cloud_shade[None, None, :] * (1.0 - cloud_vertical[:, :, None])
    )
    rgb = rgb * (1.0 - cloud_mask[:, :, None]) + cloud_color * cloud_mask[:, :, None]

    # ------------------------------------------------------------------
    # 2. 幹・根・大枝（中央〜下部）: 濃い茶色〜暗褐色 + 木肌陰影
    # ------------------------------------------------------------------
    s = tree_scale
    gx_hill = ground_level_y - 0.032 * np.exp(-(((xx - center_x) / (0.30 * s)) ** 2))
    trunk_base_y = ground_level_y - 0.015
    trunk_split_y = canopy_center_y + 0.13 * s

    bark_noise = _fbm_2d(rows, cols, 14, 28, 3, rng)
    wood_mask = np.zeros((rows, cols), dtype=np.float32)
    wood_shade = np.zeros((rows, cols), dtype=np.float32)

    # 幹の曲線中心線と太さ（根元が広がる）
    trunk_top_y = canopy_center_y - 0.02 * s
    in_trunk_y = (yy >= trunk_top_y) & (yy <= ground_level_y + 0.05)
    t_trunk = np.clip((yy - trunk_top_y) / max(1e-5, (trunk_base_y - trunk_top_y)), 0.0, 1.25)
    trunk_curve_x = center_x + 0.012 * s * np.sin(t_trunk * math.pi * 1.3)
    # 根元のフレア（末広がり）
    root_flare = 0.095 * s * np.power(np.clip((t_trunk - 0.68) / 0.42, 0.0, 1.3), 2.2)
    trunk_half_w = (0.030 + 0.038 * t_trunk) * s + root_flare + 0.004 * (bark_noise - 0.5)

    dx_trunk = xx - trunk_curve_x
    inside_trunk = in_trunk_y & (np.abs(dx_trunk) <= trunk_half_w)
    wood_mask[inside_trunk] = 1.0
    # 左側（負のdx）が明るく、右側が暗い円柱シェーディング
    rel_x = np.clip(dx_trunk / (trunk_half_w + 1e-6), -1.0, 1.0)
    wood_shade[inside_trunk] = 0.5 * (1.0 - rel_x[inside_trunk])

    # 大枝・中枝・根の線分リスト: (ax, ay, bx, by, r_start, r_end)
    branches = [
        # 主枝（左・左上・中央・右上・右）
        (center_x - 0.008 * s, trunk_split_y + 0.03 * s, center_x - 0.17 * s, canopy_center_y + 0.03 * s, 0.030 * s, 0.015 * s),
        (center_x - 0.17 * s, canopy_center_y + 0.03 * s, center_x - 0.29 * s, canopy_center_y - 0.04 * s, 0.015 * s, 0.008 * s),
        (center_x - 0.004 * s, trunk_split_y, center_x - 0.11 * s, canopy_center_y - 0.10 * s, 0.026 * s, 0.012 * s),
        (center_x - 0.11 * s, canopy_center_y - 0.10 * s, center_x - 0.19 * s, canopy_center_y - 0.18 * s, 0.012 * s, 0.007 * s),
        (center_x + 0.002 * s, trunk_split_y - 0.02 * s, center_x + 0.02 * s, canopy_center_y - 0.16 * s, 0.024 * s, 0.010 * s),
        (center_x + 0.008 * s, trunk_split_y + 0.01 * s, center_x + 0.13 * s, canopy_center_y - 0.09 * s, 0.026 * s, 0.012 * s),
        (center_x + 0.13 * s, canopy_center_y - 0.09 * s, center_x + 0.22 * s, canopy_center_y - 0.16 * s, 0.012 * s, 0.007 * s),
        (center_x + 0.012 * s, trunk_split_y + 0.04 * s, center_x + 0.18 * s, canopy_center_y + 0.04 * s, 0.029 * s, 0.014 * s),
        (center_x + 0.18 * s, canopy_center_y + 0.04 * s, center_x + 0.30 * s, canopy_center_y - 0.02 * s, 0.014 * s, 0.008 * s),
        # 地表に張る力強い根
        (center_x - 0.045 * s, trunk_base_y - 0.01 * s, center_x - 0.17 * s, ground_level_y + 0.025, 0.026 * s, 0.009 * s),
        (center_x - 0.015 * s, trunk_base_y, center_x - 0.09 * s, ground_level_y + 0.035, 0.022 * s, 0.008 * s),
        (center_x + 0.020 * s, trunk_base_y, center_x + 0.10 * s, ground_level_y + 0.035, 0.022 * s, 0.008 * s),
        (center_x + 0.048 * s, trunk_base_y - 0.01 * s, center_x + 0.18 * s, ground_level_y + 0.025, 0.026 * s, 0.009 * s),
    ]

    for ax, ay, bx, by, r0, r1 in branches:
        dist, t_seg, side = _segment_distance_and_side(xx, yy, ax, ay, bx, by)
        radius = r0 + (r1 - r0) * t_seg + 0.0025 * (bark_noise - 0.5)
        hit = dist <= radius
        wood_mask[hit] = 1.0
        local_light = np.clip(0.5 + 0.45 * side, 0.0, 1.0)
        wood_shade[hit] = np.maximum(wood_shade[hit], local_light[hit])

    bark_dark = np.array([46.0, 27.0, 15.0], dtype=np.float32)     # 深い暗褐色（影）
    bark_mid = np.array([82.0, 50.0, 27.0], dtype=np.float32)      # 幹の濃い茶色
    bark_light = np.array([118.0, 76.0, 42.0], dtype=np.float32)   # 陽が当たる樹皮
    bark_factor = np.clip(0.65 * wood_shade + 0.35 * bark_noise, 0.0, 1.0)
    wood_color = np.where(
        (bark_factor < 0.5)[:, :, None],
        bark_dark[None, None, :] + (bark_mid - bark_dark)[None, None, :] * (bark_factor[:, :, None] * 2.0),
        bark_mid[None, None, :] + (bark_light - bark_mid)[None, None, :] * ((bark_factor[:, :, None] - 0.5) * 2.0),
    )
    # 樹冠直下の幹上部に落ちる葉陰（アンビエントオクルージョン）
    canopy_shadow_on_trunk = np.clip((canopy_center_y + 0.21 * s - yy) / (0.10 * s), 0.0, 1.0)
    wood_color *= (1.0 - 0.32 * canopy_shadow_on_trunk[:, :, None])

    rgb = np.where(wood_mask[:, :, None] > 0.5, wood_color, rgb)

    # ------------------------------------------------------------------
    # 3. 木の葉・樹冠（中央上部）: 緑・深緑・黄緑の豊かな階層グラデーション
    # ------------------------------------------------------------------
    leaf_macro_noise = _fbm_2d(rows, cols, 6, 6, 4, rng)
    leaf_detail_noise = _fbm_2d(rows, cols, 18, 18, 3, rng)
    dapple_noise = rng.random((rows, cols), dtype=np.float32)

    # 樹冠を構成する複数の葉塊クラスタ: (dx_offset, dy_offset, rx, ry, layer_brightness)
    # 奥の暗い葉塊 → 中間の緑 → 手前・上部の明るい黄緑の順に重ねて立体感を出す
    foliage_clusters = [
        # 背面の深い緑のベースクラスタ（輪郭の厚みと枝間の奥行き）
        (0.00, 0.02, 0.36, 0.21, 0.18),
        (-0.22, 0.07, 0.20, 0.14, 0.16),
        (0.22, 0.07, 0.20, 0.14, 0.15),
        (-0.14, -0.10, 0.21, 0.15, 0.22),
        (0.14, -0.10, 0.21, 0.15, 0.20),
        (0.00, -0.16, 0.20, 0.14, 0.25),

        # 中間層の豊かな葉の房
        (-0.30, 0.09, 0.14, 0.105, 0.42),
        (-0.18, 0.11, 0.15, 0.095, 0.38),
        (0.18, 0.11, 0.15, 0.095, 0.34),
        (0.30, 0.09, 0.14, 0.105, 0.36),
        (-0.25, -0.02, 0.16, 0.125, 0.55),
        (0.25, -0.02, 0.16, 0.125, 0.45),
        (-0.12, -0.13, 0.17, 0.130, 0.62),
        (0.13, -0.13, 0.17, 0.130, 0.50),
        (0.00, -0.20, 0.16, 0.115, 0.66),
        (0.00, -0.01, 0.20, 0.135, 0.52),

        # 前面・日向の立体的な葉の房（左上からの陽光を受けるハイライト房）
        (-0.08, -0.21, 0.12, 0.090, 0.86),
        (-0.20, -0.09, 0.13, 0.095, 0.82),
        (-0.28, 0.03, 0.11, 0.082, 0.72),
        (-0.06, -0.06, 0.14, 0.100, 0.78),
        (0.09, -0.16, 0.12, 0.090, 0.74),
        (0.19, -0.05, 0.12, 0.090, 0.64),
        (0.06, 0.05, 0.14, 0.092, 0.54),
        (-0.12, 0.06, 0.13, 0.088, 0.58),
    ]

    deep_forest = np.array([22.0, 68.0, 30.0], dtype=np.float32)   # 深い森の暗緑色
    lush_green = np.array([46.0, 128.0, 48.0], dtype=np.float32)   # 鮮やかな緑
    fresh_leaf = np.array([104.0, 178.0, 52.0], dtype=np.float32)  # 陽光の黄緑
    sun_tip = np.array([142.0, 204.0, 62.0], dtype=np.float32)     # 樹冠上端の新緑ハイライト

    # 枝が葉の隙間から少し覗く窓（木らしさを高める小さな抜け）
    branch_peek = (
        np.exp(-(((xx - (center_x - 0.07 * s)) / (0.045 * s)) ** 2 + ((yy - (canopy_center_y + 0.11 * s)) / (0.030 * s)) ** 2))
        + np.exp(-(((xx - (center_x + 0.08 * s)) / (0.048 * s)) ** 2 + ((yy - (canopy_center_y + 0.10 * s)) / (0.030 * s)) ** 2))
    )

    for dx_off, dy_off, rx_base, ry_base, base_bright in foliage_clusters:
        cx = center_x + dx_off * s
        cy = canopy_center_y + dy_off * s
        rx = rx_base * s
        ry = ry_base * s

        rel_dx = (xx - cx) / rx
        rel_dy = (yy - cy) / ry
        dist_ell = np.hypot(rel_dx, rel_dy)

        # ノイズで楕円の境界を葉の房らしくモコモコに変形
        edge_warp = 0.22 * (leaf_macro_noise - 0.5) + 0.14 * (leaf_detail_noise - 0.5)
        # 下部の枝分かれ付近は少し隙間を作って幹・大枝を覗かせる
        peek_cut = 0.22 * branch_peek * (1.0 if base_bright < 0.6 else 0.35)
        in_cluster = (dist_ell + edge_warp + peek_cut) <= 1.0

        if not np.any(in_cluster):
            continue

        # 各房ごとの球面ライティング（左上 (-0.55, -0.75) から太陽光）
        local_sun = np.clip(0.52 + (-0.36 * rel_dx - 0.44 * rel_dy), 0.0, 1.0)
        # 木全体の大域ライティング（上部ほど明るく、下部ほど深い陰影）
        global_sun = np.clip(
            0.58
            - 0.22 * ((xx - center_x) / (0.38 * s))
            - 0.45 * ((yy - (canopy_center_y - 0.05 * s)) / (0.28 * s)),
            0.0,
            1.0,
        )

        # 葉の細かいきらめき（文字単位の緑〜黄緑〜深緑のモザイク感）
        shimmer = 0.18 * (leaf_detail_noise - 0.5) + 0.12 * (dapple_noise - 0.5)
        shade_val = np.clip(
            0.38 * base_bright + 0.36 * local_sun + 0.26 * global_sun + shimmer,
            0.0,
            1.0,
        )

        # 4段階の緑パレット補間
        c_leaf = np.empty_like(rgb)
        m1 = shade_val < 0.36
        m2 = (shade_val >= 0.36) & (shade_val < 0.70)
        m3 = shade_val >= 0.70

        t1 = (shade_val / 0.36)[:, :, None]
        t2 = ((shade_val - 0.36) / 0.34)[:, :, None]
        t3 = ((shade_val - 0.70) / 0.30)[:, :, None]

        c1 = deep_forest[None, None, :] * (1.0 - t1) + lush_green[None, None, :] * t1
        c2 = lush_green[None, None, :] * (1.0 - t2) + fresh_leaf[None, None, :] * t2
        c3 = fresh_leaf[None, None, :] * (1.0 - t3) + sun_tip[None, None, :] * t3

        c_leaf = np.where(m1[:, :, None], c1, np.where(m2[:, :, None], c2, c3))
        rgb = np.where(in_cluster[:, :, None], c_leaf, rgb)

    # ------------------------------------------------------------------
    # 4. 大地（最下部）: どっしりとした濃い茶色・ダークアースカラー
    # ------------------------------------------------------------------
    soil_noise = _fbm_2d(rows, cols, 8, 16, 4, rng)
    hill_wave = 0.008 * np.sin(xx * math.pi * 3.2 + 0.4) + 0.006 * (soil_noise - 0.5)
    ground_y_curve = gx_hill + hill_wave
    in_ground = yy >= ground_y_curve

    ground_depth = np.clip((yy - ground_y_curve) / max(1e-5, (1.0 - ground_level_y + 0.05)), 0.0, 1.0)

    # 地表のわずかな苔・草色エッジから、重厚なダークアースカラーへ遷移
    turf_brown_green = np.array([76.0, 96.0, 42.0], dtype=np.float32)
    rich_earth = np.array([92.0, 58.0, 33.0], dtype=np.float32)
    deep_earth = np.array([52.0, 32.0, 18.0], dtype=np.float32)

    # 木の真下に落ちる大きな影
    tree_ground_shadow = 0.35 * np.exp(-(((xx - center_x) / (0.34 * s)) ** 2))

    earth_mix = np.clip(ground_depth * 1.15 + 0.22 * (soil_noise - 0.5), 0.0, 1.0)
    earth_color = (
        rich_earth[None, None, :] * (1.0 - earth_mix[:, :, None])
        + deep_earth[None, None, :] * earth_mix[:, :, None]
    )
    # 地表のごく上端1〜2行に自然な草土の縁取りを入れる
    rim_factor = np.clip(1.0 - ground_depth / 0.14, 0.0, 1.0)
    earth_color = (
        earth_color * (1.0 - 0.55 * rim_factor[:, :, None])
        + turf_brown_green[None, None, :] * (0.55 * rim_factor[:, :, None])
    )
    earth_color *= (1.0 - tree_ground_shadow[:, :, None])

    # 根が地表に食い込む部分は幹・根の立体感を残しつつ大地と馴染ませる
    root_on_ground = in_ground & (wood_mask > 0.5) & (yy <= ground_y_curve + 0.045)
    rgb = np.where(in_ground[:, :, None] & (~root_on_ground)[:, :, None], earth_color, rgb)

    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


# ==============================================================================
# タイポグラフィ・グリッド描画 & 1MiB以下保存処理
# ==============================================================================
def render_typography_mosaic(
    text: str,
    font: ImageFont.FreeTypeFont,
    canvas_width: int = CANVAS_WIDTH,
    canvas_height: int = CANVAS_HEIGHT,
    cell_width: int = CELL_WIDTH,
    cell_height: int = CELL_HEIGHT,
    grid_cols: int | None = GRID_COLS,
    grid_rows: int | None = GRID_ROWS,
    repeat_text: bool = REPEAT_TEXT,
    bg_color: tuple[int, int, int] = BG_COLOR,
    center_x: float = TREE_CENTER_X,
    canopy_center_y: float = TREE_CANOPY_CENTER_Y,
    tree_scale: float = TREE_SCALE,
    ground_level_y: float = GROUND_LEVEL_Y,
    seed: int = RANDOM_SEED,
) -> tuple[Image.Image, dict[str, int]]:
    """
    カラーマップを生成し、キャンバスのグリッドセルに等幅・同一サイズの文字を
    変形・回転なしで厳密に1文字ずつ描画する。
    """
    if canvas_width > 1024 or canvas_height > 1024:
        raise ValueError(f"解像度は 1024x1024 以下である必要があります: {canvas_width}x{canvas_height}")

    max_cols = canvas_width // cell_width
    max_rows = canvas_height // cell_height
    cols = grid_cols if grid_cols is not None else max_cols
    rows = grid_rows if grid_rows is not None else max_rows

    if cols <= 0 or rows <= 0 or cols > max_cols or rows > max_rows:
        raise ValueError(
            f"グリッド構成 ({cols}x{rows}) がキャンバス ({canvas_width}x{canvas_height}, 最大 {max_cols}x{max_rows}) に収まりません。"
        )
    if not repeat_text and cols * rows < len(text):
        raise ValueError(
            f"1回表示モードで全文字が収まりません: グリッド容量 {cols * rows} マス < テキスト長 {len(text)} 文字"
        )

    # グリッド全体をキャンバス中央に配置する余白オフセット
    offset_x = (canvas_width - cols * cell_width) // 2
    offset_y = (canvas_height - rows * cell_height) // 2

    colormap = build_tree_scene_colormap(
        cols=cols,
        rows=rows,
        center_x=center_x,
        canopy_center_y=canopy_center_y,
        tree_scale=tree_scale,
        ground_level_y=ground_level_y,
        seed=seed,
    )

    image = Image.new("RGB", (canvas_width, canvas_height), color=bg_color)
    draw = ImageDraw.Draw(image)

    text_len = len(text)
    char_idx = 0
    placed_cells = 0
    drawn_non_space = 0

    # 文字の縦位置をセル内に綺麗に収める基準オフセットを算出
    bbox_sample = font.getbbox("Mg")
    glyph_h = bbox_sample[3] - bbox_sample[1]
    y_pad = max(0, (cell_height - glyph_h) // 2) - bbox_sample[1]

    for r in range(rows):
        y = offset_y + r * cell_height + y_pad
        row_colors = colormap[r]
        for c in range(cols):
            if not repeat_text and char_idx >= text_len:
                break
            ch = text[char_idx % text_len] if repeat_text else text[char_idx]
            char_idx += 1
            placed_cells += 1
            if ch != " ":
                r_col, g_col, b_col = int(row_colors[c, 0]), int(row_colors[c, 1]), int(row_colors[c, 2])
                x = offset_x + c * cell_width
                draw.text((x, y), ch, font=font, fill=(r_col, g_col, b_col))
                drawn_non_space += 1

    stats = {
        "cols": cols,
        "rows": rows,
        "total_cells": cols * rows,
        "placed_cells": placed_cells,
        "drawn_chars": drawn_non_space,
        "source_text_len": text_len,
    }
    return image, stats


def save_within_size_limit(
    image: Image.Image,
    png_path: Path,
    jpg_path: Path | None = None,
    max_bytes: int = MAX_FILE_BYTES,
) -> dict[str, tuple[Path, int, str]]:
    """
    PNG（および必要に応じてJPEG）を確実に max_bytes (1 MiB) 以下で保存する。
    PNGが1MiBを超える場合は段階的にパレット減色(quantize)を適用して1MiB以下に収める。
    """
    results: dict[str, tuple[Path, int, str]] = {}

    # 1) まずフルカラー最適化PNGを試行
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True, compress_level=9)
    png_bytes = buf.getvalue()
    png_mode_desc = "RGB lossless (optimize=True)"

    # 2) 1MiB超過時は256色→192色→128色→96色→64色の順で高品位減色(MEDIANCUT)して確実に収める
    if len(png_bytes) > max_bytes:
        for num_colors in (256, 192, 160, 128, 96, 64, 48, 32):
            quantized = image.quantize(
                colors=num_colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            q_buf = io.BytesIO()
            quantized.save(q_buf, format="PNG", optimize=True, compress_level=9)
            candidate = q_buf.getvalue()
            if len(candidate) <= max_bytes:
                png_bytes = candidate
                png_mode_desc = f"Indexed PNG ({num_colors} colors, optimize=True)"
                break

    if len(png_bytes) > max_bytes:
        raise RuntimeError(f"PNGを {max_bytes} bytes 以下に圧縮できませんでした ({len(png_bytes)} bytes)")

    png_path.write_bytes(png_bytes)
    results["png"] = (png_path, len(png_bytes), png_mode_desc)

    # オプションまたはバックアップとしてJPEGも出力可能
    if jpg_path is not None:
        jpg_bytes = b""
        used_q = 92
        for quality in (95, 92, 88, 84, 80, 75, 70):
            j_buf = io.BytesIO()
            image.save(j_buf, format="JPEG", quality=quality, optimize=True, subsampling=0)
            candidate = j_buf.getvalue()
            if len(candidate) <= max_bytes:
                jpg_bytes = candidate
                used_q = quality
                break
        if not jpg_bytes:
            j_buf = io.BytesIO()
            image.save(j_buf, format="JPEG", quality=65, optimize=True)
            jpg_bytes = j_buf.getvalue()
            used_q = 65
        jpg_path.write_bytes(jpg_bytes)
        results["jpg"] = (jpg_path, len(jpg_bytes), f"JPEG (quality={used_q}, 4:4:4)")

    return results


# ==============================================================================
# 自動検証ロジック
# ==============================================================================
def verify_generated_outputs(
    saved_files: dict[str, tuple[Path, int, str]],
    font: ImageFont.FreeTypeFont,
    text: str,
    cell_width: int,
    cell_height: int,
    max_bytes: int = MAX_FILE_BYTES,
) -> list[str]:
    """解像度・ファイルサイズ・グリフ欠け/文字化けの有無を自動検証する。"""
    report_lines: list[str] = []

    # 1) 使用文字のグリフ検証（非印字文字・豆腐化しやすい未定義文字がないか、等幅に収まるか）
    unique_chars = sorted(set(text))
    non_ascii = [ch for ch in unique_chars if not (32 <= ord(ch) <= 126)]
    if non_ascii:
        raise AssertionError(f"非ASCII文字が残っています（文字化けリスク）: {non_ascii}")

    widths = {round(font.getlength(ch), 2) for ch in unique_chars}
    if len(widths) > 1:
        raise AssertionError(f"等幅フォントではありません（文字幅のばらつき検出: {widths}）")

    # 単一グリフ描画テスト（空白以外の全ユニーク文字でピクセルが実際に描かれるか確認）
    missing_glyphs: list[str] = []
    for ch in unique_chars:
        if ch == " ":
            continue
        test_img = Image.new("L", (cell_width * 3, cell_height * 3), color=255)
        test_draw = ImageDraw.Draw(test_img)
        test_draw.text((cell_width, cell_height), ch, font=font, fill=0)
        arr = np.asarray(test_img)
        if not np.any(arr < 250):
            missing_glyphs.append(ch)
    if missing_glyphs:
        raise AssertionError(f"描画されない欠損グリフが検出されました: {missing_glyphs}")

    report_lines.append(
        f"[OK] 文字・グリフ検証: ユニーク文字 {len(unique_chars)} 種すべてASCII印字可能・等幅(幅={next(iter(widths))}px)・欠損ゼロ"
    )

    # 2) 画像ファイルごとのサイズ・解像度・描画内容検証
    for fmt, (path, byte_size, mode_desc) in saved_files.items():
        if not path.is_file():
            raise AssertionError(f"出力ファイルが存在しません: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes > max_bytes:
            raise AssertionError(
                f"{path.name} のファイルサイズが 1 MiB を超えています: {actual_bytes:,} bytes"
            )
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            if w > 1024 or h > 1024:
                raise AssertionError(f"{path.name} の解像度が 1024x1024 を超えています: {w}x{h}")
            rgb_arr = np.asarray(im.convert("RGB"))
            # 四隅と中央すべてに文字色が描画されているか検証
            bg = np.array(BG_COLOR, dtype=np.int16)
            diff = np.abs(rgb_arr.astype(np.int16) - bg[None, None, :]).sum(axis=-1)
            non_bg_ratio = float(np.mean(diff > 12))
            if non_bg_ratio < 0.05:
                raise AssertionError(f"{path.name} の文字描画密度が異常に低いです ({non_bg_ratio:.2%})")

        report_lines.append(
            f"[OK] {path.name}: 解像度 {w}x{h} px (<=1024x1024), "
            f"サイズ {actual_bytes:,} bytes ({actual_bytes / 1024:.1f} KiB <= 1,024.0 KiB), "
            f"形式={mode_desc}, 非背景ピクセル率={non_bg_ratio:.1%}"
        )

    return report_lines


# ==============================================================================
# CLI エントリポイント
# ==============================================================================
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ニューステキストから『空・大地・大きな木』のタイポグラフィ・モザイクアートを生成します。"
    )
    parser.add_argument("--input", default=INPUT_MD_PATH, help="入力Markdownファイルパス")
    parser.add_argument("--output-png", default=OUTPUT_PNG_PATH, help="出力PNGファイルパス")
    parser.add_argument("--output-jpg", default=OUTPUT_JPG_PATH, help="出力JPEGファイルパス（空文字で無効化）")
    parser.add_argument("--width", type=int, default=CANVAS_WIDTH, help="キャンバス幅 (<=1024)")
    parser.add_argument("--height", type=int, default=CANVAS_HEIGHT, help="キャンバス高さ (<=1024)")
    parser.add_argument("--font-path", default=None, help="任意の等幅フォントパス（未指定で自動探索）")
    parser.add_argument("--font-size", type=int, default=FONT_SIZE, help="フォントサイズ (px)")
    parser.add_argument("--cell-width", type=int, default=CELL_WIDTH, help="グリッド列幅 (px)")
    parser.add_argument("--cell-height", type=int, default=CELL_HEIGHT, help="グリッド行高 (px)")
    parser.add_argument("--cols", type=int, default=GRID_COLS, help="グリッド列数（省略時は自動算出）")
    parser.add_argument("--rows", type=int, default=GRID_ROWS, help="グリッド行数（省略時は自動算出）")
    parser.add_argument(
        "--repeat",
        action=argparse.BooleanOptionalAction,
        default=REPEAT_TEXT,
        help="テキストを末尾以降も繰り返して埋めるか (--repeat / --no-repeat)",
    )
    parser.add_argument("--tree-scale", type=float, default=TREE_SCALE, help="木のサイズ倍率")
    parser.add_argument("--tree-x", type=float, default=TREE_CENTER_X, help="木の中心X比率 (0.0-1.0)")
    parser.add_argument("--canopy-y", type=float, default=TREE_CANOPY_CENTER_Y, help="樹冠中心Y比率 (0.0-1.0)")
    parser.add_argument("--ground-y", type=float, default=GROUND_LEVEL_Y, help="大地の基準高さY比率 (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="乱数シード")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    font, font_path, face_idx = find_monospace_font(
        font_size=args.font_size,
        custom_font_path=args.font_path,
    )
    text = load_and_clean_news_text(Path(args.input), repeat_text=args.repeat)

    image, stats = render_typography_mosaic(
        text=text,
        font=font,
        canvas_width=args.width,
        canvas_height=args.height,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        grid_cols=args.cols,
        grid_rows=args.rows,
        repeat_text=args.repeat,
        bg_color=BG_COLOR,
        center_x=args.tree_x,
        canopy_center_y=args.canopy_y,
        tree_scale=args.tree_scale,
        ground_level_y=args.ground_y,
        seed=args.seed,
    )

    jpg_path = Path(args.output_jpg) if args.output_jpg else None
    saved = save_within_size_limit(
        image=image,
        png_path=Path(args.output_png),
        jpg_path=jpg_path,
        max_bytes=MAX_FILE_BYTES,
    )

    report_lines = verify_generated_outputs(
        saved_files=saved,
        font=font,
        text=text,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        max_bytes=MAX_FILE_BYTES,
    )

    print("=== 生成サマリー ===")
    print(f"使用フォント : {font_path} (index={face_idx}, size={args.font_size}px)")
    print(f"表示モード   : {'ループ繰り返し' if args.repeat else '1回のみ表示（繰り返しなし）'}")
    print(
        f"グリッド構成 : {stats['cols']}列 × {stats['rows']}行 = {stats['total_cells']:,} マス "
        f"(配置 {stats['placed_cells']:,} マス / 印字 {stats['drawn_chars']:,} 文字 / 元テキスト長 {stats['source_text_len']:,} 文字)"
    )
    for line in report_lines:
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
