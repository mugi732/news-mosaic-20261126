#!/usr/bin/env python3
"""
generate_news_mosaic_20261126.py (v3)

make.fun ニュース画像コンペ（VLM 1対1比較審査）向け v3 生成スクリプト。
クロスレビュー（Meta Muse / Gemini Neon）の指摘をすべて反映：
1. 最上部の独立題字ヘッダー帯（日付・件数・モチーフ意図を明示）
2. 回り込み大見出しカード（Hero Callout Cards 3枠）：
   - 30件のモザイク本文がカード領域を自動的に避けてワードラップ（1文字も隠れない）
   - 1行につきテキスト領域が1区間のみになるよう配置し、読み順の混乱を防止
   - パッと見0.5秒で今日の3大ニュース（米関税20-40%・神舟24号・AI株下落）が飛び込む階層構造を実現
3. 二層モザイク配色の完全二極化（中間輝度の死角を排除）：
   - 宇宙・海・大陸・関税の壁は「深みのある濃色セル＋白系文字」に統一（青や緑の上の黒文字を廃止）
   - 黒文字は「神舟24号の超高輝度イエロー〜ホワイト光条コア」のみに限定
   - 関税の壁の上下に暗い亀裂エッジを入れ、サムネイル大でも「壁で分断された地球」を明確化
4. フッター文言を報道中立表現（COMPILED / REPORTED）へ改善
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont

ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 16 * 1024 * 1024)

# ==============================================================================
# 定数設定
# ==============================================================================
INPUT_MD_PATH = "news_20261126.md"
OUTPUT_PNG_PATH = "entry.png"

CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 1024
MAX_FILE_BYTES = 1_048_576  # 1 MiB 厳守

# 最上部ヘッダー帯（76px）
HEADER_HEIGHT = 76
HEADER_BG_TOP = (6, 12, 28)
HEADER_BG_BOTTOM = (12, 24, 52)
HEADER_ACCENT_LINE = (72, 198, 255)
HEADER_TITLE = "WORLD NEWS MOSAIC — NOVEMBER 26, 2026"
HEADER_SUBTITLE = (
    "30 Reported Global Dispatches  |  Motif: US Tariff Wall Dividing Earth & Shenzhou-24 Launch"
)

RANDOM_SEED = 20261126

# 自動探索する等幅フォント候補: (path, regular_index, bold_index)
FONT_CANDIDATES: tuple[tuple[str, int, int], ...] = (
    ("/System/Library/Fonts/Menlo.ttc", 0, 1),
    ("/System/Library/Fonts/Courier.ttc", 0, 1),
    ("/System/Library/Fonts/Monaco.ttf", 0, 0),
    ("/System/Library/Fonts/SFNSMono.ttf", 0, 0),
    ("/Library/Fonts/DejaVuSansMono.ttf", 0, 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0, 0),
)


@dataclass(frozen=True)
class HeroCardSpec:
    """モザイクグリッド内にくり抜いて配置するトップニュース大見出しカード定義。"""

    row_start: int
    row_end: int   # exclusive
    col_start: int
    col_end: int   # exclusive
    badge: str
    headline: str
    bullet1: str
    bullet2: str
    accent_rgb: tuple[int, int, int]
    bg_rgb: tuple[int, int, int]


# ==============================================================================
# フォント読み込み
# ==============================================================================
def find_font_family() -> tuple[str, int, int]:
    for font_path, reg_idx, bold_idx in FONT_CANDIDATES:
        p = Path(font_path)
        if not p.is_file():
            continue
        try:
            f_reg = ImageFont.truetype(str(p), size=16, index=reg_idx)
            use_bold = bold_idx
            try:
                f_bold = ImageFont.truetype(str(p), size=16, index=bold_idx)
                _, style = f_bold.getname()
                if "italic" in style.lower() or "oblique" in style.lower():
                    use_bold = reg_idx
                elif round(f_reg.getlength("M"), 2) != round(f_bold.getlength("M"), 2):
                    use_bold = reg_idx
            except OSError:
                use_bold = reg_idx
            return str(p), reg_idx, use_bold
        except OSError:
            continue
    raise FileNotFoundError("利用可能なシステム等幅フォントが見つかりませんでした。")


def make_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)


# ==============================================================================
# ニュース解析・[01 TOPIC] タグ化
# ==============================================================================
def _clean_ascii(s: str) -> str:
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
        s = s.replace(src, dst)
    s = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def parse_news_items(md_path: Path) -> list[tuple[int, str, str]]:
    if not md_path.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {md_path}")

    raw = md_path.read_text(encoding="utf-8")
    items: list[tuple[int, str, str]] = []
    pattern = re.compile(r"^(\d+)\.\s*\*\*(.+?)\*\*\s*(.+)$")

    for line in raw.splitlines():
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue
        m = pattern.match(line_s)
        if m:
            num = int(m.group(1))
            topic = _clean_ascii(m.group(2).rstrip(":")).upper()
            body = _clean_ascii(m.group(3))
            tag = f"[{num:02d} {topic}]"
            items.append((num, tag, body))

    if not items:
        raise ValueError(f"ニュース項目を解析できませんでした: {md_path}")
    return items


def build_tagged_tokens(
    items: list[tuple[int, str, str]],
    sep_token: str = "|",
) -> list[tuple[str, bool, bool]]:
    """
    単語単位のトークン列 `(word, is_tag, is_lead_tag)` を返す。
    主要注目記事 (#07, #16, #21) のタグには `is_lead_tag=True` を付与。
    """
    lead_nums = {4, 7, 16, 18, 21}
    tokens: list[tuple[str, bool, bool]] = []
    for idx, (num, tag, body) in enumerate(items):
        if idx > 0 and sep_token:
            tokens.append((sep_token, True, False))
        is_lead = num in lead_nums
        if len(tag) <= 18:
            tokens.append((tag, True, is_lead))
        else:
            for part in tag.split(" "):
                tokens.append((part, True, is_lead))
        for word in body.split(" "):
            if word:
                tokens.append((word, False, False))
    return tokens


# ==============================================================================
# 大見出しカード配置と回り込み（ラップアラウンド）組版
# ==============================================================================
def build_hero_cards(cols: int, rows: int) -> list[HeroCardSpec]:
    """
    グリッド `(cols, rows)` に合わせて3つの大見出しカードのセル領域を定義する。
    ★ v4 最終調整（Neon & Muse 共通指摘の完全解消）：
    - 上段(rows 0..6)は左にCard1、中央にCard2を配置し、右端(cols 83..cols)は本文を流し込まず
      「神舟24号の先端スターバースト＆軌道アイコン専用ショーケース領域」として開放する。
    - これにより、右上角での「狭い列のジャスティファイ大空白」と「金色の光条と本文・カードの競合」を同時に根絶し、
      本文 `[01 THAILAND]` は row 7 の左端から全幅(102列)で美しくスタートする。
    """
    top_r0, top_r1 = 0, 6
    c1_start = 1
    c1_end = int(round(cols * 0.44))
    c2_start = c1_end + 1
    c2_end = int(round(cols * 0.82))

    bot_r1 = rows - 1
    bot_r0 = bot_r1 - 6
    c3_start = int(round(cols * 0.54))
    c3_end = cols - 1

    return [
        HeroCardSpec(
            row_start=top_r0,
            row_end=top_r1,
            col_start=c1_start,
            col_end=c1_end,
            badge="LEAD [#16-#19]  |  GLOBAL TRADE BARRIER",
            headline="U.S. 20-40% TARIFF WALL",
            bullet1="- Hits 14 partners: China, EU, Japan, S.Korea, SG",
            bullet2="- Sparked by excess factory capacity probe; Asia reacts",
            accent_rgb=(255, 82, 66),
            bg_rgb=(14, 18, 38),
        ),
        HeroCardSpec(
            row_start=top_r0,
            row_end=top_r1,
            col_start=c2_start,
            col_end=c2_end,
            badge="SPACE [#07]  |  TIANGONG MISSION  >>>",
            headline="SHENZHOU-24 LAUNCHES",
            bullet1="- Pakistani astronaut Muhammad Zeeshan Ali aboard",
            bullet2="- Marks 1st foreign crew member at Tiangong station",
            accent_rgb=(255, 214, 64),
            bg_rgb=(14, 20, 42),
        ),
        HeroCardSpec(
            row_start=bot_r0,
            row_end=bot_r1,
            col_start=c3_start,
            col_end=c3_end,
            badge="MARKETS & TECH [#20-#22]  |  AI & EQUITIES",
            headline="AI SELL-OFF: NASDAQ -0.6%",
            bullet1="- Nvidia & Anthropic fall on AI capex concerns",
            bullet2="- Europe & Asia advance (STI +0.7%); U.S. closed",
            accent_rgb=(72, 212, 255),
            bg_rgb=(10, 22, 46),
        ),
    ]


def compute_row_segments(
    cols: int,
    rows: int,
    cards: Sequence[HeroCardSpec],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    カード領域および上段右端の「神舟24号スターバースト鑑賞ウィンドウ」をマスクし、
    各行で本文を流し込める連続区間 `(row, col_start, col_end)` のリストを返す。
    """
    blocked = np.zeros((rows, cols), dtype=bool)
    for card in cards:
        cs = max(0, card.col_start - 1)
        ce = min(cols, card.col_end + 1)
        blocked[card.row_start : card.row_end, cs:ce] = True

    # 上段カード帯(rows 0..top_r1)は全幅をヘッダー下ショーケース帯として予約し、狭い右端列への本文流入を禁止
    if cards:
        top_band_end = max(c.row_end for c in cards if c.row_start == 0)
        blocked[0:top_band_end, :] = True

    segments: list[tuple[int, int, int]] = []
    for r in range(rows):
        c = 0
        while c < cols:
            if blocked[r, c]:
                c += 1
                continue
            start = c
            while c < cols and not blocked[r, c]:
                c += 1
            if c - start >= 24:
                segments.append((r, start, c))
    return blocked, segments


def wrap_tokens_around_cards(
    tokens: list[tuple[str, bool, bool]],
    cols: int,
    rows: int,
    cards: Sequence[HeroCardSpec],
) -> tuple[list[list[str]], np.ndarray, np.ndarray, bool, int]:
    """
    カード領域を避けながら全トークンを単語折り返し＆均等割り付けで配置する。
    戻り値: `(char_grid, bold_grid, lead_grid, all_fit, remaining_Empty_segments)`
    """
    _, segments = compute_row_segments(cols, rows, cards)

    char_grid = [[" " for _ in range(cols)] for _ in range(rows)]
    bold_grid = np.zeros((rows, cols), dtype=bool)
    lead_grid = np.zeros((rows, cols), dtype=bool)

    tok_idx = 0
    n_tok = len(tokens)
    used_seg_count = 0

    for seg_i, (r, c0, c1) in enumerate(segments):
        seg_w = c1 - c0
        if tok_idx >= n_tok:
            break

        line_toks: list[tuple[str, bool, bool]] = []
        cur_len = 0
        while tok_idx < n_tok:
            w, is_b, is_l = tokens[tok_idx]
            if len(w) > seg_w:
                # 狭い右端カラムに長いタグが来た場合はこのセグメントを装飾ドットで埋めて次の広い行へ送る
                if not line_toks:
                    break
                break
            needed = len(w) if not line_toks else (1 + len(w))
            if cur_len + needed <= seg_w:
                line_toks.append((w, is_b, is_l))
                cur_len += needed
                tok_idx += 1
            else:
                break

        if not line_toks:
            # セグメント幅より長い単一トークンだった場合の安全処理
            dots = ("· " * seg_w)[:seg_w]
            for i, ch in enumerate(dots):
                char_grid[r][c0 + i] = ch
            continue

        used_seg_count = seg_i + 1
        is_final_news_seg = tok_idx >= n_tok
        words_len = sum(len(w) for w, _, _ in line_toks)
        gaps = len(line_toks) - 1

        row_chars: list[str] = []
        row_b: list[bool] = []
        row_l: list[bool] = []

        if gaps > 0 and not is_final_news_seg and (seg_w - words_len) <= gaps * 3:
            total_sp = seg_w - words_len
            base_sp = total_sp // gaps
            extra_sp = total_sp % gaps
            for i, (w, is_b, is_l) in enumerate(line_tokens := line_toks):
                row_chars.extend(list(w))
                row_b.extend([is_b] * len(w))
                row_l.extend([is_l] * len(w))
                if i < gaps:
                    sp = base_sp + (1 if i < extra_sp else 0)
                    row_chars.extend([" "] * sp)
                    row_b.extend([False] * sp)
                    row_l.extend([False] * sp)
        elif not is_final_news_seg:
            # 万一1ギャップあたり4スペース以上空く極端な行があれば左寄せ＋末尾ドットで不自然な大空白を防ぐ
            for i, (w, is_b, is_l) in enumerate(line_toks):
                if i > 0:
                    row_chars.append(" ")
                    row_b.append(False)
                    row_l.append(False)
                row_chars.extend(list(w))
                row_b.extend([is_b] * len(w))
                row_l.extend([is_l] * len(w))
            rem_pad = seg_w - len(row_chars)
            if rem_pad > 0:
                pad_dots = (" " + "·" * rem_pad)[:rem_pad]
                row_chars.extend(list(pad_dots))
                row_b.extend([False] * rem_pad)
                row_l.extend([False] * rem_pad)
        else:
            for i, (w, is_b, is_l) in enumerate(line_toks):
                if i > 0:
                    row_chars.append(" ")
                    row_b.append(False)
                    row_l.append(False)
                row_chars.extend(list(w))
                row_b.extend([is_b] * len(w))
                row_l.extend([is_l] * len(w))
            rem = seg_w - len(row_chars)
            if rem > 0:
                short_end = " [END OF 30 DISPATCHES] "
                long_end = "  [END OF 30 REPORTED DISPATCHES - NOV 26, 2026] "
                chosen_end = long_end if rem >= len(long_end) else (short_end if rem >= len(short_end) else " ")
                tail = (chosen_end + ("· " * seg_w))[:rem]
                row_chars.extend(list(tail))
                row_b.extend([True] * len(tail))
                row_l.extend([False] * len(tail))

        for i in range(seg_w):
            char_grid[r][c0 + i] = row_chars[i]
            bold_grid[r, c0 + i] = row_b[i]
            lead_grid[r, c0 + i] = row_l[i]

    all_fit = tok_idx >= n_tok
    spare_segments = len(segments) - used_seg_count

    # 余った下部セグメントは幅に応じた完結文言で左右対称に埋める（途中切れ・"VERIFIED" なし）
    if all_fit and spare_segments > 0:
        wide_footers = [
            " 30 GLOBAL DISPATCHES COMPILED | REPORTED EDITION: NOV 26, 2026 | NO OMISSIONS ",
            " REGIONS: AMERICAS | EUROPE | MIDDLE EAST | AFRICA | ASIA-PACIFIC | ORBIT ",
        ]
        narrow_footers = [
            " 30 DISPATCHES COMPILED | NOV 26, 2026 ",
            " AMERICAS | EMEA | ASIA-PACIFIC | ORBIT ",
            " REPORTED WORLD EDITION | ITEMS #01-#30 ",
        ]
        f_i = 0
        for r, c0, c1 in segments[used_seg_count:]:
            seg_w = c1 - c0
            pool = wide_footers if seg_w >= 80 else narrow_footers
            tmpl = pool[f_i % len(pool)]
            f_i += 1
            if len(tmpl) <= seg_w:
                pad = seg_w - len(tmpl)
                lp = "=" * (pad // 2)
                rp = "=" * (pad - len(lp))
                line_s = lp + tmpl + rp
            else:
                line_s = "=" * seg_w
            for i, ch in enumerate(line_s):
                char_grid[r][c0 + i] = ch
                bold_grid[r, c0 + i] = True

    return char_grid, bold_grid, lead_grid, all_fit, spare_segments


def select_optimal_v3_layout(
    items: list[tuple[int, str, str]],
    mosaic_width: int,
    mosaic_height: int,
    font_path: str,
    reg_idx: int,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    list[HeroCardSpec],
    list[list[str]],
    np.ndarray,
    np.ndarray,
]:
    """3つの大見出しカードを避けつつ30件すべてが最大フォントで綺麗に収まるグリッドを選択する。"""
    tokens = build_tagged_tokens(items, sep_token="|")

    candidates = [
        (17, 10, 17),
        (16, 10, 17),
        (16, 10, 16),
        (15, 9, 16),
        (15, 9, 15),
        (14, 9, 15),
    ]

    for fs, cw, ch in candidates:
        f_reg = make_font(font_path, reg_idx, fs)
        cw_eff = max(cw, int(round(f_reg.getlength("M"))))
        cols = mosaic_width // cw_eff
        rows = mosaic_height // ch
        cards = build_hero_cards(cols, rows)
        char_grid, bold_grid, lead_grid, all_fit, spare_segs = wrap_tokens_around_cards(
            tokens, cols, rows, cards
        )
        if all_fit:
            # 余り行が2行以上ある場合は行高を広げて無駄な余り行を最小化する
            for ch_try in (ch + 2, ch + 1):
                rows_try = mosaic_height // ch_try
                cards_try = build_hero_cards(cols, rows_try)
                cg2, bg2, lg2, fit2, _ = wrap_tokens_around_cards(
                    tokens, cols, rows_try, cards_try
                )
                if fit2:
                    return fs, cw_eff, ch_try, cols, rows_try, cards_try, cg2, bg2, lg2
            return fs, cw_eff, ch, cols, rows, cards, char_grid, bold_grid, lead_grid

    raise RuntimeError("全30件を収容できるグリッドが見つかりませんでした。")


# ==============================================================================
# プロシージャルモチーフ生成 v3（二極化高コントラスト＆亀裂エッジ強化版）
# ==============================================================================
def _smooth_noise_2d(
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coarse = rng.random((grid_h + 1, grid_w + 1), dtype=np.float32)
    ys = np.linspace(0, grid_h, height, endpoint=False, dtype=np.float32)
    xs = np.linspace(0, grid_w, width, endpoint=False, dtype=np.float32)
    yi = np.floor(ys).astype(np.int32)
    xi = np.floor(xs).astype(np.int32)
    yf = ys - yi
    xf = xs - xi
    uy = yf * yf * (3.0 - 2.0 * yf)
    ux = xf * xf * (3.0 - 2.0 * xf)
    c00 = coarse[yi[:, None], xi[None, :]]
    c10 = coarse[yi[:, None], xi[None, :] + 1]
    c01 = coarse[yi[:, None] + 1, xi[None, :]]
    c11 = coarse[yi[:, None] + 1, xi[None, :] + 1]
    top = c00 + ux[None, :] * (c10 - c00)
    bot = c01 + ux[None, :] * (c11 - c01)
    return top + uy[:, None] * (bot - top)


def _fbm_2d(
    height: int,
    width: int,
    base_h: int,
    base_w: int,
    octaves: int,
    rng: np.random.Generator,
) -> np.ndarray:
    acc = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    gh, gw = base_h, base_w
    for _ in range(octaves):
        acc += amp * _smooth_noise_2d(height, width, gh, gw, rng)
        amp_sum += amp
        amp *= 0.5
        gh *= 2
        gw *= 2
    acc /= amp_sum
    lo, hi = float(acc.min()), float(acc.max())
    return (acc - lo) / (hi - lo + 1e-6)


def build_motif_layers_v3(
    cols: int,
    rows: int,
    aspect_w_over_h: float,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    モチーフのセル背景RGB `(rows, cols, 3)` と、
    「黒文字を使うべき超高輝度コアマスク `is_bright_core` `(rows, cols)`」を返す。

    ★ v3 配色原則：
    - 宇宙・海・緑の大陸・赤い関税の壁・ロケット外縁グローはすべて「深みのある濃色（輝度 <= 95）」に設計し、
      100% 白系文字でクッキリ統一（海や緑の上で黒文字が混じる現象を根絶）。
    - 神舟24号の軌跡コア＆先端スターバーストのみ「高輝度イエロー〜アイボリー（輝度 >= 215）」へ一気にジャンプさせ、
      そこだけ濃色文字にする（中間ベージュの死角ゼロ）。
    """
    rng = np.random.default_rng(seed)

    ys = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    xs = (np.arange(cols, dtype=np.float32) + 0.5) / cols
    xx, yy = np.meshgrid(xs, ys)
    xx_iso = (xx - 0.5) * aspect_w_over_h

    # ------------------------------------------------------------------
    # 1. 深い宇宙空間（ミッドナイトネイビー）
    # ------------------------------------------------------------------
    nebula = _fbm_2d(rows, cols, 4, 6, 4, rng)
    space_top = np.array([7.0, 13.0, 32.0], dtype=np.float32)
    space_bot = np.array([14.0, 26.0, 62.0], dtype=np.float32)
    rgb = space_top[None, None, :] * (1.0 - yy[:, :, None]) + space_bot[None, None, :] * yy[:, :, None]

    neb_mask = np.clip((nebula - 0.50) * 1.4, 0.0, 0.45)
    rgb = rgb * (1.0 - neb_mask[:, :, None]) + np.array([26.0, 48.0, 98.0], dtype=np.float32)[None, None, :] * neb_mask[:, :, None]

    star_roll = rng.random((rows, cols), dtype=np.float32)
    stars = (star_roll > 0.985) & (yy < 0.48)
    rgb = np.where(stars[:, :, None], np.array([52.0, 84.0, 148.0], dtype=np.float32)[None, None, :], rgb)

    # ------------------------------------------------------------------
    # 2. 地球シルエット（輪郭がより明確に伝わる球体配置＋大気リム＋海と大陸）
    #    ※「壁で分断された地球」がサムネイルでも分かるよう、関税の壁の箇所で上下半球に少し食い違い・亀裂を入れる
    # ------------------------------------------------------------------
    earth_cx_iso = -0.03 * aspect_w_over_h
    earth_cy = 0.88
    earth_r = 0.58

    dx_e = xx_iso - earth_cx_iso
    dy_e = yy - earth_cy
    dist_e = np.hypot(dx_e, dy_e)

    # 地球外周の鮮やかなブルー大気リング（白文字がしっかり読める明度に制御）
    atmo_ring = np.exp(-(((dist_e - earth_r) / 0.032) ** 2)) * (dist_e >= earth_r * 0.95)
    atmo_rgb = np.array([28.0, 118.0, 206.0], dtype=np.float32)
    rgb = rgb * (1.0 - 0.85 * atmo_ring[:, :, None]) + atmo_rgb[None, None, :] * (0.85 * atmo_ring[:, :, None])

    in_earth = dist_e <= earth_r
    nz = np.sqrt(np.clip(1.0 - (dist_e / earth_r) ** 2, 0.0, 1.0))
    nx_s = dx_e / earth_r
    ny_s = dy_e / earth_r
    sun_dot = np.clip(0.28 * nx_s - 0.55 * ny_s + 0.78 * nz, 0.0, 1.0)

    # 海：白文字とのコントラスト比 7:1 以上を保つリッチなディープサファイア〜コバルトブルー
    ocean_deep = np.array([9.0, 44.0, 112.0], dtype=np.float32)
    ocean_lit = np.array([18.0, 86.0, 174.0], dtype=np.float32)
    ocean_rgb = ocean_deep[None, None, :] * (1.0 - sun_dot[:, :, None]) + ocean_lit[None, None, :] * sun_dot[:, :, None]

    # 大陸：白文字がクッキリ映えるエメラルド〜フォレストグリーン（輝度を抑えつつ高彩度）
    coast_noise = _fbm_2d(rows, cols, 8, 12, 4, rng)
    continent_blobs = [
        (-0.32, 0.58, 0.13, 0.085),
        (-0.25, 0.68, 0.09, 0.070),
        (-0.20, 0.80, 0.08, 0.105),
        (-0.02, 0.44, 0.10, 0.065),
        (0.00, 0.61, 0.10, 0.115),
        (0.10, 0.53, 0.075, 0.060),
        (0.19, 0.47, 0.14, 0.085),
        (0.15, 0.58, 0.075, 0.068),
        (0.29, 0.56, 0.085, 0.060),
        (0.31, 0.76, 0.09, 0.068),
    ]
    land_field = np.zeros((rows, cols), dtype=np.float32)
    for bx, by, brx, bry in continent_blobs:
        d_b = np.hypot((xx_iso - bx) / brx, (yy - by) / bry)
        land_field = np.maximum(land_field, 1.15 - d_b)

    is_land = in_earth & ((land_field + 0.32 * (coast_noise - 0.48)) > 0.37)

    land_dark = np.array([14.0, 74.0, 38.0], dtype=np.float32)
    land_lit = np.array([28.0, 122.0, 56.0], dtype=np.float32)
    land_mix = np.clip(0.72 * sun_dot + 0.28 * coast_noise, 0.0, 1.0)
    land_rgb = land_dark[None, None, :] * (1.0 - land_mix[:, :, None]) + land_lit[None, None, :] * land_mix[:, :, None]

    earth_rgb = np.where(is_land[:, :, None], land_rgb, ocean_rgb)
    # 地球の丸い輪郭を際立たせるシャープなリムハイライト
    rim_band = (dist_e >= earth_r * 0.93) & in_earth
    earth_rgb = np.where(
        rim_band[:, :, None],
        earth_rgb * 0.45 + np.array([34.0, 126.0, 214.0], dtype=np.float32)[None, None, :] * 0.55,
        earth_rgb,
    )
    rgb = np.where(in_earth[:, :, None], earth_rgb, rgb)

    # ------------------------------------------------------------------
    # 3. 地球を真っ二つに分断する「赤い関税の壁（Tariff Wall）」＋上下の暗い亀裂
    # ------------------------------------------------------------------
    wall_center_y = 0.575 - 0.13 * (xx - 0.48) + 0.045 * ((xx - 0.48) ** 2)
    wall_half_thick = 0.048
    chasm_thick = 0.020  # 壁の上下に入る暗い断層スリット（サムネイルでの分断視認性を劇的向上）

    in_wall_xspan = dist_e <= (earth_r * 1.08)

    # 城壁・バリケード状のブロック凹凸
    block_wave = np.sin(xx * math.pi * 28.0)
    crenel = 0.006 * np.where(block_wave > 0.0, 1.0, -0.6)
    dy_wall = yy - (wall_center_y + crenel * (yy < wall_center_y))

    # 上下の暗い断層スリット（地球が壁で断ち切られている視覚効果）
    in_chasm = in_earth & (np.abs(dy_wall) > wall_half_thick) & (np.abs(dy_wall) <= wall_half_thick + chasm_thick)
    chasm_rgb = np.array([10.0, 8.0, 20.0], dtype=np.float32)
    rgb = np.where(in_chasm[:, :, None], chasm_rgb[None, None, :], rgb)

    in_wall = in_wall_xspan & (np.abs(dy_wall) <= wall_half_thick)
    wall_v = (dy_wall + wall_half_thick) / (2.0 * wall_half_thick)
    brick_seam = 0.86 + 0.14 * np.where(block_wave > -0.85, 1.0, 0.55)

    # 白文字が100%映える高彩度クリムゾン〜バーミリオンレッド
    wall_top = np.array([216.0, 48.0, 26.0], dtype=np.float32)
    wall_mid = np.array([182.0, 20.0, 26.0], dtype=np.float32)
    wall_bot = np.array([124.0, 12.0, 20.0], dtype=np.float32)

    wall_rgb = np.where(
        (wall_v < 0.35)[:, :, None],
        wall_top[None, None, :] * (1.0 - (wall_v / 0.35)[:, :, None])
        + wall_mid[None, None, :] * (wall_v / 0.35)[:, :, None],
        wall_mid[None, None, :] * (1.0 - ((wall_v - 0.35) / 0.65)[:, :, None])
        + wall_bot[None, None, :] * ((wall_v - 0.35) / 0.65)[:, :, None],
    )
    wall_rgb *= brick_seam[:, :, None]
    rgb = np.where(in_wall[:, :, None], wall_rgb, rgb)

    # ------------------------------------------------------------------
    # 4. 神舟24号の打ち上げ軌跡（中間ベージュを完全排除した二段階発光）
    #    地球右側 (p0) → 右上宇宙空間 Card2 の右隣 (p2) へ伸びる黄金〜白光の弧
    # ------------------------------------------------------------------
    p0 = (0.14 * aspect_w_over_h, 0.45)
    p1 = (0.28 * aspect_w_over_h, 0.24)
    p2 = (0.40 * aspect_w_over_h, 0.075)  # Card2のすぐ右隣で輝く先端スターバースト

    t_s = np.linspace(0.0, 1.0, 180, dtype=np.float32)
    bx_c = ((1 - t_s) ** 2) * p0[0] + 2 * (1 - t_s) * t_s * p1[0] + (t_s**2) * p2[0]
    by_c = ((1 - t_s) ** 2) * p0[1] + 2 * (1 - t_s) * t_s * p1[1] + (t_s**2) * p2[1]

    dx_c = xx_iso[:, :, None] - bx_c[None, None, :]
    dy_c = yy[:, :, None] - by_c[None, None, :]
    d2_c = dx_c * dx_c + dy_c * dy_c
    idx_min = np.argmin(d2_c, axis=-1)
    min_dist = np.sqrt(np.take_along_axis(d2_c, idx_min[:, :, None], axis=-1)[:, :, 0])
    t_curve = t_s[idx_min]

    core_r = 0.017 + 0.012 * np.sin(t_curve * math.pi * 0.85)
    halo_r = core_r * 2.1

    # 先端スターバースト
    dx_tip = xx_iso - p2[0]
    dy_tip = yy - p2[1]
    dist_tip = np.hypot(dx_tip, dy_tip)
    cross_flare = (
        np.exp(-((dy_tip / 0.013) ** 2)) * np.exp(-((dx_tip / 0.078) ** 2))
        + np.exp(-((dx_tip / 0.013) ** 2)) * np.exp(-((dy_tip / 0.078) ** 2))
    )

    # 【外縁グロー】：暗色側（輝度 <= 65）に留め、白文字が沈まないディープアンバーにする
    in_outer_halo = (min_dist <= halo_r) | (dist_tip <= 0.062)
    halo_amber_dark = np.array([92.0, 52.0, 16.0], dtype=np.float32)
    rgb = np.where(
        in_outer_halo[:, :, None],
        rgb * 0.35 + halo_amber_dark[None, None, :] * 0.65,
        rgb,
    )

    # 【超高輝度コア＆先端輝き】：一気に輝度 >= 220 の鮮やかなイエロー〜アイボリーホワイトへ振り切る！
    is_bright_core = (
        (min_dist <= core_r)
        | (dist_tip <= 0.034)
        | (cross_flare > 0.48)
        | (np.hypot(xx_iso - p0[0], yy - p0[1]) <= 0.026)
    )
    whiteness = np.clip(
        np.maximum(1.0 - min_dist / (core_r + 1e-6), np.clip(1.0 - dist_tip / 0.040, 0.0, 1.0)),
        0.0,
        1.0,
    )
    core_yellow = np.array([255.0, 224.0, 78.0], dtype=np.float32)
    core_ivory = np.array([255.0, 252.0, 230.0], dtype=np.float32)
    bright_rgb = (
        core_yellow[None, None, :] * (1.0 - whiteness[:, :, None])
        + core_ivory[None, None, :] * whiteness[:, :, None]
    )
    rgb = np.where(is_bright_core[:, :, None], bright_rgb, rgb)

    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8), is_bright_core


def compute_v3_dual_colors(
    motif_rgb: np.ndarray,
    is_bright_core: np.ndarray,
    bold_grid: np.ndarray,
    lead_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    二極化されたセル背景色と文字色を生成：
    - `is_bright_core == False`（全体の約96%：宇宙・海・緑の大陸・赤い壁・暗色アンバー外縁）：
      すべて明るい白系ティント文字（タグは純白、注目記事タグは鮮やかなゴールドイエロー文字＋濃色背景）
    - `is_bright_core == True`（ロケットの白〜黄色コアのみ）：
      引き締まったジェットブラック〜エスプレッソ濃色文字
    """
    bg = motif_rgb.astype(np.float32).copy()

    # 注目記事タグ (#04, #07, #16, #18, #21) が暗色セル上にある場合、セル背景を少し引き締めてゴールド文字で際立たせる
    lead_on_dark = lead_grid & (~is_bright_core)
    bg = np.where(lead_on_dark[:, :, None], bg * 0.65, bg)

    white = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    gold_tag = np.array([255.0, 236.0, 110.0], dtype=np.float32)

    # 暗色背景上の文字色（通常本文=86%ホワイト混色で背景色相をほんのり残しつつ超高コントラスト、タグ=純白/ゴールド）
    fg_dark_zone = bg * 0.14 + white[None, None, :] * 0.86
    fg_dark_zone = np.where(bold_grid[:, :, None], white[None, None, :], fg_dark_zone)
    fg_dark_zone = np.where(lead_on_dark[:, :, None], gold_tag[None, None, :], fg_dark_zone)

    # 明色コア（ロケット光条）上の文字色（濃黒ブラウン）
    fg_bright_zone = np.where(
        bold_grid[:, :, None],
        np.array([8.0, 5.0, 2.0], dtype=np.float32)[None, None, :],
        np.array([24.0, 15.0, 6.0], dtype=np.float32)[None, None, :],
    )

    fg = np.where(is_bright_core[:, :, None], fg_bright_zone, fg_dark_zone)

    return (
        np.clip(np.rint(bg), 0, 255).astype(np.uint8),
        np.clip(np.rint(fg), 0, 255).astype(np.uint8),
    )


# ==============================================================================
# 大見出しカード描画 & 全体レンダリング
# ==============================================================================
def draw_hero_cards(
    image: Image.Image,
    cards: Sequence[HeroCardSpec],
    font_path: str,
    bold_idx: int,
    reg_idx: int,
    header_height: int,
    pad_left: int,
    pad_top: int,
    cell_w: int,
    cell_h: int,
) -> None:
    """予約されたくり抜き領域に、視線を掴むトップニュース大見出しカードを描画する。"""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    f_badge = make_font(font_path, bold_idx, 11)
    f_head = make_font(font_path, bold_idx, 20)
    f_sub = make_font(font_path, reg_idx, 12)

    for card in cards:
        x0 = pad_left + card.col_start * cell_w + 2
        y0 = header_height + pad_top + card.row_start * cell_h + 3
        x1 = pad_left + card.col_end * cell_w - 2
        y1 = header_height + pad_top + card.row_end * cell_h - 3

        # 外側の微細なドロップシャドウ枠 + 半透明ダークガラス背景
        odraw.rounded_rectangle(
            [x0 - 1, y0 - 1, x1 + 1, y1 + 1],
            radius=6,
            fill=(4, 8, 18, 235),
            outline=(*card.accent_rgb, 230),
            width=2,
        )
        odraw.rounded_rectangle(
            [x0 + 1, y0 + 1, x1 - 1, y1 - 1],
            radius=5,
            fill=(*card.bg_rgb, 232),
        )
        # 左端の太いアクセントカラーバー
        odraw.rounded_rectangle(
            [x0 + 1, y0 + 1, x0 + 7, y1 - 1],
            radius=3,
            fill=(*card.accent_rgb, 255),
        )

        tx = x0 + 14
        max_text_w = (x1 - tx) - 8

        # 1段目：カテゴリバッジ
        odraw.text((tx, y0 + 7), card.badge, font=f_badge, fill=(*card.accent_rgb, 255))

        # 2段目：大見出し（幅に収まる最大サイズを自動選択）
        cur_head_font = f_head
        for try_sz in (20, 19, 18, 17, 16):
            cand_f = make_font(font_path, bold_idx, try_sz)
            if cand_f.getlength(card.headline) <= max_text_w:
                cur_head_font = cand_f
                break
        odraw.text((tx, y0 + 23), card.headline, font=cur_head_font, fill=(255, 255, 255, 255))

        # 3段目・4段目：要点箇条書き2行（カードの空白をなくし情報階層を強化）
        for idx_b, b_text in enumerate((card.bullet1, card.bullet2)):
            cur_sub_font = f_sub
            for try_sz in (12, 11, 10):
                cand_f = make_font(font_path, reg_idx, try_sz)
                if cand_f.getlength(b_text) <= max_text_w:
                    cur_sub_font = cand_f
                    break
            by = y0 + 52 + idx_b * 18
            if by + 13 <= y1 - 3:
                odraw.text((tx, by), b_text, font=cur_sub_font, fill=(214, 232, 252, 255))

    # 右上ショーケース窓（Card2の右隣・神舟24号スターバースト先端部）に控えめな軌道ラベルとポインタを添える
    if len(cards) >= 2:
        c2 = cards[1]
        c2_x1 = pad_left + c2.col_end * cell_w - 2
        c2_y0 = header_height + pad_top + c2.row_start * cell_h + 4
        c2_y1 = header_height + pad_top + c2.row_end * cell_h - 3
        py_mid = (c2_y0 + c2_y1) // 2
        # Card2右縁から光条先端へ向かう短いゴールドポインタ（本文がない領域なので一切文字と干渉しない）
        odraw.line([(c2_x1 + 1, py_mid), (c2_x1 + 18, py_mid)], fill=(*c2.accent_rgb, 255), width=2)
        odraw.polygon(
            [(c2_x1 + 24, py_mid), (c2_x1 + 16, py_mid - 5), (c2_x1 + 16, py_mid + 5)],
            fill=(*c2.accent_rgb, 255),
        )
        f_cap = make_font(font_path, bold_idx, 10)
        cap_text = "SZ-24 ORBIT"
        cap_w = int(f_cap.getlength(cap_text))
        cap_x = c2_x1 + max(10, ((image.width - c2_x1) - cap_w) // 2)
        cap_y = c2_y1 - 14
        odraw.rounded_rectangle(
            [cap_x - 5, cap_y - 2, cap_x + cap_w + 5, cap_y + 13],
            radius=3,
            fill=(8, 14, 32, 220),
            outline=(255, 214, 64, 200),
            width=1,
        )
        odraw.text((cap_x, cap_y), cap_text, font=f_cap, fill=(255, 232, 120, 255))

    composited = Image.alpha_composite(image.convert("RGBA"), overlay)
    image.paste(composited.convert("RGB"))


def render_competition_entry_v3(
    md_path: Path,
    canvas_width: int = CANVAS_WIDTH,
    canvas_height: int = CANVAS_HEIGHT,
    header_height: int = HEADER_HEIGHT,
    seed: int = RANDOM_SEED,
) -> tuple[Image.Image, dict[str, int | str]]:
    items = parse_news_items(md_path)
    font_path, reg_idx, bold_idx = find_font_family()
    mosaic_height = canvas_height - header_height

    fs, cell_w, cell_h, cols, rows, cards, char_grid, bold_grid, lead_grid = (
        select_optimal_v3_layout(
            items=items,
            mosaic_width=canvas_width,
            mosaic_height=mosaic_height,
            font_path=font_path,
            reg_idx=reg_idx,
        )
    )

    f_reg = make_font(font_path, reg_idx, fs)
    f_bold = make_font(font_path, bold_idx, fs)
    f_h1 = make_font(font_path, bold_idx, 26)
    f_h2 = make_font(font_path, bold_idx, 13)

    motif_rgb, is_bright_core = build_motif_layers_v3(
        cols=cols,
        rows=rows,
        aspect_w_over_h=canvas_width / mosaic_height,
        seed=seed,
    )
    cell_bg_rgb, text_fg_rgb = compute_v3_dual_colors(
        motif_rgb=motif_rgb,
        is_bright_core=is_bright_core,
        bold_grid=bold_grid,
        lead_grid=lead_grid,
    )

    # ------------------------------------------------------------------
    # 1. ヘッダー＋セル背景レイヤーのピクセル構築
    # ------------------------------------------------------------------
    canvas_arr = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    hy = np.linspace(0.0, 1.0, header_height, dtype=np.float32)[:, None, None]
    h_top = np.array(HEADER_BG_TOP, dtype=np.float32)[None, None, :]
    h_bot = np.array(HEADER_BG_BOTTOM, dtype=np.float32)[None, None, :]
    header_arr = np.clip(np.rint(h_top * (1.0 - hy) + h_bot * hy), 0, 255).astype(np.uint8)
    canvas_arr[:header_height, :, :] = np.broadcast_to(
        header_arr, (header_height, canvas_width, 3)
    )
    canvas_arr[header_height - 2 : header_height, :, :] = np.array(
        HEADER_ACCENT_LINE, dtype=np.uint8
    )

    grid_w = cols * cell_w
    grid_h = rows * cell_h
    pad_left = (canvas_width - grid_w) // 2
    pad_top = (mosaic_height - grid_h) // 2

    px_cols = np.clip((np.arange(canvas_width) - pad_left) // cell_w, 0, cols - 1)
    px_rows = np.clip((np.arange(mosaic_height) - pad_top) // cell_h, 0, rows - 1)
    canvas_arr[header_height:, :, :] = cell_bg_rgb[px_rows[:, None], px_cols[None, :]]

    image = Image.fromarray(canvas_arr, mode="RGB")
    draw = ImageDraw.Draw(image)

    # ------------------------------------------------------------------
    # 2. 最上部題字ヘッダー描画
    # ------------------------------------------------------------------
    t1_bbox = draw.textbbox((0, 0), HEADER_TITLE, font=f_h1)
    t1_x = (canvas_width - (t1_bbox[2] - t1_bbox[0])) // 2 - t1_bbox[0]
    t1_y = 11 - t1_bbox[1]
    draw.text((t1_x, t1_y), HEADER_TITLE, font=f_h1, fill=(255, 255, 255))

    t2_bbox = draw.textbbox((0, 0), HEADER_SUBTITLE, font=f_h2)
    t2_x = (canvas_width - (t2_bbox[2] - t2_bbox[0])) // 2 - t2_bbox[0]
    t2_y = 49 - t2_bbox[1]
    draw.text((t2_x, t2_y), HEADER_SUBTITLE, font=f_h2, fill=(138, 222, 255))

    # ------------------------------------------------------------------
    # 3. 回り込み等幅グリッド文字の描画（変形・回転なし）
    # ------------------------------------------------------------------
    sample_bbox = f_reg.getbbox("Mg")
    glyph_h = sample_bbox[3] - sample_bbox[1]
    y_glyph_offset = max(0, (cell_h - glyph_h) // 2) - sample_bbox[1]
    x_glyph_offset = max(0, int(round((cell_w - f_reg.getlength("M")) / 2.0)))

    drawn_chars = 0
    for r in range(rows):
        y = header_height + pad_top + r * cell_h + y_glyph_offset
        for c in range(cols):
            ch = char_grid[r][c]
            if ch == " ":
                continue
            x = pad_left + c * cell_w + x_glyph_offset
            fg_tuple = (
                int(text_fg_rgb[r, c, 0]),
                int(text_fg_rgb[r, c, 1]),
                int(text_fg_rgb[r, c, 2]),
            )
            use_f = f_bold if bold_grid[r, c] else f_reg
            draw.text((x, y), ch, font=use_f, fill=fg_tuple)
            drawn_chars += 1

    # ------------------------------------------------------------------
    # 4. くり抜き領域へのトップ3大見出しカード描画
    # ------------------------------------------------------------------
    draw_hero_cards(
        image=image,
        cards=cards,
        font_path=font_path,
        bold_idx=bold_idx,
        reg_idx=reg_idx,
        header_height=header_height,
        pad_left=pad_left,
        pad_top=pad_top,
        cell_w=cell_w,
        cell_h=cell_h,
    )

    # ------------------------------------------------------------------
    # 5. 全30件の完全収録検証（カード領域に1文字も隠れていないことを保証）
    # ------------------------------------------------------------------
    blocked_mask, _ = compute_row_segments(cols, rows, cards)
    overlapped_chars = sum(
        1
        for r in range(rows)
        for c in range(cols)
        if blocked_mask[r, c] and char_grid[r][c] != " "
    )
    if overlapped_chars != 0:
        raise AssertionError(f"カード領域とモザイク文字が重複しています: {overlapped_chars} 文字")

    rendered_stream = re.sub(
        r"\s+",
        " ",
        " ".join("".join(char_grid[r]) for r in range(rows)),
    )
    included_count = 0
    for _, tag, body in items:
        if tag in rendered_stream and body[:25] in rendered_stream and body[-20:] in rendered_stream:
            included_count += 1

    pure_news_chars = sum(len(tag) + 1 + len(body) for _, tag, body in items)
    stats: dict[str, int | str] = {
        "font_path": font_path,
        "font_size": fs,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "cols": cols,
        "rows": rows,
        "total_cells": cols * rows,
        "drawn_chars": drawn_chars,
        "news_count": len(items),
        "included_count": included_count,
        "pure_news_chars": pure_news_chars,
        "hero_cards": len(cards),
    }
    return image, stats


def save_deterministic_png(
    image: Image.Image,
    output_path: Path,
    max_bytes: int = MAX_FILE_BYTES,
) -> tuple[int, str]:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True, compress_level=9)
    data = buf.getvalue()
    desc = "RGB lossless PNG (optimize=True)"

    if len(data) > max_bytes:
        for colors in (256, 224, 192, 160, 128, 96):
            q_img = image.quantize(
                colors=colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            q_buf = io.BytesIO()
            q_img.save(q_buf, format="PNG", optimize=True, compress_level=9)
            cand = q_buf.getvalue()
            if len(cand) <= max_bytes:
                data = cand
                desc = f"Indexed PNG ({colors} colors, dither=NONE, optimize=True)"
                break

    if len(data) > max_bytes:
        raise RuntimeError(f"1 MiB 以下に圧縮できませんでした: {len(data):,} bytes")

    output_path.write_bytes(data)
    return len(data), desc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2026年11月26日版 ニュース二層モザイク v3（回り込み大見出しカード搭載）を生成します。"
    )
    parser.add_argument("--input", default=INPUT_MD_PATH, help="入力Markdownパス")
    parser.add_argument("--output", default=OUTPUT_PNG_PATH, help="出力PNGパス (既定: entry.png)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="乱数シード")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.output)

    image, stats = render_competition_entry_v3(
        md_path=Path(args.input),
        seed=args.seed,
    )
    byte_len, mode_desc = save_deterministic_png(image, out_path)
    sha256_hex = hashlib.sha256(out_path.read_bytes()).hexdigest()

    if stats["included_count"] != stats["news_count"]:
        raise AssertionError(
            f"ニュース収録漏れエラー: {stats['included_count']}/{stats['news_count']} 件"
        )

    print("=== コンペ提出画像 v3 生成レポート ===")
    print(f"出力ファイル : {out_path} ({CANVAS_WIDTH}x{CANVAS_HEIGHT} px)")
    print(f"ファイル容量 : {byte_len:,} bytes ({byte_len / 1024:.1f} KiB <= 1,024.0 KiB) [{mode_desc}]")
    print(f"使用フォント : {stats['font_path']} (本文 {stats['font_size']}px, セル {stats['cell_w']}x{stats['cell_h']}px)")
    print(
        f"グリッド構成 : {stats['cols']}列 × {stats['rows']}行 = {stats['total_cells']:,} マス "
        f"(大見出しカード {stats['hero_cards']}枠 + 回り込み印字 {stats['drawn_chars']:,} 文字)"
    )
    print(
        f"記事収録確認 : {stats['included_count']}/{stats['news_count']} 件すべて完全収録・カード重なりゼロ "
        f"(計 {stats['pure_news_chars']:,} 文字)"
    )
    print(f"SHA-256      : {sha256_hex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
