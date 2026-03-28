"""
Thai Spell Checker — Streamlit Application
ตรวจสะกดภาษาไทยจากรูปภาพและวิดีโอ ด้วย EasyOCR + PyThaiNLP
"""

import re
import html
import time
import json
import datetime
import tempfile
from difflib import SequenceMatcher
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from PIL import Image
import streamlit as st
from pythainlp.tokenize import word_tokenize
from pythainlp.spell import correct
from pythainlp.corpus.common import thai_words

# ─────────────────────────────────────────────
# ค่าคงที่และเส้นทางไฟล์
# ─────────────────────────────────────────────
BRANDS_FILE      = Path(__file__).parent / "brands.json"
WHITELIST_FILE   = Path(__file__).parent / "whitelist.json"
CORRECTIONS_FILE = Path(__file__).parent / "ocr_corrections.json"
PHONES_FILE      = Path(__file__).parent / "phones.json"
PHRASES_FILE     = Path(__file__).parent / "phrases.json"

UNIT_WORDS   = ["แคปซูล", "เม็ด", "ซอง", "ขวด", "กล่อง"]
UNIT_OPTIONS = ["แคปซูล", "เม็ด", "ซอง", "ขวด", "กล่อง"]

# Whitelist เริ่มต้น — คำสุขภาพ แบรนด์ หน่วยนับ คำโฆษณา คำนวัตกรรม รางวัล สถาบัน
# และคำในวิดีโอที่ไม่ควรแจ้งว่าผิด
DEFAULT_WHITELIST: set = {
    # ──────────────────────────────────────────────
    # ชื่อแบรนด์ ภาษาอังกฤษ
    # ──────────────────────────────────────────────
    "betaoil", "betalife", "betaherb", "betaxplus", "betalivproplus", "betaliv",
    "labfarm", "betacalproplus", "hopeful", "beta", "Hopeful", "Beta",
    # ชื่อแบรนด์ ภาษาไทย
    "เบต้าออยล์", "เบต้าไลฟ์", "เบต้าเฮิร์บ", "เบต้าเอ็กซ์พลัส",
    "เบต้าลีฟโปรพลัส", "เบต้าลีฟ", "แล็บฟาร์ม", "แลปฟาร์ม", "แล็ปฟาร์ม",
    "เบต้าแคลโปรพลัส", "โฮปฟูล",

    # ──────────────────────────────────────────────
    # หน่วยนับ
    # ──────────────────────────────────────────────
    "แคปซูล", "เม็ด", "กล่อง", "ซอง", "ขวด",

    # ──────────────────────────────────────────────
    # คำสุขภาพ / โรค / อวัยวะ
    # ──────────────────────────────────────────────
    "คอเลสเตอรอล", "ไตรกลีเซอไรด์", "อินซูลิน", "ดื้ออินซูลิน", "เบาหวาน",
    "ความดัน", "ไขมัน", "หลอดเลือด", "ตับ", "ไต", "ปอด", "ภูมิคุ้มกัน",
    "หัวใจ", "เลือด", "น้ำตาลในเลือด", "น้ำตาล",
    # โรคไต
    "เบาหวานลงไต", "ไตวาย", "ไตวายเรื้อรัง", "ไตเรื้อรัง", "โรคไต",
    "ภาวะไตวาย", "ค่าไตสูง", "ค่าครีเอตินีน", "โปรตีนรั่ว", "ฟอกไต", "ล้างไต",
    "diabetic", "nephropathy",
    # ตับ
    "ไขมันพอกตับ", "ตับอักเสบ", "ตับแข็ง", "มะเร็งตับ", "ค่าตับ",
    # ภูมิคุ้มกัน / มะเร็ง
    "มะเร็ง", "เนื้องอก", "คีโม", "ฉายแสง", "NK", "Cells",
    "เม็ดเลือดขาว", "เซลล์ภูมิคุ้มกัน", "อนุมูลอิสระ",
    # ระบบหายใจ
    "หอบหืด", "ภูมิแพ้", "ไอเรื้อรัง", "ฝุ่นควัน",
    # ค่าเลือด
    "HbA1c", "LDL", "HDL", "SGOT", "SGPT", "GFR", "eGFR",

    # ──────────────────────────────────────────────
    # สารสกัด / วัตถุดิบ
    # ──────────────────────────────────────────────
    "เปปไทด์", "หลินจือ", "หลินจือแดง", "ไคโตซาน", "มะระขี้นก",
    "เบต้ากลูแคน", "ซีบัคธอร์น", "โกจิเบอร์รี่", "แดนดิไลออน",
    "เจียวกู่หลาน", "รังผึ้ง", "แอลกอฮอล์",
    "เบต้าแคโรทีน", "ส้มขม", "ลูกพลับ", "ถั่วเขียว", "แอลกอฮอล์",
    "ออริซานอล", "โคลีน",

    # ──────────────────────────────────────────────
    # นวัตกรรม / เทคโนโลยี (ชื่อเฉพาะ)
    # ──────────────────────────────────────────────
    "เทคโนโลยีเปปไทด์", "ทุติยภูมิ",
    # ภาษาอังกฤษ
    "Immuneo", "Plus", "M-Gard", "Liverean", "ALA", "Maqui", "Berry",
    "Quercetin", "BETA", "OIL", "LIFE", "HERB", "LAB", "FARM",
    "LIVOX", "LIVOTOX", "CalFusion", "Rice", "Peptide",
    "Natto", "Artichoke", "Buckwheat", "Ginger", "Aegle", "Ginkgo", "Ginseng",
    "Gamma", "Oryzanol", "Vitamin", "Alpha", "Lipoic", "Acid",
    "Golden", "Ratio", "M-Gard",
    # อิมมูนีโอพลัส ทั้ง 2 รูปแบบ
    "อิมมูนีโอพลัส", "อิมมูนีโอ",

    # ──────────────────────────────────────────────
    # รางวัล / สถาบัน
    # ──────────────────────────────────────────────
    "เหรียญทอง", "เหรียญเงิน", "รางวัล", "นวัตกรรม", "สิทธิบัตร",
    "รางวัลระดับโลก", "งานวิจัย",
    # ชื่อรางวัล/เมือง
    "เจนีวา", "Geneva", "Silicon", "Valley", "KIPA", "Seoul",
    # สถาบัน
    "วว", "GMP", "Toyo", "Hakko",
    "มหาวิทยาลัยรังสิต", "เภสัชวิทยา", "พิษวิทยา",
    "สถาบันวิจัยวิทยาศาสตร์", "สวิตเซอร์แลนด์", "นอร์เวย์", "โรมาเนีย",

    # ──────────────────────────────────────────────
    # คำโฆษณา / การขาย
    # ──────────────────────────────────────────────
    "โปรโมชั่น", "โปรโมชั่นพิเศษ", "สั่งซื้อ", "สั่งซื้อเลย",
    "จัดส่งฟรี", "ส่งฟรี", "ส่งฟรีทั่วประเทศ",
    "ของแท้", "ของแท้100", "รับประกัน", "รับประกันของแท้",
    "ปรึกษาผู้เชี่ยวชาญสายตรง", "ปรึกษาผู้เชี่ยวชาญ", "ปรึกษาฟรี",
    "เก็บเงินปลายทาง", "มีผู้เชี่ยวชาญที่คอยดูแล",
    "โทรเลย", "แอดไลน์", "LINE", "จากปกติ",

    # ──────────────────────────────────────────────
    # คำทั่วไปในวิดีโอสุขภาพ
    # ──────────────────────────────────────────────
    "คุมน้ำตาล", "ลดน้ำตาล", "คุมน้ำตาลให้ดี", "ลดไขมัน",
    "บำรุงตับ", "บำรุงไต", "บำรุงปอด", "ดูแลหัวใจ", "เสริมภูมิ",
    "ผู้เชี่ยวชาญ", "สายตรง", "จดแจ้ง", "มาตรฐาน", "มาตรฐานสากล",
    "งานวิจัย", "สารสกัด", "ธรรมชาติ", "ปลอดภัย",
    "ไม่มีผลข้างเคียง", "ผลิตภัณฑ์เสริมอาหาร",

    # ──────────────────────────────────────────────
    # เลข อย. / กฎหมาย
    # ──────────────────────────────────────────────
    "อย", "ทะเบียน",
}

# ตาราง OCR อ่านผิดเริ่มต้น — คู่ (อ่านผิด → ที่ถูกต้อง)
DEFAULT_OCR_CORRECTIONS: dict[str, str] = {
    # ผ ถูกอ่านเป็น ม/ย
    "มู้":     "ผู้",   "ยู้":     "ผู้",   "มู":     "ผู",   "ยู":     "ผู",
    "มั":      "ผั",    "ยั":      "ผั",    "มา":     "ผา",   "ยา":     "ผา",
    "มิ":      "ผิ",    "มี":      "ผี",    "มึ":     "ผึ",   "มื":     "ผื",
    "มุ":      "ผุ",    "มเ":     "ผเ",
    # บ/ป สับสน
    "บ้":      "ป้",    "บ่":      "ป่",    "บ๊":     "ป๊",
    "ปั":      "บั",    "ปุ":      "บุ",
    # ด/ต สับสน
    "ดั":      "ตั",    "ดา":     "ตา",    "ดี":     "ตี",
    # หนิ/หนี → นิ/นี
    "หนิ":     "นิ",    "หนี":     "นี",
    # คำผิดบ่อย
    "ฉัน":     "กัน",
    # OCR อ่านสลับอักษร — รค (ร+ค) มักถูกอ่านแทน คุ
    "รค":      "คุ",
    # OCR อ่านแทนที่สระ
    "เเ":      "แ",     "น่าตาล":  "น้ำตาล",
    # คำประสมที่ OCR มักอ่านผิด
    "ตดเด็ม":  "เต็ม",  "ชึกิญ":   "เชี่ยวชาญ",
    "ไมปผอ":   "ไม่พอ", "วาส":     "สาว",
    "ไลฟ":     "ไลฟ์",
}


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class BrandIssue:
    issue_type:    str   # "brand_thai_misspell" | "wrong_unit"
    brand_id:      str
    brand_english: str
    brand_thai:    str   # ชื่อภาษาไทยที่ถูกต้อง
    found:         str   # สิ่งที่พบในข้อความ
    expected:      str   # สิ่งที่ควรจะเป็น
    context:       str   # ข้อความรอบข้าง


@dataclass
class CheckResult:
    file_type:    str
    filename:     str
    timestamp:    str
    raw_text:     str
    wrong_words:  list   # [{"word":str, "suggestion":str, "conf_type":"spell_error"|"ocr_unclear", "ocr_conf":float}]
    brand_issues: list   # [BrandIssue]
    frame:        object = None
    ocr_details:  list   = field(default_factory=list)  # [(text, conf), ...]


# ─────────────────────────────────────────────
# JSON I/O Functions
# ─────────────────────────────────────────────

def load_brands() -> list:
    """โหลดข้อมูลแบรนด์จาก JSON สร้างไฟล์ default ถ้ายังไม่มี
    Migration: เติม fields ใหม่ให้กับ brands เก่าที่ยังไม่มี
    (phone, benefits, target, ingredients, ayor, prices, innovation, awards, research)
    """
    _BRAND_DEFAULTS = {
        "betaoil":        {
            "phone": "061-826-3693",
            "benefits": "ดูแลหัวใจและหลอดเลือด ลดไขมันในเลือด ลดคอเลสเตอรอล ลดความดันโลหิต",
            "target": "ผู้มีไขมันในเลือดสูง ความดันสูง เวียนหัว มือชาเท้าชา เหนื่อยง่าย",
            "ingredients": "น้ำมันรำข้าว น้ำมันมะกอก คาโนล่าออยล์ เปปไทด์จากข้าวสีนิล",
            "ayor": "13-1-01566-5-0001",
            "prices": {"1":690,"4":1790,"8":3290,"15":5690},
            "prices_original": {"1":1290,"4":5160,"8":10320,"15":19350},
            "innovation": "เทคโนโลยีเปปไทด์จากข้าวสีนิล (Rice Peptide Technology) จากประเทศสวิตเซอร์แลนด์ รับรองโดย วว.",
            "awards": ["รางวัลผลิตภัณฑ์ยอดเยี่ยม เจนีวา สวิตเซอร์แลนด์","เหรียญเงิน นวัตกรรมเพื่อสุขภาพจากธรรมชาติ เจนีวา","เหรียญทอง นวัตกรรมเพื่อสุขภาพ สหรัฐอเมริกา"],
            "research": "รับรองโดยสถาบันวิจัยวิทยาศาสตร์และเทคโนโลยีแห่งประเทศไทย (วว.)"},
        "betalife":       {
            "phone": "063-479-1496",
            "benefits": "บำรุงไต ปกป้องเซลล์ไต ขับสารพิษ ฟื้นฟูสุขภาพไต",
            "target": "ผู้ที่มีความเสี่ยงโรคไต ทานยาเยอะ ทานเค็ม มีเบาหวาน/ความดัน",
            "ingredients": "เห็ดหลินจือแดง ALA",
            "ayor": "",
            "prices": {"1":690,"4":1790,"8":3290,"15":5690},
            "prices_original": {"1":1290,"4":5160,"8":10320,"15":19350},
            "innovation": "สารออกฤทธิ์ทุติยภูมิจากเห็ดหลินจือแดง สูตรฟอสฟอรัสต่ำ ร่วมพัฒนากับ วว.",
            "awards": ["เหรียญทอง The 50th International Exhibition of Inventions Geneva"],
            "research": "ผลงานวิจัย: สารออกฤทธิ์ทุติยภูมิจากเห็ดหลินจือแดงสำหรับบรรเทาอาการของโรคไต"},
        "betaherb":       {
            "phone": "063-479-1496",
            "benefits": "ควบคุมระดับน้ำตาลในเลือด ส่งเสริมการทำงานของอินซูลิน ลดความเสี่ยงเบาหวาน",
            "target": "ผู้ที่น้ำตาลในเลือดสูง เสี่ยงเบาหวาน ความดันสูง",
            "ingredients": "มะระขี้นก ไคโตซาน สารสกัดจากรังผึ้ง เจียวกู่หลาน ยีสต์เบต้ากลูแคน",
            "ayor": "",
            "prices": {"1":690,"4":1790,"8":3290,"15":5690},
            "prices_original": {"1":1590,"4":6360,"8":12720,"15":23850},
            "innovation": "สารสกัดจากมะระขี้นก 9 ชนิด ส่งเสริมการทำงานของอินซูลิน สารสกัดเจียวกู่หลาน ยีสต์เบต้ากลูแคน",
            "awards": [],
            "research": "สารสกัดจากมะระขี้นก 9 ชนิด ผ่านกระบวนการสกัดมาตรฐาน"},
        "betaxplus":      {
            "phone": "061-826-3693",
            "benefits": "บำรุงปอด เสริมภูมิคุ้มกัน ดูแลระบบทางเดินหายใจ",
            "target": "ผู้มีปัญหาภูมิแพ้ หอบหืด ไอเรื้อรัง หายใจไม่สะดวก เจอฝุ่นควัน",
            "ingredients": "ยีสต์เบต้ากลูแคน M-Gard สายพันธุ์ 1,3/1,6 กลูแคน",
            "ayor": "",
            "prices": {"1":690,"4":1790,"8":3290,"15":5690},
            "prices_original": {"1":1290,"4":5160,"8":10320,"15":19350},
            "innovation": "นวัตกรรม M-Gard® ยีสต์เบต้ากลูแคน สายพันธุ์ 1,3/1,6 กลูแคน นำเข้าจากนอร์เวย์",
            "awards": [],
            "research": "M-Gard® นำเข้าจากประเทศนอร์เวย์ มาตรฐานระดับสากล"},
        "betalivproplus": {
            "phone": "063-479-1496",
            "benefits": "บำรุงตับ ล้างสารพิษตับ ลดไขมันพอกตับ ฟื้นฟูตับ",
            "target": "ผู้ที่ดื่มแอลกอฮอล์ ทานของมัน ค่าตับสูง ไขมันพอกตับ ตับอักเสบ",
            "ingredients": "LIVOX™ LIVOTOX® โกจิเบอร์รี่ เบต้าแคโรทีน ส้มขม ลูกพลับ ถั่วเขียว Buckwheat ALA Natto Artichoke โคลีน",
            "ayor": "",
            "prices": {"1":690,"4":1790,"8":3290,"15":5690},
            "prices_original": {"1":1890,"4":7560,"8":15120,"15":28350},
            "innovation": "นวัตกรรม LIVOX™ + LIVOTOX® ร่วมพัฒนากับ Toyo Hakko ประเทศญี่ปุ่น",
            "awards": ["เหรียญทอง KIPA Seoul International Invention Fair 2009/2011/2012","เหรียญทอง + รางวัลพิเศษจากโรมาเนีย (รวม 7 รางวัล)"],
            "research": "ร่วมพัฒนากับ Toyo Hakko ประเทศญี่ปุ่น สารสกัด 9 ชนิด"},
        "labfarm":        {
            "phone": "063-479-1496",
            "benefits": "เสริมภูมิคุ้มกัน กระตุ้น NK Cells ต้านอนุมูลอิสระ สำหรับผู้ป่วยโรคร้าย/มะเร็ง",
            "target": "ผู้ป่วยมะเร็ง ผู้ทำคีโม ผู้ต้องการเสริมภูมิคุ้มกัน",
            "ingredients": "Immuneo Plus™ เบต้ากลูแคน M-Gard® ซีบัคธอร์น Maqui Berry Quercetin Vitamin D3 Ginger Aegle Black Pepper Ginkgo Ginseng Vitamin B complex Gamma Oryzanol",
            "ayor": "73-1-02663-5-0191",
            "prices": {"1":890,"4":2490,"8":4590,"15":7890},
            "prices_original": {"1":990,"4":3690,"8":7920,"15":14850},
            "innovation": "นวัตกรรม Immuneo Plus™ + M-Gard® Golden Ratio 4:1 ทดสอบโดย มหาวิทยาลัยรังสิต",
            "awards": ["ทดสอบโดยมหาวิทยาลัยรังสิต ยืนยันประสิทธิภาพยับยั้งเซลล์มะเร็ง"],
            "research": "ทดสอบโดยกลุ่มวิจัยเภสัชวิทยาและพิษวิทยา มหาวิทยาลัยรังสิต"},
        "betacalproplus": {
            "phone": "",
            "benefits": "นวัตกรรม CalFusion™ เสริมแคลเซียมและบำรุงกระดูก",
            "target": "ผู้ต้องการเสริมแคลเซียม บำรุงกระดูก",
            "ingredients": "CalFusion™",
            "ayor": "",
            "prices": {},
            "prices_original": {},
            "innovation": "นวัตกรรม CalFusion™",
            "awards": ["รางวัลนวัตกรรมนานาชาติ Silicon Valley International Inventions Festival 2025"],
            "research": "Silicon Valley International Inventions Festival 2025"},
    }
    if not BRANDS_FILE.exists():
        BRANDS_FILE.write_text(
            json.dumps({"brands": list({**{"id":bid,"english":bid,"thai":d["ingredients"][:4],"unit":"แคปซูล"},**d}
                                       for bid, d in _BRAND_DEFAULTS.items())},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    data   = json.loads(BRANDS_FILE.read_text(encoding="utf-8"))
    brands = data.get("brands", [])

    # Migration: เติม fields ใหม่ที่ยังขาดอยู่
    changed = False
    for brand in brands:
        bid = brand.get("id", "")
        if bid in _BRAND_DEFAULTS:
            for key, val in _BRAND_DEFAULTS[bid].items():
                if key not in brand:
                    brand[key] = val
                    changed = True
    if changed:
        save_brands(brands)
    return brands


def save_brands(brands: list) -> None:
    """บันทึกรายการแบรนด์ลงไฟล์ JSON"""
    BRANDS_FILE.write_text(
        json.dumps({"brands": brands}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_whitelist() -> set:
    """โหลด whitelist คำที่ไม่ต้องตรวจสอบ
    Seed จาก DEFAULT_WHITELIST ถ้าไฟล์ยังไม่มี
    คืนค่า DEFAULT_WHITELIST รวมกับคำที่ user เพิ่มเอง
    """
    if not WHITELIST_FILE.exists():
        save_whitelist(DEFAULT_WHITELIST)
        return set(DEFAULT_WHITELIST)
    data  = json.loads(WHITELIST_FILE.read_text(encoding="utf-8"))
    saved = set(data.get("words", []))
    return DEFAULT_WHITELIST | saved


def save_whitelist(words: set) -> None:
    """บันทึก whitelist ลงไฟล์ JSON (บันทึกเฉพาะคำที่ user เพิ่มเอง ไม่บันทึก defaults)"""
    user_words = words - DEFAULT_WHITELIST
    WHITELIST_FILE.write_text(
        json.dumps({"words": sorted(user_words)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_phones() -> dict:
    """โหลดกฎเบอร์โทรจาก JSON สร้างไฟล์ default ถ้ายังไม่มี"""
    _default = {
        "description": "กฎการตรวจสอบเบอร์โทรศัพท์ตามสินค้า — บริษัท โฮปฟูล จำกัด",
        "known_phones": ["063-479-1496", "061-826-3693"],
        "rules": [
            {
                "phone": "061-826-3693",
                "product_ids": ["betaoil", "betaxplus"],
                "description": "เบอร์สำหรับ Beta Oil และ Beta X Plus",
            },
            {
                "phone": "063-479-1496",
                "product_ids": ["betalife", "betaherb", "betalivproplus", "labfarm"],
                "description": "เบอร์สำหรับ Beta Life, Beta Herb, Beta Liv Pro Plus, Lab Farm",
            },
        ],
    }
    if not PHONES_FILE.exists():
        PHONES_FILE.write_text(json.dumps(_default, ensure_ascii=False, indent=2), encoding="utf-8")
        return _default
    return json.loads(PHONES_FILE.read_text(encoding="utf-8"))


def save_phones(data: dict) -> None:
    """บันทึกกฎเบอร์โทรลงไฟล์ JSON"""
    PHONES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ocr_corrections() -> dict:
    """
    โหลดตาราง OCR อ่านผิด → คำที่ถูก จาก JSON
    ถ้าไม่มีไฟล์ สร้างจาก DEFAULT_OCR_CORRECTIONS
    """
    if not CORRECTIONS_FILE.exists():
        save_ocr_corrections(DEFAULT_OCR_CORRECTIONS)
        return dict(DEFAULT_OCR_CORRECTIONS)
    data = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
    return data.get("corrections", {})


def save_ocr_corrections(corrections: dict) -> None:
    """บันทึกตาราง corrections ลงไฟล์ JSON"""
    CORRECTIONS_FILE.write_text(
        json.dumps({"corrections": corrections}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_phrases() -> dict:
    """โหลดประโยคต้นแบบจาก JSON สร้างไฟล์ default ถ้ายังไม่มี"""
    _default = {
        "description": "ประโยคต้นแบบที่ใช้บ่อยในวิดีโอสินค้า Hopeful",
        "threshold": 0.7,
        "phrases": [
            "คุมน้ำตาลให้ดี", "คนที่เป็นเบาหวาน ต้องระวังไต",
            "ปรึกษาผู้เชี่ยวชาญสายตรง", "ผากเป็นพิเศษ",
            "มีผู้เชี่ยวชาญที่คอยดูแล", "ลดเต็มอย่างเดียวไม่พอ",
            "กินเค็มทุกวัน ไตทำงานหนักทุกวัน", "เสริมด้วยเบต้าไลฟ์",
            "ดูแลหัวใจและหลอดเลือด", "บำรุงไต ปกป้องเซลล์ไต",
            "ควบคุมน้ำตาลในเลือด", "ผลิตภัณฑ์เสริมอาหาร",
        ],
    }
    if not PHRASES_FILE.exists():
        PHRASES_FILE.write_text(json.dumps(_default, ensure_ascii=False, indent=2), encoding="utf-8")
        return _default
    return json.loads(PHRASES_FILE.read_text(encoding="utf-8"))


def save_phrases(data: dict) -> None:
    """บันทึกประโยคต้นแบบลงไฟล์ JSON"""
    PHRASES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────
# OCR Debug Log (module-level accumulator)
# รีเซ็ตทุกครั้งที่เริ่มสแกนใหม่ ไม่ใช้ thread-safety เพราะ Streamlit single-thread
# ─────────────────────────────────────────────

_OCR_LOG: list = []


def ocr_log_clear() -> None:
    global _OCR_LOG
    _OCR_LOG = []


def ocr_log_add(entry: dict) -> None:
    """เพิ่ม log entry และ print ไปที่ terminal ด้วย (เพื่อ debug)"""
    global _OCR_LOG
    _OCR_LOG.append(entry)
    t = entry.get("type", "?")
    orig = entry.get("original", "")
    corr = entry.get("corrected", "")
    ratio = entry.get("ratio", 1.0)
    if t == "char":
        print(f"[OCR-CHAR]   '{orig}' → '{corr}'", flush=True)
    elif t == "phrase":
        print(f"[OCR-PHRASE] '{orig}' → '{corr}'  (ratio={ratio:.2f})", flush=True)


def apply_ocr_corrections(text: str, corrections: dict) -> str:
    """
    แทนที่คำที่ OCR มักอ่านผิดด้วยคำที่ถูกต้อง (character/word level)
    Log ทุก correction ที่เกิดขึ้นลง _OCR_LOG และ terminal
    """
    for wrong, right in corrections.items():
        if wrong in text:
            new_text = text.replace(wrong, right)
            ocr_log_add({"type": "char", "original": wrong, "corrected": right,
                          "context": text, "ratio": 1.0})
            text = new_text
    return text


def apply_phrase_corrections(text: str, phrases: list, threshold: float = 0.7) -> str:
    """
    เทียบข้อความแต่ละ OCR box กับประโยคต้นแบบโดยใช้ SequenceMatcher
    ถ้า ratio >= threshold → แทนที่ด้วยประโยคต้นแบบ (auto-correct)
    Log ผลลัพธ์ลง _OCR_LOG และ terminal เสมอ
    """
    if not phrases or not text.strip():
        return text

    best_ratio   = 0.0
    best_phrase  = None
    text_stripped = text.strip()

    for phrase in phrases:
        ratio = SequenceMatcher(None, text_stripped, phrase).ratio()
        if ratio > best_ratio:
            best_ratio  = ratio
            best_phrase = phrase

    if best_phrase and best_ratio >= threshold and best_ratio < 1.0:
        ocr_log_add({
            "type":      "phrase",
            "original":  text_stripped,
            "corrected": best_phrase,
            "ratio":     best_ratio,
        })
        return best_phrase

    # Log ว่า phrase matching ลอง match แต่ไม่ถึง threshold (สำหรับ debug)
    if best_phrase and best_ratio >= 0.4:
        print(f"[OCR-PHRASE] no match: '{text_stripped[:40]}' ~ '{best_phrase}' ratio={best_ratio:.2f} (< {threshold})", flush=True)

    return text


# ─────────────────────────────────────────────
# Cached Model Loading
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_ocr(use_gpu: bool = False):
    """ตรวจสอบ pytesseract พร้อมใช้งาน (use_gpu ไม่มีผล — tesseract ใช้ CPU เสมอ)"""
    pytesseract.get_tesseract_version()
    return None


@st.cache_resource(show_spinner=False)
def load_dictionary() -> set:
    """โหลดพจนานุกรมภาษาไทย (cached)"""
    return set(thai_words())


# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────

def preprocess_image(
    img: np.ndarray,
    upscale_factor: float = 2.0,
    do_denoise: bool = True,
    do_binarize: bool = False,
) -> np.ndarray:
    """
    เพิ่มคุณภาพภาพก่อนส่ง OCR เพื่อให้อ่านตัวอักษรไทยแม่นยำขึ้น:

    1. Upscale — ขยายภาพด้วย INTER_CUBIC ช่วยให้ OCR อ่านตัวเล็กได้ดีขึ้น
    2. CLAHE   — เพิ่ม contrast แบบ adaptive (ไม่ washout บริเวณสว่างมาก)
    3. Bilateral filter — กรอง noise แต่รักษาขอบตัวอักษรไว้
    4. Adaptive threshold (optional) — ทำขาว-ดำ เหมาะกับป้ายหรือข้อความบนพื้นเรียบ
    """
    # 1. Upscale
    if upscale_factor > 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(
            img,
            (int(w * upscale_factor), int(h * upscale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )

    # แปลงเป็น Grayscale เพื่อ contrast/noise operations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE — เพิ่ม contrast แบบ adaptive
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Bilateral filter — กรอง noise รักษาขอบตัวอักษร
    if do_denoise:
        enhanced = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Adaptive binarization (optional) — ดีสำหรับข้อความบนพื้นสีเดียว
    if do_binarize:
        enhanced = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11, C=2,
        )

    # แปลงกลับเป็น BGR (EasyOCR รับได้ทั้ง gray และ BGR)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────
# OCR Function
# ─────────────────────────────────────────────

def ocr_image(
    reader,
    img_array: np.ndarray,
    min_confidence: float = 0.5,
    corrections: dict = None,
    phrases: list = None,
    phrase_threshold: float = 0.7,
    preprocess: bool = True,
    upscale_factor: float = 2.0,
    do_denoise: bool = True,
    do_binarize: bool = False,
) -> tuple:
    """
    ทำ OCR บนรูปภาพ คืนค่า (full_text, ocr_details)

    Pipeline:
    1. preprocess_image() ถ้า preprocess=True
    2. EasyOCR readtext (detail=1, paragraph=False)
    3. กรอง: conf >= min_confidence AND len >= 2
    4. apply_ocr_corrections()  — char-level auto-correct + log
    5. apply_phrase_corrections() — phrase fuzzy-match + log ถ้า ratio >= phrase_threshold
    """
    if corrections is None:
        corrections = {}
    if phrases is None:
        phrases = []

    # 1. Preprocess
    if preprocess:
        img_array = preprocess_image(img_array, upscale_factor, do_denoise, do_binarize)

    # 2. OCR ด้วย pytesseract
    pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(
        pil_img, lang="tha+eng",
        output_type=pytesseract.Output.DICT,
        config="--psm 3 --oem 1",
    )

    texts   = []
    details = []
    for text, conf in zip(data["text"], data["conf"]):
        text     = text.strip()
        conf_int = int(conf)
        if conf_int < 0 or not text:
            continue
        conf_norm = conf_int / 100.0
        if conf_norm < min_confidence or len(text) < 2:
            continue

        # 3. Char-level corrections
        after_char = apply_ocr_corrections(text, corrections)

        # 4. Phrase-level fuzzy corrections
        after_phrase = apply_phrase_corrections(after_char, phrases, phrase_threshold)

        texts.append(after_phrase)
        details.append((after_phrase, conf_norm))

    full_text = " ".join(texts)
    return full_text, details


# ─────────────────────────────────────────────
# Phone Number Helpers
# ─────────────────────────────────────────────

def normalize_phone(phone: str) -> str:
    """ลบเครื่องหมายออกจากเบอร์โทร เหลือแต่ตัวเลข: '063-479-1496' → '0634791496'"""
    return re.sub(r"[^0-9]", "", phone)


def extract_phones(text: str) -> list:
    """
    ดึงเบอร์โทรจากข้อความ รองรับหลายรูปแบบ:
    063-479-1496 / 0634791496 / 063 479 1496 / 063.479.1496
    """
    pattern = r"\b0\d{1,2}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    return re.findall(pattern, text)


def check_phone_issues(full_text: str, brands: list) -> list:
    """
    ตรวจว่าเบอร์โทรในเฟรมตรงกับสินค้าที่พบหรือไม่

    Algorithm:
    1. ดึงเบอร์โทรทั้งหมดจาก full_text
    2. ตรวจว่ามีชื่อแบรนด์ใดอยู่ในข้อความบ้าง
    3. สำหรับแบรนด์ที่พบ → ตรวจว่าเบอร์ที่พบตรงกับเบอร์ของแบรนด์หรือไม่
    4. ถ้าพบเบอร์ของแบรนด์อื่น (ไม่ใช่เบอร์ตัวเอง) → BrandIssue(issue_type="wrong_phone")
    """
    found_phones_raw = extract_phones(full_text)
    if not found_phones_raw:
        return []

    # map normalized → raw string
    found_phones: dict[str, str] = {normalize_phone(p): p for p in found_phones_raw}

    # รวบรวมเบอร์ที่รู้จักทั้งหมด (เพื่อแยกเบอร์ที่เป็นของแบรนด์อื่น)
    all_brand_phones: dict[str, str] = {}  # normalized → formatted
    for brand in brands:
        ph = brand.get("phone", "")
        if ph:
            all_brand_phones[normalize_phone(ph)] = ph

    # ตรวจว่าแบรนด์ใดปรากฏในข้อความ (ชื่อ EN หรือชื่อไทย)
    mentioned: list[dict] = []
    text_lower = full_text.lower()
    for brand in brands:
        if brand["english"].lower() in text_lower or brand["thai"] in full_text:
            mentioned.append(brand)

    if not mentioned:
        return []

    issues = []
    seen: set = set()

    for brand in mentioned:
        expected_norm = normalize_phone(brand.get("phone", ""))
        if not expected_norm:
            continue

        for norm, raw in found_phones.items():
            # ถ้าเบอร์ที่พบ ≠ เบอร์ของแบรนด์ AND เบอร์นั้นเป็นเบอร์ของแบรนด์อื่น
            if norm != expected_norm and norm in all_brand_phones:
                key = (brand["id"], norm)
                if key not in seen:
                    seen.add(key)
                    issues.append(BrandIssue(
                        issue_type    = "wrong_phone",
                        brand_id      = brand["id"],
                        brand_english = brand["english"],
                        brand_thai    = brand["thai"],
                        found         = raw,
                        expected      = brand.get("phone", ""),
                        context       = "",
                    ))

    return issues


# ─────────────────────────────────────────────
# Brand Issue Detection
# ─────────────────────────────────────────────

def check_brand_issues(text: str, brands: list) -> list:
    """
    ตรวจหาปัญหาเกี่ยวกับแบรนด์ในข้อความ:
    1. พบชื่อแบรนด์ภาษาอังกฤษ → ตรวจ Thai name และหน่วยในบริเวณใกล้เคียง (±5 tokens)
    2. พบคำภาษาไทย ≥ 3 ตัว → ตรวจ fuzzy match กับชื่อแบรนด์ทุกรายการ
    """
    tokens = re.findall(r'[a-zA-Z]+|[\u0E00-\u0E7F]+|\d+', text)
    issues = []
    seen   = set()  # (type, found, expected)

    for i, token in enumerate(tokens):

        # กรณี 1: ชื่อแบรนด์ภาษาอังกฤษ
        for brand in brands:
            if token.lower() == brand["english"].lower():
                win_start = max(0, i - 5)
                win_end   = min(len(tokens), i + 6)
                window    = tokens[win_start:win_end]

                # ตรวจชื่อไทย fuzzy
                thai_in_win = [t for t in window if re.fullmatch(r'[\u0E00-\u0E7F]+', t)]
                for thai_tok in thai_in_win:
                    ratio = SequenceMatcher(None, thai_tok, brand["thai"]).ratio()
                    if 0.6 <= ratio < 1.0:
                        key = ("brand_thai_misspell", thai_tok, brand["thai"])
                        if key not in seen:
                            seen.add(key)
                            ctx = " ".join(tokens[max(0, i-3):min(len(tokens), i+4)])
                            issues.append(BrandIssue(
                                issue_type    = "brand_thai_misspell",
                                brand_id      = brand["id"],
                                brand_english = brand["english"],
                                brand_thai    = brand["thai"],
                                found         = thai_tok,
                                expected      = brand["thai"],
                                context       = ctx,
                            ))

                # ตรวจหน่วย
                unit_in_win = [t for t in window if t in UNIT_WORDS]
                for unit_tok in unit_in_win:
                    if unit_tok != brand["unit"]:
                        key = ("wrong_unit", unit_tok, brand["unit"])
                        if key not in seen:
                            seen.add(key)
                            ctx = " ".join(tokens[max(0, i-3):min(len(tokens), i+4)])
                            issues.append(BrandIssue(
                                issue_type    = "wrong_unit",
                                brand_id      = brand["id"],
                                brand_english = brand["english"],
                                brand_thai    = brand["thai"],
                                found         = unit_tok,
                                expected      = brand["unit"],
                                context       = ctx,
                            ))

        # กรณี 2: คำภาษาไทย ≥ 3 ตัว — fuzzy กับชื่อแบรนด์ทุกรายการ
        if re.fullmatch(r'[\u0E00-\u0E7F]{3,}', token):
            for brand in brands:
                if token == brand["thai"]:
                    continue  # exact match ถูกต้องแล้ว ข้ามไป
                ratio = SequenceMatcher(None, token, brand["thai"]).ratio()
                if 0.65 <= ratio < 1.0:
                    key = ("brand_thai_misspell", token, brand["thai"])
                    if key not in seen:
                        seen.add(key)
                        ctx = " ".join(tokens[max(0, i-3):min(len(tokens), i+4)])
                        issues.append(BrandIssue(
                            issue_type    = "brand_thai_misspell",
                            brand_id      = brand["id"],
                            brand_english = brand["english"],
                            brand_thai    = brand["thai"],
                            found         = token,
                            expected      = brand["thai"],
                            context       = ctx,
                        ))

    return issues


# ─────────────────────────────────────────────
# Promotion Requirements Check
# ─────────────────────────────────────────────

def check_promotion_requirements(text: str, brands: list) -> list:
    """
    ตรวจสอบข้อมูลโปรสงกรานต์ 5 รายการ:
    1. duration - ระยะเวลาโปรโมชั่น
    2. price - ราคา
    3. benefits_gifts - สิทธิ/ของแถม
    4. promotion_type - ประเภทโปรโมชั่น
    5. bundle_exclusivity - เงื่อนไขสินค้าแถม
    """
    issues = []
    text_lower = text.lower()
    
    # Keywords สำหรับแต่ละหมวดหมู่
    duration_keywords = ["ระยะเวลา", "โปรโมชั่น", "โปร", "วันที่", "ส่วนลด", "ราคาพิเศษ", "เมื่อ", "ถึง", "วันที่"]
    price_keywords = ["ราคา", "บาท", "฿", "ลด", "เหลือ", "ปกติ", "บาท/ซอง", "บาท/แคปซูล", "บาท/เม็ด"]
    benefits_keywords = ["ของแถม", "แถม", "ฟรี", "เพิ่ม", "เพิ่มเติม", "ยิ่งซื้อ", "ซื้อมากขึ้น", "ได้รับ", "สิทธิ"]
    promotion_type_keywords = ["สงกรานต์", "โปรโมชั่น", "โปร", "ปีใหม่", "เทศกาล", "ครบรอบ", "ลดราคา"]
    exclusivity_keywords = ["ไม่ซ้ำ", "สินค้าต่างชนิด", "ต่างสินค้า", "ห้ามซ้ำ", "อื่นๆ", "แตกต่าง", "นอกจากสินค้าซื้อ"]
    
    # Mention brands ที่พบ
    mentioned_brands = [b for b in brands if b["english"].lower() in text_lower or b["thai"] in text]
    
    if not mentioned_brands:
        return []
    
    # ตรวจสอบแต่ละ requirement สำหรับแต่ละ brand
    requirements = ["duration", "price", "benefits_gifts", "promotion_type", "bundle_exclusivity"]
    keyword_map = {
        "duration": duration_keywords,
        "price": price_keywords,
        "benefits_gifts": benefits_keywords,
        "promotion_type": promotion_type_keywords,
        "bundle_exclusivity": exclusivity_keywords,
    }
    
    for brand in mentioned_brands:
        for req in requirements:
            keywords = keyword_map[req]
            found = any(kw in text_lower for kw in keywords)
            
            if not found and brand.get("promotion_requirements", {}).get(req) == "required":
                issues.append(BrandIssue(
                    issue_type    = f"missing_promotion_{req}",
                    brand_id      = brand["id"],
                    brand_english = brand["english"],
                    brand_thai    = brand["thai"],
                    found         = "",
                    expected      = f"ต้องระบุ {req.replace('_', ' ')}",
                    context       = f"⚠️ ข้อมูลโปรสงกรานต์ {brand['thai']} ขาดหายไป",
                ))
    
    return issues


# ─────────────────────────────────────────────
# Spell Check Helper
# ─────────────────────────────────────────────

def check_spelling(text: str, thai_dict: set, ignore_words: set, min_token_len: int) -> list:
    """
    tokenize ข้อความ → กรองเฉพาะคำภาษาไทย → ตรวจพจนานุกรม
    คืนค่า [{"word": str, "suggestion": str}]
    """
    results = []
    tokens  = word_tokenize(text, engine="newmm")
    for tok in tokens:
        tok = tok.strip()
        if (
            re.fullmatch(r'[\u0E00-\u0E7F]+', tok)
            and len(tok) >= min_token_len
            and tok not in ignore_words
            and tok not in thai_dict
        ):
            suggestion = correct(tok)
            if suggestion and suggestion != tok:
                results.append({"word": tok, "suggestion": suggestion})
            else:
                results.append({"word": tok, "suggestion": tok})
    return results


# ─────────────────────────────────────────────
# Process OCR for Spelling (main pipeline)
# ─────────────────────────────────────────────

def process_ocr_for_spelling(
    ocr_details: list,
    thai_dict: set,
    ignore_words: set,
    brands: list,
    min_token_len: int,
    spell_conf_threshold: float = 0.7,
    ocr_misread_threshold: float = 0.85,
) -> tuple:
    """
    จำแนกผล OCR เป็น 3 ระดับตาม confidence:

    conf < spell_conf_threshold               → "ocr_unclear"  (OCR ไม่ชัด — ไม่ตรวจสะกด)
    spell_conf_threshold <= conf < misread_th → "ocr_misread"  (OCR อาจอ่านผิด — ตรวจสะกดแต่แจ้งเตือนแยก)
    conf >= ocr_misread_threshold             → "spell_error"  (OCR อ่านชัด — คำผิดจริง)

    ชื่อแบรนด์ไทยถูกเพิ่มใน ignore_words อัตโนมัติเพื่อป้องกัน false positive
    คืนค่า: (full_text, wrong_words_list, brand_issues_list)
    """
    effective_ignore = set(ignore_words)
    for brand in brands:
        effective_ignore.add(brand["thai"])

    all_texts   = []
    wrong_words = []
    seen_words  = set()

    for (text, conf) in ocr_details:
        all_texts.append(text)

        if conf >= spell_conf_threshold:
            # ตรวจสะกด — จำแนก conf_type ตาม threshold
            conf_type = "spell_error" if conf >= ocr_misread_threshold else "ocr_misread"
            errors    = check_spelling(text, thai_dict, effective_ignore, min_token_len)
            for err in errors:
                w = err["word"]
                if w not in seen_words:
                    seen_words.add(w)
                    wrong_words.append({
                        "word":       w,
                        "suggestion": err["suggestion"],
                        "conf_type":  conf_type,
                        "ocr_conf":   conf,
                    })
        else:
            # confidence ต่ำมาก → mark เฉพาะ token ไทย ว่า "ocr_unclear"
            for tok in re.findall(r'[\u0E00-\u0E7F]+', text):
                tok = tok.strip()
                if len(tok) >= min_token_len and tok not in effective_ignore and tok not in seen_words:
                    seen_words.add(tok)
                    wrong_words.append({
                        "word":       tok,
                        "suggestion": tok,
                        "conf_type":  "ocr_unclear",
                        "ocr_conf":   conf,
                    })

    full_text    = " ".join(all_texts)
    brand_issues = (
        check_brand_issues(full_text, brands) +
        check_phone_issues(full_text, brands) +
        check_promotion_requirements(full_text, brands)
    )
    return full_text, wrong_words, brand_issues


# ─────────────────────────────────────────────
# Text Highlighting
# ─────────────────────────────────────────────

def highlight_text(
    raw_text: str,
    wrong_words: list,
    dismissed: set = None,
    whitelist: set = None,
) -> str:
    """
    เน้นสีคำผิดในข้อความ HTML:
    - spell_error + ไม่ dismissed/whitelist → class="wrong-word"
    - spell_error + dismissed/whitelist     → class="dismissed-word"
    - ocr_unclear                           → class="ocr-unclear"
    """
    dismissed = dismissed or set()
    whitelist  = whitelist or set()

    # สร้าง map: คำ → CSS class
    word_class = {}
    for item in wrong_words:
        w  = item["word"]
        ct = item["conf_type"]
        if ct == "spell_error":
            word_class[w] = "dismissed-word" if (w in dismissed or w in whitelist) else "wrong-word"
        elif ct == "ocr_misread":
            # OCR อาจอ่านผิด — แสดงสีส้ม (ถ้ายังไม่ถูก dismiss)
            if w not in word_class:
                word_class[w] = "dismissed-word" if (w in dismissed or w in whitelist) else "ocr-misread"
        elif ct == "ocr_unclear":
            if w not in word_class:
                word_class[w] = "ocr-unclear"

    if not word_class:
        return html.escape(raw_text)

    escaped = html.escape(raw_text)
    # แทนจากคำยาวไปสั้นเพื่อกัน substring ซ้อน
    for w in sorted(word_class.keys(), key=len, reverse=True):
        cls       = word_class[w]
        escaped_w = html.escape(w)
        escaped   = escaped.replace(escaped_w, f'<span class="{cls}">{escaped_w}</span>')

    return escaped


# ─────────────────────────────────────────────
# HTML Report Generator
# ─────────────────────────────────────────────

def generate_html_report(all_results: list, dismissed: set = None, whitelist: set = None) -> str:
    """สร้าง HTML report ธีมมืด พร้อมส่วนปัญหาแบรนด์"""
    dismissed = dismissed or set()
    whitelist  = whitelist or set()

    rows = []
    for r in all_results:
        highlighted   = highlight_text(r.raw_text, r.wrong_words, dismissed, whitelist)
        spell_errors  = [w for w in r.wrong_words
                         if w["conf_type"] == "spell_error"
                         and w["word"] not in dismissed
                         and w["word"] not in whitelist]
        ocr_unclears  = [w for w in r.wrong_words if w["conf_type"] == "ocr_unclear"]

        brand_rows = ""
        for bi in r.brand_issues:
            if bi.issue_type == "brand_thai_misspell":
                badge = '<span class="badge-brand-warn">สะกดชื่อแบรนด์ผิด</span>'
            elif bi.issue_type == "wrong_unit":
                badge = '<span class="badge-unit-error">หน่วยผิด</span>'
            else:
                badge = '<span class="badge-phone-error">⚠️ เบอร์โทรผิด</span>'
            brand_rows += f"""
            <tr>
              <td>{badge}</td>
              <td>{html.escape(bi.brand_english)} / {html.escape(bi.brand_thai)}</td>
              <td>{html.escape(bi.found)}</td>
              <td>{html.escape(bi.expected)}</td>
            </tr>"""

        brand_section = ""
        if brand_rows:
            brand_section = f"""
            <h4 style="color:#fbbf24;margin-top:16px;">🏷️ ปัญหาแบรนด์</h4>
            <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
              <thead>
                <tr style="color:#94a3b8;">
                  <th style="text-align:left;padding:4px 8px;">ประเภท</th>
                  <th style="text-align:left;padding:4px 8px;">แบรนด์</th>
                  <th style="text-align:left;padding:4px 8px;">พบ</th>
                  <th style="text-align:left;padding:4px 8px;">ที่ถูกต้อง</th>
                </tr>
              </thead>
              <tbody>{brand_rows}</tbody>
            </table>"""

        rows.append(f"""
        <tr>
          <td style="padding:8px;border-bottom:1px solid #334155;">{html.escape(r.filename)}</td>
          <td style="padding:8px;border-bottom:1px solid #334155;">{html.escape(r.timestamp)}</td>
          <td style="padding:8px;border-bottom:1px solid #334155;">{highlighted}</td>
          <td style="padding:8px;border-bottom:1px solid #334155;text-align:center;">{len(spell_errors)}</td>
          <td style="padding:8px;border-bottom:1px solid #334155;text-align:center;">{len(ocr_unclears)}</td>
          <td style="padding:8px;border-bottom:1px solid #334155;text-align:center;">{len(r.brand_issues)}</td>
        </tr>
        <tr>
          <td colspan="6" style="padding:4px 8px 16px 8px;border-bottom:2px solid #475569;">
            {brand_section}
          </td>
        </tr>""")

    all_rows_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>Thai Spell Checker Report</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:'Sarabun',sans-serif; padding:24px; }}
  h1   {{ background:linear-gradient(135deg,#6366f1,#8b5cf6);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  table {{ width:100%; border-collapse:collapse; }}
  th    {{ background:#1e293b; color:#94a3b8; padding:10px 8px; text-align:left; }}
  .wrong-word     {{ color:#ff4d4d; font-weight:800; text-decoration:underline wavy #ff4d4d; }}
  .ocr-unclear    {{ color:#94a3b8; border-bottom:1px dashed #64748b; }}
  .dismissed-word {{ color:#475569; text-decoration:line-through; }}
  .badge-spell-error,.badge-ocr-unclear,.badge-brand-warn,.badge-unit-error {{
    display:inline-block; border-radius:999px; padding:2px 10px;
    font-size:.82rem; font-weight:700; margin:2px;
  }}
  .badge-spell-error {{ background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.3); }}
  .badge-ocr-unclear {{ background:rgba(100,116,139,.12); color:#94a3b8; border:1px solid rgba(100,116,139,.25); }}
  .badge-brand-warn  {{ background:rgba(251,191,36,.15); color:#fbbf24; border:1px solid rgba(251,191,36,.3); }}
  .badge-unit-error  {{ background:rgba(251,146,60,.15); color:#fb923c; border:1px solid rgba(251,146,60,.3); }}
</style>
</head>
<body>
<h1>🔍 Thai Spell Checker Report</h1>
<p style="color:#64748b;">สร้างเมื่อ {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<table>
  <thead>
    <tr>
      <th>ไฟล์</th><th>เวลา</th><th>ข้อความ</th>
      <th>คำผิดสะกด</th><th>OCR ไม่ชัด</th><th>ปัญหาแบรนด์</th>
    </tr>
  </thead>
  <tbody>{all_rows_html}</tbody>
</table>
</body>
</html>"""


# ─────────────────────────────────────────────
# Process Image
# ─────────────────────────────────────────────

def process_image(
    uploaded_file,
    reader,
    thai_dict: set,
    ignore_words: set,
    brands: list,
    corrections: dict,
    min_confidence: float,
    spell_conf_threshold: float,
    ocr_misread_threshold: float,
    min_token_len: int,
    preprocess: bool,
    upscale_factor: float,
    do_denoise: bool,
    do_binarize: bool,
    phrases: list = None,
    phrase_threshold: float = 0.7,
) -> list:
    """ประมวลผลไฟล์รูปภาพ คืน list[CheckResult]"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return []

    raw_text, ocr_details = ocr_image(
        reader, img, min_confidence, corrections,
        phrases or [], phrase_threshold,
        preprocess, upscale_factor, do_denoise, do_binarize,
    )
    full_text, wrong_words, brand_issues = process_ocr_for_spelling(
        ocr_details, thai_dict, ignore_words, brands,
        min_token_len, spell_conf_threshold, ocr_misread_threshold,
    )

    return [CheckResult(
        file_type    = "image",
        filename     = uploaded_file.name,
        timestamp    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        raw_text     = full_text,
        wrong_words  = wrong_words,
        brand_issues = brand_issues,
        frame        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        ocr_details  = ocr_details,
    )]


# ─────────────────────────────────────────────
# Process Video
# ─────────────────────────────────────────────

def process_video(
    uploaded_file,
    reader,
    thai_dict: set,
    ignore_words: set,
    brands: list,
    corrections: dict,
    min_confidence: float,
    spell_conf_threshold: float,
    ocr_misread_threshold: float,
    min_token_len: int,
    sample_every_sec: int,
    preprocess: bool,
    upscale_factor: float,
    do_denoise: bool,
    do_binarize: bool,
    phrases: list = None,
    phrase_threshold: float = 0.7,
    progress_cb=None,
) -> list:
    """ประมวลผลไฟล์วิดีโอ สุ่มตรวจทุก N วินาที คืน list[CheckResult]"""
    results = []

    # บันทึก temp file เพราะ OpenCV ต้องการ path จริง
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass
        return []

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames  = max(1, int(fps * sample_every_sec))
    total_steps  = max(1, total_frames // step_frames)

    frame_idx = 0
    step_num  = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps
        ts_str        = str(datetime.timedelta(seconds=int(timestamp_sec)))

        raw_text, ocr_details = ocr_image(
            reader, frame, min_confidence, corrections,
            phrases or [], phrase_threshold,
            preprocess, upscale_factor, do_denoise, do_binarize,
        )
        full_text, wrong_words, brand_issues = process_ocr_for_spelling(
            ocr_details, thai_dict, ignore_words, brands,
            min_token_len, spell_conf_threshold, ocr_misread_threshold,
        )

        if full_text.strip():
            results.append(CheckResult(
                file_type    = "video",
                filename     = f"{uploaded_file.name} [{ts_str}]",
                timestamp    = ts_str,
                raw_text     = full_text,
                wrong_words  = wrong_words,
                brand_issues = brand_issues,
                frame        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                ocr_details  = ocr_details,
            ))

        step_num  += 1
        frame_idx += step_frames

        if progress_cb:
            progress_cb(min(step_num / total_steps, 1.0))

        if frame_idx >= total_frames:
            break

    cap.release()
    try:
        Path(tmp_path).unlink()
    except Exception:
        pass

    return results


# ─────────────────────────────────────────────
# CSS — Dark Theme
# ─────────────────────────────────────────────

CSS = """
<style>
/* Dark base */
.stApp { background: #0f172a; }

/* Title */
.main-title {
  font-size: 2.4rem; font-weight: 900;
  background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1.2; margin-bottom: 4px;
}
.main-subtitle { color: #64748b; font-size: 1rem; margin-bottom: 24px; }

/* Highlight classes */
.wrong-word {
  color: #ff4d4d; font-weight: 800;
  text-decoration: underline wavy #ff4d4d;
}
/* OCR อาจอ่านผิด — สีส้มอ่อน ขีดเส้นประ */
.ocr-misread {
  color: #fb923c; font-weight: 700;
  border-bottom: 2px dashed #fb923c;
}
.ocr-unclear { color: #94a3b8; border-bottom: 1px dashed #64748b; }
.dismissed-word { color: #475569; text-decoration: line-through; }

/* Badge base + variants */
.badge-spell-error, .badge-ocr-unclear, .badge-ocr-misread,
.badge-brand-warn, .badge-unit-error {
  display: inline-block; border-radius: 999px;
  padding: 2px 10px; font-size: 0.82rem; font-weight: 700; margin: 2px;
}
.badge-spell-error {
  background: rgba(239,68,68,0.15); color: #f87171;
  border: 1px solid rgba(239,68,68,0.3);
}
/* OCR อาจอ่านผิด — ส้ม */
.badge-ocr-misread {
  background: rgba(251,146,60,0.15); color: #fb923c;
  border: 1px solid rgba(251,146,60,0.35);
}
.badge-ocr-unclear {
  background: rgba(100,116,139,0.12); color: #94a3b8;
  border: 1px solid rgba(100,116,139,0.25);
}
.badge-brand-warn {
  background: rgba(251,191,36,0.15); color: #fbbf24;
  border: 1px solid rgba(251,191,36,0.3);
}
.badge-unit-error {
  background: rgba(251,146,60,0.15); color: #fb923c;
  border: 1px solid rgba(251,146,60,0.3);
}
/* เบอร์โทรผิด — สีแดงสด */
.badge-phone-error {
  background: rgba(239,68,68,0.15); color: #f87171;
  border: 1px solid rgba(239,68,68,0.35);
}
.phone-alert {
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.3);
  border-radius: 10px; padding: 10px 16px; margin: 6px 0;
  font-size: 0.9rem; color: #fca5a5;
}

/* File count chips */
.count-chip {
  display: inline-block; background: #1e293b;
  border: 1px solid #334155; border-radius: 8px;
  padding: 6px 14px; margin: 4px;
  font-size: 0.9rem; color: #cbd5e1;
}

/* Brand / unit alert cards */
.brand-alert {
  background: rgba(251,191,36,0.08);
  border: 1px solid rgba(251,191,36,0.25);
  border-radius: 10px; padding: 10px 16px; margin: 6px 0;
  font-size: 0.9rem; color: #fde68a;
}
.unit-alert {
  background: rgba(251,146,60,0.08);
  border: 1px solid rgba(251,146,60,0.25);
  border-radius: 10px; padding: 10px 16px; margin: 6px 0;
  font-size: 0.9rem; color: #fed7aa;
}

/* Metric card */
.metric-card {
  background: #1e293b; border: 1px solid #334155;
  border-radius: 12px; padding: 16px; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 900; color: #e2e8f0; }
.metric-label { font-size: 0.8rem; color: #64748b; margin-top: 4px; }

/* OCR text display */
.ocr-text-box {
  background: #0f172a; border: 1px solid #334155;
  border-radius: 8px; padding: 14px;
  font-size: 0.95rem; line-height: 1.8; color: #cbd5e1;
  font-family: 'Sarabun', sans-serif;
}

/* Whitelist chip */
.wl-chip {
  display: inline-block; background: #1e293b;
  border: 1px solid #334155; border-radius: 999px;
  padding: 3px 12px; margin: 3px;
  font-size: 0.82rem; color: #94a3b8;
}
</style>
"""


# ─────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title = "Thai Spell Checker",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────

if "scan_results"    not in st.session_state:
    st.session_state["scan_results"]    = []
if "dismissed_words" not in st.session_state:
    st.session_state["dismissed_words"] = set()
if "scan_elapsed"    not in st.session_state:
    st.session_state["scan_elapsed"]    = 0.0
if "ocr_log"         not in st.session_state:
    st.session_state["ocr_log"]         = []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 Thai Spell Checker")
    st.markdown("<small style='color:#64748b;'>EasyOCR + PyThaiNLP</small>", unsafe_allow_html=True)
    st.divider()

    # ── OCR Settings ──
    st.markdown("### ⚙️ OCR Settings")
    min_confidence = st.slider(
        "Confidence ขั้นต่ำ",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="กรอง OCR box ที่มี confidence ต่ำกว่านี้ออก",
    )
    spell_conf_threshold = st.slider(
        "Threshold ตรวจสะกด",
        min_value=0.5, max_value=0.9, value=0.7, step=0.05,
        help="box ที่ confidence ต่ำกว่านี้จะถูก mark ว่า OCR ไม่ชัด",
    )
    ocr_misread_threshold = st.slider(
        "Threshold คำผิดจริง",
        min_value=0.7, max_value=1.0, value=0.85, step=0.05,
        help=(
            "≥ ค่านี้ → 🔴 คำผิดจริง (spell_error)\n"
            "ระหว่างสองค่า → 🟠 OCR อาจอ่านผิด (ocr_misread)"
        ),
    )
    min_token_len = st.slider(
        "ความยาวคำขั้นต่ำ",
        min_value=2, max_value=5, value=3, step=1,
        help="คำที่สั้นกว่านี้จะถูกข้ามการตรวจสะกด",
    )

    st.divider()

    # ── Image Preprocessing ──
    st.markdown("### 🖼️ Image Preprocessing")
    preprocess    = st.checkbox("เปิดใช้ Preprocessing", value=True,
                                help="เพิ่มคุณภาพภาพก่อนส่ง OCR (แนะนำให้เปิด)")
    upscale_factor = st.slider("Upscale factor", 1.0, 4.0, 2.0, 0.5,
                                help="ขยายภาพก่อน OCR (2× ดีสำหรับข้อความเล็ก)",
                                disabled=not preprocess)
    do_denoise    = st.checkbox("Bilateral Denoise", value=True,
                                help="กรอง noise รักษาขอบตัวอักษร",
                                disabled=not preprocess)
    do_binarize   = st.checkbox("Adaptive Binarize", value=False,
                                help="ทำขาว-ดำ เหมาะกับป้ายบนพื้นเรียบ (ปิดถ้าภาพมีสีสลับซับซ้อน)",
                                disabled=not preprocess)

    st.divider()

    # ── Video Settings ──
    st.markdown("### 🎬 Video Settings")
    sample_every_sec = st.slider(
        "สุ่มตรวจทุกกี่วินาที",
        min_value=1, max_value=10, value=2, step=1,
    )
    use_gpu = False  # pytesseract ใช้ CPU เสมอ

    st.divider()

    # ── Brand Dictionary ──
    st.markdown("### 📚 พจนานุกรมแบรนด์")
    brands = load_brands()

    with st.expander("จัดการแบรนด์", expanded=True):

        # Import
        import_file = st.file_uploader("นำเข้า brands.json", type="json", key="brand_import")
        if import_file:
            try:
                imported     = json.loads(import_file.read())
                existing_ids = {b["id"] for b in brands}
                new_brands   = [b for b in imported.get("brands", [])
                                if b["id"] not in existing_ids]
                if new_brands:
                    save_brands(brands + new_brands)
                    st.success(f"นำเข้า {len(new_brands)} แบรนด์สำเร็จ")
                    st.rerun()
                else:
                    st.info("ไม่มีแบรนด์ใหม่ที่จะนำเข้า")
            except Exception as e:
                st.error(f"นำเข้าล้มเหลว: {e}")

        # Export
        export_data = json.dumps({"brands": brands}, ensure_ascii=False, indent=2)
        st.download_button(
            label            = "⬇️ Export brands.json",
            data             = export_data.encode("utf-8"),
            file_name        = "brands.json",
            mime             = "application/json",
            use_container_width = True,
        )

        st.markdown("---")

        # Brand list with delete buttons
        for brand in brands:
            c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 2, 1])
            c1.markdown(f"<small style='color:#94a3b8;'>{brand['english']}</small>",
                        unsafe_allow_html=True)
            c2.markdown(f"<small style='color:#e2e8f0;'>{brand['thai']}</small>",
                        unsafe_allow_html=True)
            c3.markdown(f"<small style='color:#64748b;'>{brand['unit']}</small>",
                        unsafe_allow_html=True)
            c4.markdown(
                f"<small style='color:#38bdf8;'>{brand.get('phone','—')}</small>",
                unsafe_allow_html=True,
            )
            if c5.button("🗑️", key=f"del_brand_{brand['id']}", help=f"ลบ {brand['english']}"):
                brands = [b for b in brands if b["id"] != brand["id"]]
                save_brands(brands)
                st.rerun()

        st.markdown("---")
        st.markdown("**เพิ่มแบรนด์ใหม่**")

        with st.form("add_brand_form", clear_on_submit=True):
            new_en    = st.text_input("ชื่อภาษาอังกฤษ (ID)", placeholder="เช่น mybrand")
            new_th    = st.text_input("ชื่อภาษาไทย",         placeholder="เช่น มายแบรนด์")
            new_unit  = st.selectbox("หน่วย", UNIT_OPTIONS)
            new_phone = st.text_input("เบอร์โทร", placeholder="เช่น 063-479-1496")
            submitted = st.form_submit_button("➕ เพิ่มแบรนด์", use_container_width=True)

            if submitted:
                if new_en.strip() and new_th.strip():
                    new_id = re.sub(r'\s+', '', new_en.strip().lower())
                    if any(b["id"] == new_id for b in brands):
                        st.error("มีแบรนด์นี้อยู่แล้ว")
                    else:
                        brands.append({
                            "id":      new_id,
                            "english": new_en.strip(),
                            "thai":    new_th.strip(),
                            "unit":    new_unit,
                            "phone":   new_phone.strip(),
                        })
                        save_brands(brands)
                        st.success(f"เพิ่ม '{new_en.strip()}' สำเร็จ")
                        st.rerun()
                else:
                    st.warning("กรุณากรอกชื่อภาษาอังกฤษและภาษาไทย")

    st.divider()

    # ── Whitelist ──
    st.markdown("### 🚫 คำที่ไม่ต้องตรวจ (Whitelist)")
    persistent_whitelist = load_whitelist()

    if persistent_whitelist:
        chips_html = "".join(
            f'<span class="wl-chip">{html.escape(w)}</span>'
            for w in sorted(persistent_whitelist)
        )
        st.markdown(chips_html, unsafe_allow_html=True)

        if st.button("🗑️ ลบทั้งหมด", key="clear_whitelist", use_container_width=True):
            save_whitelist(set())
            st.rerun()
    else:
        st.markdown(
            "<small style='color:#475569;'>ยังไม่มีคำใน whitelist</small>",
            unsafe_allow_html=True,
        )

    wl_input = st.text_area(
        "เพิ่มคำ (แต่ละบรรทัด)",
        height=80,
        placeholder="คำ1\nคำ2\nคำ3",
        key="wl_text_input",
    )
    if st.button("➕ เพิ่มใน Whitelist", key="add_wl", use_container_width=True):
        new_words = {w.strip() for w in wl_input.splitlines() if w.strip()}
        if new_words:
            updated = persistent_whitelist | new_words
            save_whitelist(updated)
            st.success(f"เพิ่ม {len(new_words)} คำ")
            st.rerun()

    st.divider()

    # ── Phone Rules ──
    st.markdown("### 📞 กฎเบอร์โทรศัพท์")
    phone_data = load_phones()

    with st.expander(f"จัดการกฎเบอร์โทร ({len(phone_data.get('rules', []))} กฎ)", expanded=False):

        # Export
        st.download_button(
            label               = "⬇️ Export phones.json",
            data                = json.dumps(phone_data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name           = "phones.json",
            mime                = "application/json",
            use_container_width = True,
            key                 = "export_phones",
        )
        st.markdown("---")

        # แสดงกฎปัจจุบัน + ปุ่มลบ
        for r_i, rule in enumerate(phone_data.get("rules", [])):
            pr1, pr2, pr3 = st.columns([2, 3, 1])
            pr1.markdown(
                f"<small style='color:#38bdf8;font-weight:700;'>{html.escape(rule['phone'])}</small>",
                unsafe_allow_html=True,
            )
            pr2.markdown(
                f"<small style='color:#94a3b8;'>{html.escape(', '.join(rule.get('product_ids', [])))}</small>",
                unsafe_allow_html=True,
            )
            if pr3.button("🗑️", key=f"del_phone_rule_{r_i}"):
                phone_data["rules"].pop(r_i)
                save_phones(phone_data)
                st.rerun()

        st.markdown("---")
        st.markdown("**เพิ่มกฎใหม่**")
        with st.form("add_phone_rule_form", clear_on_submit=True):
            np1, np2 = st.columns(2)
            new_ph    = np1.text_input("เบอร์โทร", placeholder="063-479-1496")
            new_ids   = np2.text_input("IDs สินค้า (คั่นด้วย ,)", placeholder="betalife,betaherb")
            new_ph_desc = st.text_input("คำอธิบาย (optional)")
            if st.form_submit_button("➕ เพิ่มกฎ", use_container_width=True):
                if new_ph.strip() and new_ids.strip():
                    phone_data["rules"].append({
                        "phone":       new_ph.strip(),
                        "product_ids": [x.strip() for x in new_ids.split(",") if x.strip()],
                        "description": new_ph_desc.strip(),
                    })
                    if new_ph.strip() not in phone_data.get("known_phones", []):
                        phone_data.setdefault("known_phones", []).append(new_ph.strip())
                    save_phones(phone_data)
                    st.success(f"เพิ่มกฎ {new_ph.strip()} สำเร็จ")
                    st.rerun()
                else:
                    st.warning("กรุณากรอกเบอร์โทรและ IDs สินค้า")

    st.divider()

    # ── OCR Corrections Table ──
    st.markdown("### 🔤 ตาราง OCR อ่านผิด")
    ocr_corrections = load_ocr_corrections()

    with st.expander(f"จัดการ corrections ({len(ocr_corrections)} รายการ)", expanded=False):
        # Export
        st.download_button(
            label            = "⬇️ Export corrections.json",
            data             = json.dumps({"corrections": ocr_corrections}, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name        = "ocr_corrections.json",
            mime             = "application/json",
            use_container_width = True,
            key              = "export_corrections",
        )

        st.markdown("---")

        # แสดงรายการ + ปุ่มลบ
        for wrong, right in list(ocr_corrections.items()):
            cc1, cc2, cc3, cc4 = st.columns([2, 1, 2, 1])
            cc1.markdown(f"<small style='color:#f87171;'>**{html.escape(wrong)}**</small>",
                         unsafe_allow_html=True)
            cc2.markdown("<small style='color:#475569;'>→</small>", unsafe_allow_html=True)
            cc3.markdown(f"<small style='color:#4ade80;'>{html.escape(right)}</small>",
                         unsafe_allow_html=True)
            if cc4.button("🗑️", key=f"del_corr_{wrong}", help=f"ลบ {wrong}→{right}"):
                del ocr_corrections[wrong]
                save_ocr_corrections(ocr_corrections)
                st.rerun()

        st.markdown("---")
        st.markdown("**เพิ่มคู่ correction ใหม่**")
        with st.form("add_correction_form", clear_on_submit=True):
            fc1, fc2 = st.columns(2)
            new_wrong = fc1.text_input("OCR อ่านผิดเป็น", placeholder="เช่น มู้")
            new_right = fc2.text_input("ที่ถูกต้องคือ",   placeholder="เช่น ผู้")
            if st.form_submit_button("➕ เพิ่ม", use_container_width=True):
                if new_wrong.strip() and new_right.strip():
                    ocr_corrections[new_wrong.strip()] = new_right.strip()
                    save_ocr_corrections(ocr_corrections)
                    st.success(f"เพิ่ม '{new_wrong}' → '{new_right}'")
                    st.rerun()
                else:
                    st.warning("กรุณากรอกทั้งสองช่อง")

    st.divider()

    # ── Promotion Requirements ──
    st.markdown("### 📋 ข้อมูลโปรสงกรานต์")
    with st.expander("จัดการข้อมูลโปรสงกรานต์", expanded=False):
        brands_with_promo = [b for b in brands if b.get("promotion_requirements")]
        
        if brands_with_promo:
            promo_brand = st.selectbox(
                "เลือกแบรนด์เพื่อตรวจสอบข้อมูล",
                options=brands_with_promo,
                format_func=lambda b: f"{b['thai']} ({b['english']})",
                key="promo_brand_select",
            )
            
            if promo_brand:
                st.markdown(f"**{promo_brand['thai']}** ({promo_brand['english']})")
                
                promo_reqs = promo_brand.get("promotion_requirements", {})
                req_list = [
                    ("duration", "ระยะเวลาโปรโมชั่น"),
                    ("price", "ราคา"),
                    ("benefits_gifts", "สิทธิ/ของแถม"),
                    ("promotion_type", "ประเภทโปรโมชั่น"),
                    ("bundle_exclusivity", "เงื่อนไขสินค้าแถม"),
                ]
                
                st.markdown("**ข้อมูลที่จำเป็น:**")
                for req_key, req_label in req_list:
                    status = promo_reqs.get(req_key, "optional")
                    badge = "🔴 บังคับ" if status == "required" else "🟢 ไม่บังคับ"
                    st.markdown(f"  • {req_label}: {badge}")
                
                st.info("💡 ข้อมูลเหล่านี้จะถูกตรวจสอบจากข้อความในรูปภาพ/วิดีโอ")
        else:
            st.warning("ไม่มีแบรนด์ที่มีข้อมูลโปรสงกรานต์")

    st.divider()

    # ── Phrase Templates ──
    st.markdown("### 💬 ประโยคต้นแบบ (Phrase Templates)")
    phrase_data       = load_phrases()
    phrase_list       = phrase_data.get("phrases", [])
    phrase_threshold  = st.slider(
        "Phrase match threshold",
        min_value=0.5, max_value=1.0, value=float(phrase_data.get("threshold", 0.7)), step=0.05,
        help="ถ้า OCR อ่านได้คล้ายประโยคต้นแบบ ≥ ค่านี้ จะ auto-correct เป็นประโยคต้นแบบ",
        key="phrase_threshold_slider",
    )
    # บันทึก threshold ล่าสุดกลับไป JSON ถ้าเปลี่ยน
    if phrase_threshold != phrase_data.get("threshold"):
        phrase_data["threshold"] = phrase_threshold
        save_phrases(phrase_data)

    with st.expander(f"จัดการประโยคต้นแบบ ({len(phrase_list)} ประโยค)", expanded=False):

        st.download_button(
            label               = "⬇️ Export phrases.json",
            data                = json.dumps(phrase_data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name           = "phrases.json",
            mime                = "application/json",
            use_container_width = True,
            key                 = "export_phrases",
        )
        st.markdown("---")

        # แสดงรายการ + ปุ่มลบ
        for p_i, phrase in enumerate(phrase_list):
            pp1, pp2 = st.columns([5, 1])
            pp1.markdown(
                f"<small style='color:#a5f3fc;'>{html.escape(phrase)}</small>",
                unsafe_allow_html=True,
            )
            if pp2.button("🗑️", key=f"del_phrase_{p_i}", help=f"ลบ '{phrase}'"):
                phrase_data["phrases"].pop(p_i)
                save_phrases(phrase_data)
                st.rerun()

        st.markdown("---")
        st.markdown("**เพิ่มประโยคต้นแบบใหม่**")
        with st.form("add_phrase_form", clear_on_submit=True):
            new_phrase = st.text_input("ประโยคต้นแบบ", placeholder="เช่น คุมน้ำตาลให้ดี")
            if st.form_submit_button("➕ เพิ่ม", use_container_width=True):
                if new_phrase.strip():
                    if new_phrase.strip() not in phrase_data["phrases"]:
                        phrase_data["phrases"].append(new_phrase.strip())
                        save_phrases(phrase_data)
                        st.success(f"เพิ่ม '{new_phrase.strip()}' สำเร็จ")
                        st.rerun()
                    else:
                        st.info("มีประโยคนี้อยู่แล้ว")
                else:
                    st.warning("กรุณากรอกประโยค")

    st.divider()

    # ── Dictionary info footer ──
    with st.spinner("โหลดพจนานุกรม…"):
        thai_dict = load_dictionary()

    st.markdown(
        f"<small style='color:#475569;'>"
        f"📖 พจนานุกรม: {len(thai_dict):,} คำ<br>"
        f"🏷️ แบรนด์: {len(brands)} รายการ<br>"
        f"🚫 Whitelist: {len(persistent_whitelist)} คำ<br>"
        f"💬 Phrase templates: {len(phrase_list)} ประโยค"
        f"</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────

st.markdown('<div class="main-title">🔍 Thai Spell Checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    'ตรวจสะกดภาษาไทยจากรูปภาพและวิดีโอ ด้วย EasyOCR + PyThaiNLP'
    '</div>',
    unsafe_allow_html=True,
)

# ── File Uploader ──
IMAGE_EXTS = {"png", "jpg", "jpeg", "bmp", "webp"}
VIDEO_EXTS = {"mp4", "avi", "mov", "mkv"}

uploaded_files = st.file_uploader(
    "อัปโหลดรูปภาพหรือวิดีโอ",
    type=list(IMAGE_EXTS | VIDEO_EXTS),
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    n_images = sum(1 for f in uploaded_files
                   if Path(f.name).suffix.lstrip(".").lower() in IMAGE_EXTS)
    n_videos = sum(1 for f in uploaded_files
                   if Path(f.name).suffix.lstrip(".").lower() in VIDEO_EXTS)
    n_total  = len(uploaded_files)
    st.markdown(
        f'<span class="count-chip">🖼️ รูปภาพ: {n_images}</span>'
        f'<span class="count-chip">🎬 วิดีโอ: {n_videos}</span>'
        f'<span class="count-chip">📁 รวม: {n_total}</span>',
        unsafe_allow_html=True,
    )

# ── Scan Button ──
run = st.button(
    "🚀 เริ่มตรวจสอบ",
    type="primary",
    disabled=not uploaded_files,
)

if run and uploaded_files:
    # รีเซ็ต state สำหรับการสแกนใหม่
    st.session_state["scan_results"]    = []
    st.session_state["dismissed_words"] = set()
    st.session_state["scan_elapsed"]    = 0.0
    st.session_state["ocr_log"]         = []
    ocr_log_clear()  # รีเซ็ต module-level log

    persistent_whitelist = load_whitelist()
    brands               = load_brands()
    ocr_corrections      = load_ocr_corrections()
    _phrase_data         = load_phrases()
    _phrases             = _phrase_data.get("phrases", [])
    _phrase_threshold    = float(_phrase_data.get("threshold", 0.7))
    effective_ignore     = persistent_whitelist | {b["thai"] for b in brands}

    with st.spinner("กำลังตรวจสอบ Tesseract OCR…"):
        reader = load_ocr()

    all_results  = []
    t_start      = time.time()
    progress_bar = st.progress(0, text="กำลังประมวลผล…")

    for f_idx, uploaded_file in enumerate(uploaded_files):
        ext = Path(uploaded_file.name).suffix.lstrip(".").lower()
        progress_bar.progress(
            f_idx / len(uploaded_files),
            text=f"กำลังประมวลผล {uploaded_file.name}…",
        )

        if ext in IMAGE_EXTS:
            results = process_image(
                uploaded_file, reader, thai_dict, effective_ignore, brands,
                ocr_corrections, min_confidence, spell_conf_threshold,
                ocr_misread_threshold, min_token_len,
                preprocess, upscale_factor, do_denoise, do_binarize,
                _phrases, _phrase_threshold,
            )
            all_results.extend(results)

        elif ext in VIDEO_EXTS:

            def make_progress_cb(bar, fname, total, idx):
                def cb(pct):
                    overall = (idx + pct) / total
                    bar.progress(min(overall, 1.0), text=f"{fname} — {int(pct*100)}%")
                return cb

            results = process_video(
                uploaded_file, reader, thai_dict, effective_ignore, brands,
                ocr_corrections, min_confidence, spell_conf_threshold,
                ocr_misread_threshold, min_token_len, sample_every_sec,
                preprocess, upscale_factor, do_denoise, do_binarize,
                _phrases, _phrase_threshold,
                progress_cb=make_progress_cb(
                    progress_bar, uploaded_file.name, len(uploaded_files), f_idx
                ),
            )
            all_results.extend(results)

    progress_bar.progress(1.0, text="เสร็จสิ้น!")
    elapsed = time.time() - t_start

    st.session_state["scan_results"]  = all_results
    st.session_state["scan_elapsed"]  = elapsed
    st.session_state["ocr_log"]       = list(_OCR_LOG)  # snapshot ณ ตอนที่สแกนเสร็จ


# ─────────────────────────────────────────────
# RESULTS SECTION
# อยู่นอก if run: เพื่อให้คงผลหลัง rerun จาก whitelist / dismiss buttons
# ─────────────────────────────────────────────

all_results          = st.session_state.get("scan_results", [])
dismissed_words      = st.session_state.get("dismissed_words", set())
scan_elapsed         = st.session_state.get("scan_elapsed", 0.0)
ocr_log              = st.session_state.get("ocr_log", [])
persistent_whitelist = load_whitelist()   # โหลดใหม่เสมอ สะท้อน save ล่าสุด
brands               = load_brands()

# ── OCR Auto-Correct Log (debug) ──
if ocr_log:
    n_char   = sum(1 for e in ocr_log if e.get("type") == "char")
    n_phrase = sum(1 for e in ocr_log if e.get("type") == "phrase")
    with st.expander(
        f"🔧 OCR Auto-Correct Log ({n_char} char, {n_phrase} phrase corrections)",
        expanded=False,
    ):
        st.markdown(
            "<small style='color:#64748b;'>ทุก correction ที่เกิดขึ้นระหว่างการสแกน "
            "(แสดงเฉพาะที่มีการเปลี่ยนแปลงจริง)</small>",
            unsafe_allow_html=True,
        )
        for entry in ocr_log:
            t    = entry.get("type", "?")
            orig = html.escape(entry.get("original", ""))
            corr = html.escape(entry.get("corrected", ""))
            if t == "char":
                st.markdown(
                    f'<span style="color:#f87171;">**[char]**</span> '
                    f'`{orig}` → <span style="color:#4ade80;">`{corr}`</span>',
                    unsafe_allow_html=True,
                )
            elif t == "phrase":
                ratio = entry.get("ratio", 0)
                ctx   = html.escape(entry.get("original", ""))
                st.markdown(
                    f'<span style="color:#38bdf8;">**[phrase {ratio:.0%}]**</span> '
                    f'`{ctx}` → <span style="color:#4ade80;">`{corr}`</span>',
                    unsafe_allow_html=True,
                )

if all_results:

    # ── Metrics ──
    total_spell = sum(
        len([w for w in r.wrong_words
             if w["conf_type"] == "spell_error"
             and w["word"] not in dismissed_words
             and w["word"] not in persistent_whitelist])
        for r in all_results
    )
    total_unclear = sum(
        len([w for w in r.wrong_words if w["conf_type"] == "ocr_unclear"])
        for r in all_results
    )
    total_brand = sum(
        len([bi for bi in r.brand_issues 
             if bi.issue_type not in ("wrong_phone",) and not bi.issue_type.startswith("missing_promotion_")])
        for r in all_results
    )
    total_phone = sum(
        len([bi for bi in r.brand_issues if bi.issue_type == "wrong_phone"])
        for r in all_results
    )
    total_promo = sum(
        len([bi for bi in r.brand_issues if bi.issue_type.startswith("missing_promotion_")])
        for r in all_results
    )

    st.markdown("### 📊 สรุป")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

    def metric_card(col, value, label, color="#e2e8f0"):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color};">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    metric_card(m1, len(all_results), "จุดตรวจ")
    metric_card(m2, total_spell,      "พบผิดสะกด",   "#f87171")
    metric_card(m3, total_unclear,    "OCR ไม่ชัด",  "#94a3b8")
    metric_card(m4, total_brand,      "ปัญหาแบรนด์", "#fbbf24")
    metric_card(m5, total_phone,      "เบอร์ผิด",    "#f87171")
    metric_card(m6, total_promo,      "โปรสงกรานต์", "#ef4444")
    metric_card(m7, f"{scan_elapsed:.1f}s", "เวลา")

    st.divider()

    # ── Brand Issues Summary (deduplicated across all files) ──
    all_brand_issues = []
    seen_brand_keys  = set()
    for r in all_results:
        for bi in r.brand_issues:
            key = (bi.issue_type, bi.found, bi.expected)
            if key not in seen_brand_keys:
                seen_brand_keys.add(key)
                all_brand_issues.append(bi)

    if all_brand_issues:
        st.markdown("### 🏷️ ปัญหาที่พบเกี่ยวกับแบรนด์")
        
        # แยกประเภท issues
        promo_issues = [bi for bi in all_brand_issues if bi.issue_type.startswith("missing_promotion_")]
        other_issues = [bi for bi in all_brand_issues if not bi.issue_type.startswith("missing_promotion_")]
        
        # แสดง non-promo issues ก่อน
        for bi in other_issues:
            if bi.issue_type == "brand_thai_misspell":
                st.markdown(
                    f'<div class="brand-alert">'
                    f'⚠️ พบคำที่คล้าย <strong>{html.escape(bi.brand_thai)}</strong> '
                    f'→ "<em>{html.escape(bi.found)}</em>" อาจสะกดผิด '
                    f'แนะนำ "<strong>{html.escape(bi.expected)}</strong>"'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif bi.issue_type == "wrong_unit":
                st.markdown(
                    f'<div class="unit-alert">'
                    f'⚠️ <strong>{html.escape(bi.brand_thai)}</strong> '
                    f'ควรใช้หน่วย "<strong>{html.escape(bi.expected)}</strong>" '
                    f'ไม่ใช่ "<em>{html.escape(bi.found)}</em>"'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif bi.issue_type == "wrong_phone":
                st.markdown(
                    f'<div class="phone-alert">'
                    f'📞 <strong>{html.escape(bi.brand_thai)}</strong> '
                    f'ควรใช้เบอร์ "<strong>{html.escape(bi.expected)}</strong>" '
                    f'ไม่ใช่ "<em>{html.escape(bi.found)}</em>"'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        
        # แสดง promo issues แยกจากกัน
        if promo_issues:
            st.markdown("---")
            st.markdown("### 📌 ข้อมูลโปรสงกรานต์ที่ขาดหายไป")
            for bi in promo_issues:
                req_name = bi.issue_type.replace("missing_promotion_", "").replace("_", " ")
                st.markdown(
                    f'<div style="background:rgba(239,68,68,.1); border-left:4px solid #ef4444; '
                    f'padding:12px; margin:8px 0; border-radius:4px;">'
                    f'📋 <strong>{html.escape(bi.brand_thai)}</strong><br>'
                    f'⚠️ ขาดหายไป: <strong>{req_name.title()}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        
        st.divider()

    # ── Per-file Results ──
    st.markdown("### 📁 ผลลัพธ์แยกตามไฟล์")

    for r_idx, r in enumerate(all_results):
        spell_count   = len([w for w in r.wrong_words
                              if w["conf_type"] == "spell_error"
                              and w["word"] not in dismissed_words
                              and w["word"] not in persistent_whitelist])
        unclear_count = len([w for w in r.wrong_words if w["conf_type"] == "ocr_unclear"])
        brand_count   = len([bi for bi in r.brand_issues 
                            if bi.issue_type not in ("wrong_phone",) and not bi.issue_type.startswith("missing_promotion_")])
        phone_count   = len([bi for bi in r.brand_issues if bi.issue_type == "wrong_phone"])
        promo_count   = len([bi for bi in r.brand_issues if bi.issue_type.startswith("missing_promotion_")])

        parts = [f"คำผิด: {spell_count}", f"OCR ไม่ชัด: {unclear_count}",
                 f"แบรนด์: {brand_count}"]
        if phone_count:
            parts.append(f"⚠️ เบอร์ผิด: {phone_count}")
        if promo_count:
            parts.append(f"📌 โปร: {promo_count}")
        label = f"{r.filename}  [{' | '.join(parts)}]"

        with st.expander(label, expanded=(r_idx == 0)):

            # Frame preview
            if r.frame is not None:
                st.image(r.frame, caption=r.filename, use_container_width=True)

            # Highlighted OCR text
            highlighted = highlight_text(
                r.raw_text, r.wrong_words, dismissed_words, persistent_whitelist
            )
            st.markdown(
                f'<div class="ocr-text-box">{highlighted}</div>',
                unsafe_allow_html=True,
            )

            # Brand issues for this file
            if r.brand_issues:
                st.markdown("**🏷️ ปัญหาแบรนด์ในไฟล์นี้**")
                
                # แยก issues
                promo_issues = [bi for bi in r.brand_issues if bi.issue_type.startswith("missing_promotion_")]
                other_issues = [bi for bi in r.brand_issues if not bi.issue_type.startswith("missing_promotion_")]
                
                # แสดง non-promo issues
                for bi in other_issues:
                    if bi.issue_type == "brand_thai_misspell":
                        st.markdown(
                            f'<div class="brand-alert">'
                            f'⚠️ พบคำที่คล้าย <strong>{html.escape(bi.brand_thai)}</strong> '
                            f'→ "{html.escape(bi.found)}" อาจสะกดผิด '
                            f'แนะนำ "{html.escape(bi.expected)}"'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif bi.issue_type == "wrong_unit":
                        st.markdown(
                            f'<div class="unit-alert">'
                            f'⚠️ <strong>{html.escape(bi.brand_thai)}</strong> '
                            f'ควรใช้หน่วย "{html.escape(bi.expected)}" '
                            f'ไม่ใช่ "{html.escape(bi.found)}"'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif bi.issue_type == "wrong_phone":
                        st.markdown(
                            f'<div class="phone-alert">'
                            f'📞 <strong>{html.escape(bi.brand_thai)}</strong> '
                            f'ควรใช้เบอร์ "{html.escape(bi.expected)}" '
                            f'ไม่ใช่ "{html.escape(bi.found)}"'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                
                # แสดง promo issues
                if promo_issues:
                    st.markdown("---")
                    st.markdown("**📌 ข้อมูลโปรสงกรานต์ที่ขาดหายไป**")
                    for bi in promo_issues:
                        req_name = bi.issue_type.replace("missing_promotion_", "").replace("_", " ")
                        st.markdown(
                            f'<div style="background:rgba(239,68,68,.1); border-left:4px solid #ef4444; '
                            f'padding:12px; margin:8px 0; border-radius:4px;">'
                            f'📋 <strong>{html.escape(bi.brand_thai)}</strong><br>'
                            f'⚠️ ขาดหายไป: <strong>{req_name.title()}</strong>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # ── Word Review Panel (Streamlit-native) ──
            # กรองคำที่ยัง active (ไม่ถูก dismiss/whitelist)
            active_words = [
                w for w in r.wrong_words
                if not (
                    w["conf_type"] in ("spell_error", "ocr_misread")
                    and (w["word"] in dismissed_words or w["word"] in persistent_whitelist)
                )
            ]

            if active_words:
                st.markdown("**🔍 Word Review Panel**")

                # header row
                hc1, hc2, hc3, hc4, hc5 = st.columns([1.5, 2, 2, 1, 2])
                hc1.markdown("<small style='color:#475569;'>ประเภท</small>", unsafe_allow_html=True)
                hc2.markdown("<small style='color:#475569;'>คำที่พบ</small>", unsafe_allow_html=True)
                hc3.markdown("<small style='color:#475569;'>คำแนะนำ</small>", unsafe_allow_html=True)
                hc4.markdown("<small style='color:#475569;'>Conf</small>", unsafe_allow_html=True)
                hc5.markdown("<small style='color:#475569;'>การดำเนินการ</small>", unsafe_allow_html=True)

                for w_idx, item in enumerate(active_words):
                    word       = item["word"]
                    conf_type  = item["conf_type"]
                    suggestion = item["suggestion"]
                    ocr_conf   = item["ocr_conf"]

                    col1, col2, col3, col4, col5 = st.columns([1.5, 2, 2, 1, 2])

                    # col1: badge ตาม conf_type
                    if conf_type == "spell_error":
                        col1.markdown(
                            '<span class="badge-spell-error">🔴 คำผิดจริง</span>',
                            unsafe_allow_html=True,
                        )
                    elif conf_type == "ocr_misread":
                        col1.markdown(
                            '<span class="badge-ocr-misread">🟠 OCR อาจอ่านผิด</span>',
                            unsafe_allow_html=True,
                        )
                    else:  # ocr_unclear
                        col1.markdown(
                            '<span class="badge-ocr-unclear">⚪ OCR ไม่ชัด</span>',
                            unsafe_allow_html=True,
                        )

                    # col2: word
                    col2.markdown(f"**{html.escape(word)}**")

                    # col3: suggestion / คำอธิบาย
                    if conf_type == "spell_error" and suggestion != word:
                        col3.markdown(f"→ **`{html.escape(suggestion)}`**")
                    elif conf_type == "ocr_misread":
                        if suggestion != word:
                            col3.markdown(
                                f"<span style='color:#fb923c;'>→ `{html.escape(suggestion)}`</span>"
                                f"<br><small style='color:#64748b;'>OCR confidence ต่ำ — อาจอ่านผิด</small>",
                                unsafe_allow_html=True,
                            )
                        else:
                            col3.markdown(
                                "<small style='color:#64748b;'>OCR อาจอ่านผิด — ไม่มีคำแนะนำ</small>",
                                unsafe_allow_html=True,
                            )
                    elif conf_type == "ocr_unclear":
                        col3.markdown(
                            "<small style='color:#64748b;'>OCR confidence ต่ำมาก</small>",
                            unsafe_allow_html=True,
                        )
                    else:
                        col3.markdown(
                            "<small style='color:#64748b;'>ไม่พบคำแนะนำ</small>",
                            unsafe_allow_html=True,
                        )

                    # col4: confidence %
                    conf_color = "#f87171" if ocr_conf >= 0.85 else ("#fb923c" if ocr_conf >= 0.7 else "#94a3b8")
                    col4.markdown(
                        f"<small style='color:{conf_color};font-weight:700;'>{ocr_conf:.0%}</small>",
                        unsafe_allow_html=True,
                    )

                    # col5: "ไม่ใช่คำผิด" button สำหรับ spell_error และ ocr_misread
                    if conf_type in ("spell_error", "ocr_misread"):
                        if col5.button(
                            "ไม่ใช่คำผิด ✓",
                            key=f"wl_{word}_{r_idx}_{w_idx}",
                            help=f"เพิ่ม '{word}' ใน whitelist ถาวร",
                        ):
                            wl = load_whitelist()
                            wl.add(word)
                            save_whitelist(wl)
                            st.session_state["dismissed_words"].add(word)
                            st.rerun()
                    else:
                        col5.empty()

            else:
                st.markdown(
                    "<small style='color:#475569;'>"
                    "ไม่พบปัญหาในไฟล์นี้ หรือตรวจสอบทั้งหมดแล้ว"
                    "</small>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Combined HTML Table ──
    st.markdown("### 📋 ตารางรวม")

    table_rows = ""
    for r in all_results:
        highlighted = highlight_text(r.raw_text, r.wrong_words, dismissed_words, persistent_whitelist)
        spell_cnt   = len([w for w in r.wrong_words
                           if w["conf_type"] == "spell_error"
                           and w["word"] not in dismissed_words
                           and w["word"] not in persistent_whitelist])
        unclear_cnt = len([w for w in r.wrong_words if w["conf_type"] == "ocr_unclear"])
        brand_cnt   = len([bi for bi in r.brand_issues if bi.issue_type != "wrong_phone"])
        phone_cnt   = len([bi for bi in r.brand_issues if bi.issue_type == "wrong_phone"])

        table_rows += f"""
        <tr>
          <td style="padding:8px;border-bottom:1px solid #334155;white-space:nowrap;">
            {html.escape(r.filename)}
          </td>
          <td style="padding:8px;border-bottom:1px solid #334155;white-space:nowrap;">
            {html.escape(r.timestamp)}
          </td>
          <td style="padding:8px;border-bottom:1px solid #334155;line-height:1.8;">
            {highlighted}
          </td>
          <td style="padding:8px;border-bottom:1px solid #334155;text-align:center;color:#f87171;">
            {spell_cnt}
          </td>
          <td style="padding:8px;border-bottom:1px solid #334155;text-align:center;color:#94a3b8;">
            {unclear_cnt}
          </td>
          <td style="padding:8px;border-bottom:1px solid #334155;text-align:center;color:#fbbf24;">
            {brand_cnt}
          </td>
          <td style="padding:8px;border-bottom:1px solid #334155;text-align:center;color:#f87171;">
            {"⚠️ " + str(phone_cnt) if phone_cnt else "—"}
          </td>
        </tr>"""

    table_html = f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;background:#1e293b;
                  border-radius:12px;overflow:hidden;font-size:0.88rem;color:#cbd5e1;">
      <thead>
        <tr style="background:#0f172a;">
          <th style="padding:10px 8px;text-align:left;color:#94a3b8;">ไฟล์</th>
          <th style="padding:10px 8px;text-align:left;color:#94a3b8;">เวลา</th>
          <th style="padding:10px 8px;text-align:left;color:#94a3b8;">ข้อความ OCR</th>
          <th style="padding:10px 8px;text-align:center;color:#f87171;">ผิดสะกด</th>
          <th style="padding:10px 8px;text-align:center;color:#94a3b8;">OCR ไม่ชัด</th>
          <th style="padding:10px 8px;text-align:center;color:#fbbf24;">แบรนด์</th>
          <th style="padding:10px 8px;text-align:center;color:#f87171;">เบอร์โทร</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    </div>"""

    st.markdown(table_html, unsafe_allow_html=True)

    st.divider()

    # ── Download HTML Report ──
    report_html = generate_html_report(all_results, dismissed_words, persistent_whitelist)
    st.download_button(
        label     = "⬇️ ดาวน์โหลด HTML Report",
        data      = report_html.encode("utf-8"),
        file_name = f"spell_check_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime      = "text/html",
        type      = "primary",
    )

else:
    # ── Empty state ──
    st.markdown(
        """
        <div style="text-align:center;padding:60px 0;color:#475569;">
          <div style="font-size:4rem;">📂</div>
          <div style="font-size:1.1rem;margin-top:16px;">
            อัปโหลดไฟล์รูปภาพหรือวิดีโอ แล้วกด 🚀 เริ่มตรวจสอบ
          </div>
          <div style="font-size:0.85rem;margin-top:8px;color:#334155;">
            รองรับ PNG, JPG, BMP, WebP, MP4, AVI, MOV, MKV
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
