import re
import os, json, random, tempfile
import numpy as np
import soundfile as sf
import gradio as gr
import librosa
import whisper
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from pathlib import Path

FONT_PATH = Path("assets/fonts/NotoSansTC-Regular.ttf")

def setup_cjk_font():
    try:
        if FONT_PATH.exists() and FONT_PATH.stat().st_size > 50_000:
            font_manager.fontManager.addfont(str(FONT_PATH))
            fp = font_manager.FontProperties(fname=str(FONT_PATH))
            matplotlib.rcParams["font.family"] = fp.get_name()
        matplotlib.rcParams["axes.unicode_minus"] = False
        return True
    except Exception as e:
        print(f"[Font] load failed: {e}")
        matplotlib.rcParams["axes.unicode_minus"] = False
        return False

CJK_OK = setup_cjk_font()

# =========================
# ⭐ NEW: 教授圖片池（本機素材，免即時生成）
# =========================
INTERVIEWER_DIR = Path("assets/interviewers")

def pick_interviewer():
    """
    回傳 (img_path:str, gender:str) gender in {"male","female"}
    你的資料夾需長這樣：
      assets/interviewers/male/*.png
      assets/interviewers/female/*.png
    """
    gender = random.choice(["male", "female"])
    folder = INTERVIEWER_DIR / gender
    imgs = []
    if folder.exists():
        imgs = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    if not imgs:
        # 找不到就退回 logo，避免整個程式炸掉
        return "assets/logo.png", "female"
    img_path = random.choice(imgs)
    return str(img_path), gender


# =========================
# 0) 情境設定（題目池 + 權重）
# =========================
SCENES = {
    "university": {
        "label": "大學系所申請面試",
        "topics": {"自我介紹", "學習計畫", "學習動機", "未來規劃", "社會關懷"},
        "weights": {"coverage": 0.35, "structure": 0.25, "semantic": 0.20, "fluency": 0.10, "pitch": 0.10},
    },
    "graduate": {
        "label": "研究所口試 / 推甄",
        "topics": {"閱讀與研究", "道德與責任", "跨域學習", "問題解決", "專題經驗"},
        "weights": {"coverage": 0.25, "structure": 0.30, "semantic": 0.30, "fluency": 0.10, "pitch": 0.05},
    },
    "hr": {
        "label": "企業 HR 初階面試",
        "topics": {"團隊合作", "溝通表達", "時間管理", "失敗經驗", "臨場表達", "問題解決"},
        "weights": {"coverage": 0.20, "structure": 0.20, "semantic": 0.15, "fluency": 0.30, "pitch": 0.15},
    },
}
SCENE_CHOICES = [(v["label"], k) for k, v in SCENES.items()]

# =========================
# 1) 題庫：讀取與抽題
# =========================
def load_questions(path="questions/professor.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]

def filter_by_scene(questions, scene_key: str):
    scene = SCENES.get(scene_key, SCENES["university"])
    topics = scene["topics"]
    out = []
    for q in questions:
        if q.get("id") == "p-001":
            out.append(q)
        elif q.get("topic") in topics:
            out.append(q)
    return out

def pick_five_with_intro(questions, intro_id="p-001"):
    intro_list = [q for q in questions if q.get("id") == intro_id]
    if not intro_list:
        raise ValueError(f"Intro question '{intro_id}' not found.")
    intro_q = intro_list[0]

    pool = [q for q in questions if q.get("id") != intro_id]
    if len(pool) < 4:
        raise ValueError("Not enough questions for this scene (need >= 4 excluding intro).")

    others = random.sample(pool, k=4)
    return [intro_q] + others

# =========================
# 2) ASR：Whisper
# =========================
_whisper = None
def load_asr():
    global _whisper
    if _whisper is None:
        size = os.environ.get("WHISPER_SIZE", "small")  # tiny/base/small/medium
        _whisper = whisper.load_model(size)
    return _whisper

def transcribe_zh(audio_path, initial_prompt=None):
    m = load_asr()
    res = m.transcribe(
        audio_path,
        language="zh",
        task="transcribe",
        initial_prompt=initial_prompt or ""
    )
    return (res.get("text") or "").strip()

# =========================
# 3) 內容分析：把 key_points 轉成「可命中的關鍵詞」
# =========================
TRANSITIONS = ["首先", "第一", "接著", "然後", "因此", "所以", "最後", "總結", "此外", "另外"]
SEMANTIC_KEYWORDS = [
    "因為","所以","例如","比如","舉例","我覺得","我學到","我發現",
    "經驗","挑戰","成果","學習","反思","困難","改善","價值","收穫","努力"
]

def _contains_any(text: str, words):
    return any(w in text for w in words)

KEYPOINT_HINTS = {
    "個人背景清楚": ["我叫", "我來自", "就讀", "高中", "科系", "背景", "經歷", "社團", "志工", "專題"],
    "申請動機明確": ["因為", "所以", "想要", "希望", "動機", "原因", "興趣", "熱忱"],
    "與科系連結具體": ["本系", "科系", "系上", "課程", "學程", "領域", "方向", "適合", "相關", "吻合"],

    "學習方向明確": ["我會", "我想", "目標", "方向", "規劃", "計畫"],
    "具體課程或主題例子": ["課程", "專題", "主題", "研究", "領域", "例如", "比如"],
    "展現自主規劃能力": ["安排", "時間", "規劃", "行事曆", "待辦", "目標", "步驟"],

    "過程與成果具體": ["過程", "結果", "成果", "完成", "提升", "成效", "學到"],
    "個人貢獻清楚": ["我負責", "我主要", "我的角色", "分工", "貢獻"],
    "問題解決能力": ["解決", "處理", "改善", "嘗試", "方法", "策略"],

    "動機真誠具體": ["因為", "契機", "開始", "興趣", "喜歡", "想深入"],
    "有例子支持": ["例如", "比如", "舉例", "參加", "看過", "做過", "經驗"],
    "與未來規劃連結": ["未來", "畢業", "目標", "想成為", "規劃", "方向"],

    "短中長期目標": ["短期", "中期", "長期", "未來", "目標"],
    "具體步驟": ["步驟", "計畫", "方法", "安排", "準備"],
    "彈性與反思": ["如果", "調整", "反思", "檢討", "改進"],

    "合作情境具體": ["一起", "團隊", "合作", "同學", "小組"],
    "溝通協調": ["溝通", "協調", "討論", "共識", "分工"],
    "反思與成長": ["學到", "收穫", "反思", "下次", "改進"],

    "問題描述清楚": ["問題", "困難", "挑戰", "卡關"],
    "行動有條理": ["首先", "接著", "然後", "最後", "步驟", "安排"],
    "反思具深度": ["反思", "學到", "收穫", "價值", "下次"],

    "思考邏輯清楚": ["我會先", "我先想", "整理", "要點", "首先"],
    "冷靜組織回答": ["先", "整理", "想一下", "重點", "再回答"],
    "舉例說明": ["例如", "比如", "舉例", "曾經"],

    "內容掌握": ["內容", "重點", "主旨", "作者", "觀點"],
    "個人見解": ["我認為", "我覺得", "我看法", "啟發", "反思"],
    "與學系連結": ["本系", "科系", "課程", "領域", "相關"],

    "具體方法": ["行事曆", "待辦", "清單", "規劃", "安排"],
    "優先級與規劃": ["優先", "重要", "緊急", "排序", "安排"],
    "自我檢核": ["檢核", "回顧", "確認", "調整"],

    "跨域連結": ["結合", "整合", "跨域", "不同領域", "連結"],
    "創意思考": ["想到", "嘗試", "創新", "發想"],
    "實作成果": ["成果", "完成", "做出", "成效"],

    "誠實描述": ["我失敗", "不如預期", "沒有做好", "當時"],
    "具體反思": ["反思", "學到", "收穫", "原因"],
    "改進策略": ["改進", "調整", "下次", "方法"],

    "清楚表達": ["我會先說", "重點", "整理", "清楚"],
    "傾聽與同理": ["傾聽", "理解", "同理", "尊重"],
    "共識策略": ["共識", "折衷", "協調", "討論"],

    "議題認知": ["我關心", "議題", "現象", "問題"],
    "行動或觀察": ["觀察", "參與", "行動", "經驗"],
    "與學系連結與展望": ["本系", "相關", "未來", "投入"],

    "倫理原則": ["誠信", "倫理", "原則", "尊重", "公平"],
    "兼顧公平與效率": ["公平", "效率", "透明", "責任", "分工"],
}

def analyze_content(transcript: str, question: dict):
    text = (transcript or "").strip()
    kps = question.get("key_points", []) or []

    if not kps:
        return {
            "coverage_pct": 0, "hits": [], "misses": [],
            "structure_score": 0.0, "semantic_score": 0.0,
            "content_score_raw": 0, "content_advice": "本題無設定 key points。"
        }

    hits, misses = [], []
    for kp in kps:
        hints = KEYPOINT_HINTS.get(kp)
        if hints:
            hit = _contains_any(text, hints)
        else:
            hit = any(token in text for token in re.split(r"[、，,。\s]+", kp) if token)
        (hits if hit else misses).append(kp)

    coverage = len(hits) / len(kps)
    structure_score = 1.0 if _contains_any(text, TRANSITIONS) else 0.0

    sentence_count = len([s for s in re.split(r"[。！？!?.]", text) if s.strip()])
    semantic_hits = sum(1 for w in SEMANTIC_KEYWORDS if w in text)
    semantic_score = min(1.0, (sentence_count / 3) * 0.4 + (semantic_hits / 5) * 0.6)

    content_score_raw = int(round(
        coverage * 100 * 0.5 +
        structure_score * 100 * 0.2 +
        semantic_score * 100 * 0.3
    ))

    adv = []
    if coverage < 0.7 and misses:
        adv.append(f"可再補充「{'、'.join(misses[:3])}」等重點。")
    if structure_score < 1.0:
        adv.append("可加入轉折詞（如『首先、接著、因此』）提升條理性。")
    if semantic_score < 0.6:
        adv.append("建議補充具體例子或反思句，讓內容更有說服力。")
    if not adv:
        adv.append("內容完整、條理清楚，具良好論述深度。")

    return {
        "coverage_pct": int(round(coverage * 100)),
        "hits": hits,
        "misses": misses,
        "structure_score": float(structure_score),
        "semantic_score": float(round(semantic_score, 2)),
        "content_score_raw": int(content_score_raw),
        "content_advice": " ".join(adv)
    }

# =========================
# 4) 語音特徵：語速/停頓/音高變化 => 分數
# =========================
def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, int(round(x))))

def analyze_audio_and_text(audio_np, sr, zh_text):
    duration = max(1e-6, len(audio_np) / float(sr))
    char_per_min = (len(zh_text) / duration) * 60.0

    if len(audio_np) < 1024:
        metrics = {
            "transcript": zh_text, "duration_sec": round(duration, 2),
            "chars_per_min": round(char_per_min, 1),
            "pauses_(>0.3s)_count": 0, "pitch_variation_CV": 0.0,
        }
        return metrics, ["錄音太短，請錄至少 10 秒再分析。"]

    frames = librosa.util.frame(audio_np, frame_length=1024, hop_length=512)
    energy = (frames**2).mean(axis=0)
    thr = energy.mean() * 0.3
    voiced = energy > thr

    pauses = 0
    i = 0
    hop_dur = 512 / float(sr)
    while i < len(voiced):
        if not voiced[i]:
            start = i
            while i < len(voiced) and not voiced[i]:
                i += 1
            dur = (i - start) * hop_dur
            if dur >= 0.3:
                pauses += 1
        i += 1

    try:
        f0 = librosa.yin(audio_np, fmin=50, fmax=400, sr=sr, frame_length=2048)
        f0 = f0[np.isfinite(f0)]
        pitch_var = float(np.std(f0) / np.mean(f0)) if len(f0) > 0 else 0.0
    except Exception:
        pitch_var = 0.0

    advice = []
    if char_per_min > 180:
        advice.append("語速偏快：關鍵句前先停 0.3–0.5 秒，讓重點更清楚。")
    elif char_per_min < 100:
        advice.append("語速偏慢：可適度加快，避免過長停頓。")
    if pauses >= 6:
        advice.append("停頓較多：先列出 2–3 個要點再回答，降低卡頓。")
    if pitch_var < 0.10:
        advice.append("語調起伏較少：可在重點句放慢、加強抑揚。")
    if not advice:
        advice.append("整體表達穩定：維持節奏與語氣，下一步可加深內容。")

    metrics = {
        "transcript": zh_text,
        "duration_sec": round(duration, 2),
        "chars_per_min": round(char_per_min, 1),
        "pauses_(>0.3s)_count": int(pauses),
        "pitch_variation_CV": round(pitch_var, 3),
    }
    return metrics, advice

def compute_fluency_and_pitch_scores(metrics: dict):
    cpm = float(metrics.get("chars_per_min", 0))
    pauses = int(metrics.get("pauses_(>0.3s)_count", 0))
    pitch = float(metrics.get("pitch_variation_CV", 0))

    if 120 <= cpm <= 160:
        speed_score = 100
    elif 100 <= cpm < 120 or 160 < cpm <= 180:
        speed_score = 75
    else:
        speed_score = 50

    if pauses <= 2:
        pause_score = 100
    elif pauses <= 5:
        pause_score = 75
    else:
        pause_score = 50

    fluency = clamp(speed_score * 0.4 + pause_score * 0.6)

    if pitch >= 0.15:
        pitch_score = 100
    elif pitch >= 0.10:
        pitch_score = 75
    else:
        pitch_score = 50

    return fluency, pitch_score, speed_score, pause_score

def get_grade(final_score: int) -> str:
    if final_score >= 90: return "A"
    if final_score >= 80: return "B"
    if final_score >= 70: return "C"
    if final_score >= 60: return "D"
    return "E"

# =========================
# 5) 雷達圖
# =========================
def render_radar_fig(radar: dict, title="Radar (0-100)"):
    labels = ["覆蓋", "條理", "深度", "流暢", "抑揚"]
    keys = ["coverage", "structure", "semantic", "fluency", "pitch"]
    values = [float(radar.get(k, 0)) for k in keys]
    values += values[:1]

    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(4.2, 4.2), dpi=120)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_title(title, pad=18, fontsize=11)
    ax.grid(True, alpha=0.3)
    return fig

# =========================
# ⭐ NEW：右上角分數徽章用的 HTML
# =========================
def score_badge_html(score, grade=None):
    s = "--" if score is None else str(int(score))
    g = "" if not grade else f"<div class='score-grade'>Grade {grade}</div>"
    return f"""
    <div class="score-badge">
        <div class="score-num">{s}</div>
        {g}
    </div>
    """

# =========================
# 6) Gradio 回呼
# =========================
HINT_VOCAB = "面試 自我介紹 教授 學習計畫 國立台中教育大學 內容科技學系 教育學程 專題報告 團隊合作"

def start_session(scene_key):
    all_qs = load_questions()
    scene_qs = filter_by_scene(all_qs, scene_key)
    selected = pick_five_with_intro(scene_qs, intro_id="p-001")

    idx = 0
    q = selected[idx]
    q_text = f"第1題：{q['prompt']}"

    # ⭐ NEW：START 就抽一張教授圖＋性別
    interviewer_img, interviewer_gender = pick_interviewer()

    # ⭐ NEW：START 時右上角先顯示 --
    badge = score_badge_html(None, None)

    return selected, idx, q_text, "", None, None, None, interviewer_img, interviewer_gender, badge

def analyze_and_next(audio, selected, idx, scene_key):
    if audio is None:
        # ⭐ NEW：沒有錄音時，徽章不變（用 gr.update()）
        return gr.JSON.update(value=None), None, "請先錄音再送出。", gr.update(), idx, selected, gr.update(), gr.update(), gr.update()

    if isinstance(audio, tuple):
        sr, y = audio
    else:
        y, sr = librosa.load(audio, sr=16000, mono=True)

    current_q = selected[idx]
    question_prompt = current_q.get("prompt", "")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        text = transcribe_zh(tmp.name, initial_prompt=HINT_VOCAB)

    metrics, prosody_adv_list = analyze_audio_and_text(y, sr, text)
    content_res = analyze_content(text, current_q)

    radar = {
        "coverage": clamp(content_res["coverage_pct"]),
        "structure": clamp(content_res["structure_score"] * 100),
        "semantic": clamp(content_res["semantic_score"] * 100),
    }

    fluency, pitch_score, speed_score, pause_score = compute_fluency_and_pitch_scores(metrics)
    radar["fluency"] = fluency
    radar["pitch"] = pitch_score

    scene = SCENES.get(scene_key, SCENES["university"])
    w = scene["weights"]

    final_score = int(round(
        radar["coverage"] * w["coverage"] +
        radar["structure"] * w["structure"] +
        radar["semantic"] * w["semantic"] +
        radar["fluency"] * w["fluency"] +
        radar["pitch"] * w["pitch"]
    ))
    grade = get_grade(final_score)

    metrics_out = {
        "scene": scene["label"],
        "question": question_prompt,
        "transcript": metrics.get("transcript", ""),
        "duration_sec": metrics.get("duration_sec"),
        "chars_per_min": metrics.get("chars_per_min"),
        "pauses_(>0.3s)_count": metrics.get("pauses_(>0.3s)_count"),
        "pitch_variation_CV": metrics.get("pitch_variation_CV"),
        "content_hits": content_res["hits"],
        "content_misses": content_res["misses"],
        "radar(0-100)": radar,
        "weights": w,
        "final_score(0-100)": final_score,
        "grade": grade,
        "debug_speed_score": speed_score,
        "debug_pause_score": pause_score,
        "debug_pitch_score": pitch_score,
    }

    summary = (
        f"【情境】{scene['label']}\n"
        f"【本題】{question_prompt}\n"
        f"【總分】{final_score} / 100（等級 {grade}）\n"
        f"【雷達】覆蓋{radar['coverage']}、條理{radar['structure']}、深度{radar['semantic']}、流暢{radar['fluency']}、抑揚{radar['pitch']}\n"
    )

    advice = (
        summary
        + "\n【表達建議】\n" + "\n".join(prosody_adv_list)
        + "\n\n【內容建議】\n" + content_res["content_advice"]
    )

    radar_fig = render_radar_fig(radar, title=f"{scene['label']} Radar (0-100)")

    idx_next = idx + 1
    if idx_next < len(selected):
        q_next = selected[idx_next]
        q_text = f"第{idx_next+1}題：{q_next['prompt']}"
    else:
        q_text = "✅ 全部 5 題完成！可以重新開始。"

    # ⭐ NEW：每題分析完就換一張教授圖＋性別
    interviewer_img, interviewer_gender = pick_interviewer()

    # ⭐ NEW：更新右上角徽章
    badge = score_badge_html(final_score, grade)

    return metrics_out, radar_fig, advice, q_text, idx_next, selected, interviewer_img, interviewer_gender, badge

# =========================
# 7) UI
# =========================
CSS = """
.gradio-container { max-width: 100% !important; }

/* ⭐ NEW：右上角總分徽章（固定在畫面右上） */
.score-badge {
  position: fixed;
  top: 18px;
  right: 18px;
  width: 92px;
  height: 92px;
  border-radius: 18px;
  border: 4px solid #7c5cff;
  background: #ffffff;
  box-shadow: 0 10px 25px rgba(0,0,0,0.10);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
.score-num {
  font-size: 44px;
  font-weight: 800;
  color: #7c5cff;
  line-height: 1.0;
}
.score-grade {
  margin-top: 4px;
  font-size: 12px;
  font-weight: 700;
  color: #7c5cff;
}
"""

with gr.Blocks(title="AI Mock Interview (Professor)", css=CSS) as demo:
    gr.Markdown("## 👩‍🏫 AI 面試練習\n固定第 1 題自我介紹，其餘隨機 4 題，共 5 題。")

    # ⭐ NEW：用 State 存性別，讓朗讀聲音可以對上圖片
    interviewer_gender_state = gr.State("female")

    # ⭐ NEW：右上角總分徽章（用 HTML 呈現）
    score_box = gr.HTML(value=score_badge_html(None, None))

    with gr.Row(equal_height=True):
        
        # ✅ 左邊：教授圖（2/3）
        with gr.Column(scale=2):
            interviewer_img = gr.Image(value="assets/logo.png", show_label=False, interactive=False, height=360)

        # ✅ 右邊：操作區（1/3）
        with gr.Column():
            scene_dd = gr.Dropdown(choices=SCENE_CHOICES, value="university", label="選擇面試情境（影響題目池＋評分權重）")
            start_btn = gr.Button("START", variant="primary")
            speak_btn = gr.Button("🔊", scale=1)
            question_box = gr.Textbox(label="題目", interactive=False, lines=3)

    # 你原本的 speechSynthesis 保留（照你的規則：不動原本結構）
    speak_btn.click(
        fn=None,
        inputs=[question_box, interviewer_gender_state],
        outputs=None,
        js=r"""
        (text, gender)=>{
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.lang = "zh-TW";
            u.rate = 1.0;

            const pickVoice = () => {
                const voices = speechSynthesis.getVoices() || [];
                if (!voices.length) return;

                const zhVoices = voices.filter(v => (v.lang || "").toLowerCase().includes("zh"));
                const pool = zhVoices.length ? zhVoices : voices;

                const isFemale = (v) => /female|woman|mei|ting|xiaomei|hui/i.test((v.name||"") + " " + (v.voiceURI||""));
                const isMale   = (v) => /male|man|wei|jun|xiaojun|kang/i.test((v.name||"") + " " + (v.voiceURI||""));

                let chosen = null;
                if (gender === "female") chosen = pool.find(isFemale) || pool[0];
                else chosen = pool.find(isMale) || pool[0];

                if (chosen) u.voice = chosen;
            };

            pickVoice();
            setTimeout(pickVoice, 200);

            speechSynthesis.speak(u);
        }
        """
    )

    audio_in = gr.Audio(sources=["microphone"], type="numpy", label="錄音（建議 60–90 秒）")
    submit_btn = gr.Button("🧪 分析並進到下一題", variant="secondary")

    with gr.Row():
        metrics_out = gr.JSON(label="分析指標（含 transcript / 命中/未命中 / 權重 / 分數）")
        radar_out = gr.Plot(label="雷達圖（0-100）")
        advice_out = gr.Textbox(label="建議（含總分/雷達/內容/表達）", lines=12)

    # ⭐ NEW：右下角等級說明（可收合、不影響 UI；不想要可整段刪）
    with gr.Accordion("等級說明（可收合）", open=False):
        gr.Markdown(
            """
| 分數 | 等級 | 說明 |
|---:|:---:|:---|
| 90–100 | A | 表現優秀：內容完整、條理清晰、表達自然 |
| 80–89 | B | 表現良好：重點大致到位，可再強化例子/反思 |
| 70–79 | C | 基本可行：內容或結構尚可，但說服力不足 |
| 60–69 | D | 需加強：重點缺漏或表達不穩、停頓較多 |
| < 60 | E | 建議重練：先準備框架與關鍵句，再重新錄音 |
            """
        )

    selected_state = gr.State([])
    idx_state = gr.State(0)

    # ⭐ NEW：START outputs 多接「面試官圖、面試官性別、右上角徽章」
    start_btn.click(
        fn=start_session,
        inputs=[scene_dd],
        outputs=[selected_state, idx_state, question_box, advice_out, metrics_out, audio_in, radar_out, interviewer_img, interviewer_gender_state, score_box]
    )

    # ⭐ NEW：SUBMIT outputs 多接「面試官圖、面試官性別、右上角徽章」（每題都換）
    submit_btn.click(
        fn=analyze_and_next,
        inputs=[audio_in, selected_state, idx_state, scene_dd],
        outputs=[metrics_out, radar_out, advice_out, question_box, idx_state, selected_state, interviewer_img, interviewer_gender_state, score_box]
    )

if __name__ == "__main__":
    demo.launch()