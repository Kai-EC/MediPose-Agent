# 🦾 MediPose-Agent

**基於邊緣視覺姿態估計與在地大語言模型之居家復健生物力學分析決策系統**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-113%20passed-brightgreen?logo=pytest)](./tests/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-orange)](./README.md)

> **一鍵啟動，完全在地。** 整合電腦視覺、生物力學演算法與在地大語言模型，自動生成具備臨床可追溯性的物理治療評估報告，患者資料全程不離裝置。

---

## 📋 目錄

- [系統概覽](#-系統概覽)
- [核心技術亮點](#-核心技術亮點)
- [系統架構](#-系統架構)
- [效能基準](#-效能基準)
- [專案結構](#-專案結構)
- [環境建置](#-環境建置)
- [快速開始](#-快速開始)
- [各階段使用說明](#-各階段使用說明)
- [技術選型比較](#-技術選型比較)
- [已知限制](#-已知限制)
- [未來研究方向](#-未來研究方向)
- [參考文獻](#-參考文獻)

---

## 🔍 系統概覽

### 臨床痛點

| 問題 | 現況 | 本系統解法 |
|---|---|---|
| 三維動作分析造價高昂 | 數百萬元，僅限醫院場域 | 單一消費級顯卡即可運行 |
| 居家復健缺乏即時回饋 | 患者易因代償姿勢造成二次傷害 | 自動偵測代償徵兆並警示 |
| 雲端 AI 醫療的隱私風險 | 影像須上傳第三方雲端 | 100% 在地推論，資料不離機 |
| AI 報告可信度難以驗證 | 業界多以定性宣稱「消除幻覺」 | 程式化引用可追溯性驗證機制 |

### 系統定位

本系統定位為**臨床決策支援工具（CDSS）**，不取代醫師或物理治療師的臨床判斷。所有 AI 診斷建議均標注引用來源，並計算可追溯覆蓋率，供使用者判斷報告可信度。

---

## ✨ 核心技術亮點

### 1. 視角自適應幾何補償（自研演算法）

一般系統要求固定側面 90° 拍攝，使用不便。本系統開發**透視縮短比例估算視角偏轉角 φ** 的一階補償方法：

```
cos(φ) ≈ L' / L_ref
θ_corrected = 180° - (180° - θ_raw) / cos(φ)
```

允許相機自由擺放，超過 60° 偏轉角自動標記不可信，並以**雙骨段分歧旗標**偵測模型假設失效，誠實揭露量測侷限而非靜默輸出失真數據。

### 2. 程式化引用可追溯性驗證

取代「RAG 消除幻覺」的不嚴謹宣稱，改以正規表達式程式化驗證每個 `[來源: chunk_id]` 是否確實存在於本次檢索結果，計算**可追溯覆蓋率（Traceability Score）**，將 LLM 輸出品質轉為可量化指標。

### 3. 四項量化生物力學指標

| 指標 | 方法 | 臨床意義 |
|---|---|---|
| **Max ROM** | 濾波角度序列峰對峰值 | 膝關節活動範圍，術後恢復核心指標 |
| **LDLJ Jerk Score** | Hogan & Sternad (2009) | 神經肌肉控制品質，反映動作熟練度 |
| **Robinson SI** | `(X_L - X_R) / [0.5(X_L + X_R)] × 100%` | 左右肢體功能對稱性 |
| **代償偵測** | 角速度超標 + SI 超標規則 | 下肢代償徵兆警示 |

### 4. 113 項測試，完整 TDD 覆蓋

```
Phase 1 視覺底座        14 項
Phase 2 生物力學        47 項（含端到端整合測試）
Phase 3 RAG + LLM      28 項（無需 Ollama 即可完整測試）
Phase 4 Gradio 前端     15 項
Phase 5 基準測試         9 項
─────────────────────────
總計                   113 項全數通過
```

---

## 🏗️ 系統架構

```
┌──────────────────────────────────────────────────────────────┐
│                     MediPose-Agent                           │
├──────────────┬───────────────┬─────────────┬────────────────┤
│  Phase 1     │   Phase 2     │  Phase 3    │   Phase 4      │
│  視覺感知層   │  生物力學層    │  決策層      │  前端呈現層    │
│              │               │             │                │
│ YOLOv11-Pose │ 角度計算       │ FAISS 索引  │ Gradio Blocks  │
│ 關節座標提取  │ 視角補償       │ RAG 檢索    │ Plotly 圖表    │
│ 信心值過濾   │ 巴特沃斯濾波     │ LLM 推論    │ 串流輸出       │
│ 非同步擷取   │ 特徵聚合        │ 引用驗證    │ 三種執行模式     │
└──────────────┴───────────────┴─────────────┴────────────────┘
         ↓               ↓              ↓
    joint_seq.json  analysis.json  report.json
```

### 資料流

```
影像輸入
  └→ YOLOv11-Pose 推論 → 關節座標 (frame_id, x, y, confidence)
       └→ 信心值過濾 → 多人擇優 → JointSequenceBuffer → session.json
            └→ 向量幾何角度計算 → 視角自適應補償 → 巴特沃斯濾波
                 └→ 特徵聚合 (ROM / Jerk / SI / 代償) → analysis.json
                      └→ RAG 檢索 (FAISS + TF-IDF / SentenceTransformer)
                           └→ Prompt 組裝 → Ollama LLM → 引用驗證
                                └→ 可追溯性報告 → Gradio 串流顯示
```

---

## ⚡ 效能基準

> 測試環境：Python 3.12 / Linux，30 次重複取中位數

| 模組 | 中位數延遲 | 備註 |
|---|---|---|
| 關節角度計算（180 幀） | 4.46 ms | 等效 ~40,000 FPS 處理速度 |
| 視角補償計算（180 幀） | 9.56 ms | |
| 巴特沃斯濾波（180 幀） | 0.53 ms | |
| 特徵聚合（四項指標） | 0.29 ms | |
| **Phase 2 完整流程** | **3.12 ms** | **吞吐量 ~57,000 幀/秒** |
| RAG 向量檢索（Top-3） | 0.49 ms | |
| Prompt 組裝 + 引用驗證 | 0.03 ms | |
| **Phase 2+3 端對端** | **6.60 ms** | **不含 Phase 1 視覺推論** |

> Phase 1（YOLOv11-Pose）的 FPS 數據需在實際環境量測，目前 benchmark 僅覆蓋純演算法層。

自行執行基準測試：

```bash
python -m medipose_agent.benchmark
python -m medipose_agent.benchmark --output results/benchmark.json
```

---

## 📁 專案結構

```
medipose_agent/
├── medipose_agent/                  # 主套件
│   ├── config.py                    # 集中設定（無 Magic Number）
│   ├── capture.py                   # Phase 1：非同步影像擷取
│   ├── pose_estimator.py            # Phase 1：YOLOv11-Pose 封裝
│   ├── joint_extractor.py           # Phase 1：下肢關節提取與信心過濾
│   ├── buffer.py                    # Phase 1：關節序列緩衝區
│   ├── main.py                      # Phase 1：主流程（CLI）
│   ├── biomechanics/                # Phase 2：生物力學計算層
│   │   ├── geometry.py              #   向量幾何關節角度計算
│   │   ├── correction.py            #   視角自適應幾何補償（自研）
│   │   ├── filtering.py             #   巴特沃斯低通濾波器
│   │   ├── metrics.py               #   特徵聚合（ROM/Jerk/SI/代償）
│   │   └── calibration.py           #   像素-公分空間校正（A4 紙）
│   ├── io_utils.py                  # Phase 2：讀取 Phase 1 JSON
│   ├── pipeline.py                  # Phase 2：主流程（CLI）
│   ├── rag/                         # Phase 3：RAG 檢索層
│   │   ├── document_loader.py       #   中文句界優先文件切片
│   │   ├── embeddings.py            #   TF-IDF / SentenceTransformer 雙後端
│   │   ├── vector_store.py          #   FAISS 向量索引封裝
│   │   └── retriever.py             #   統一檢索介面
│   ├── llm/                         # Phase 3：LLM 推論層
│   │   ├── client.py                #   OllamaClient / MockLLMClient
│   │   ├── prompt_builder.py        #   System/User Prompt 組裝
│   │   └── citation_checker.py      #   引用可追溯性驗證
│   ├── report_pipeline.py           # Phase 3：報告生成主流程（CLI）
│   ├── ui/                          # Phase 4：Gradio 前端
│   │   ├── demo_data.py             #   合成展示資料生成
│   │   ├── charts.py                #   Plotly 圖表 + HTML 指標看板
│   │   └── backend.py               #   yield 生成器串接三個階段
│   ├── app.py                       # Phase 4：Gradio 主介面
│   └── benchmark.py                 # Phase 5：效能基準測試
├── tests/                           # 單元測試 + 整合測試（113 項）
├── docs/                            # 推甄備審文件
│   ├── 研究計畫書.md
│   ├── 備審亮點摘要.md
│   └── 面試題庫.md
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 🔧 環境建置

### 系統需求

| 項目 | 最低需求 | 建議 |
|---|---|---|
| Python | 3.10+ | 3.10 |
| 作業系統 | Windows / macOS / Linux | Ubuntu 22.04 |
| RAM | 8 GB | 16 GB |
| 儲存空間 | 10 GB（含模型） | 20 GB |

> **Phase 1（YOLO 推論）與 Phase 3（Ollama LLM）** 需要 GPU 以達到即時效能。Demo 模式與 Phase 2/3 的純演算法部分可完全在 CPU 上執行。

### Step 1：建立虛擬環境

```bash
conda create -n medipose python=3.10 -y
conda activate medipose
```

### Step 2：安裝 PyTorch

```bash
# CUDA 12.1（RTX 30/40 系列）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 僅 CPU（Demo 模式）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3：安裝相依套件

```bash
pip install -r requirements.txt
```

### Step 4：下載 YOLOv11-Pose 模型（Phase 1 需要）

```bash
mkdir -p models
python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"
mv yolo11m-pose.pt models/
```

### Step 5：安裝 Ollama（Phase 3 完整模式需要）

```bash
# 安裝 Ollama（詳見 https://ollama.com）
curl -fsSL https://ollama.com/install.sh | sh

# 下載繁中優化模型
ollama pull llama3-taiwan-8b-instruct

# 啟動服務
ollama serve
```

### 驗證安裝

```bash
pytest -v
# 預期輸出：113 passed
```

---

## 🚀 快速開始

### Demo 模式（不需要任何模型或外部服務）

```bash
python -m medipose_agent.app --mode demo
```

瀏覽器前往 `http://localhost:7860`，點擊「🚀 開始分析」，你會依序看到：

1. **角度曲線圖**逐幀繪製（模擬 Phase 1 關節座標提取）
2. **量化指標看板**出現（Max ROM / 對稱指數 / Jerk Score）
3. **AI 診斷報告**逐字串流輸出（含引用可追溯覆蓋率）

---

## 📖 各階段使用說明

### Phase 1：視覺底座

```bash
# 從影片檔（推薦用於測試）
python -m medipose_agent.main --source path/to/video.mp4

# 無頭環境（不開預覽視窗）
python -m medipose_agent.main --source path/to/video.mp4 --no-display

# 指定輸出檔名
python -m medipose_agent.main --source path/to/video.mp4 --export-name session_001.json
```

按 `q` 或 `Ctrl+C` 結束，關節序列自動匯出至 `output/joint_sequences/`。

**輸出格式：**

```json
{
  "frame_count": 180,
  "quality_stats": {
    "valid_ratio": 0.97,
    "multi_person_ratio": 0.0
  },
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.033,
      "person_detected": true,
      "joints": {
        "left_hip":   [320.5, 210.2, 0.93],
        "left_knee":  [318.1, 410.7, 0.91],
        "left_ankle": [315.3, 600.1, 0.88],
        "right_hip":  [420.5, 208.9, 0.94],
        "right_knee": [418.2, 409.3, 0.92],
        "right_ankle":[414.8, 598.7, 0.90]
      }
    }
  ]
}
```

> `valid_ratio` 低於 0.8 時，建議調整光線或拍攝距離。

---

### Phase 2：生物力學分析

```bash
python -m medipose_agent.pipeline \
  --input output/joint_sequences/session_xxx.json
```

**輸出格式：**

```json
{
  "metrics": {
    "max_rom_left_deg": 85.3,
    "max_rom_right_deg": 94.7,
    "jerk_score_left": -7.56,
    "jerk_score_right": -7.21,
    "symmetry_index_pct": 10.6,
    "velocity_anomaly_count_left": 0,
    "velocity_anomaly_count_right": 0,
    "compensation_suspected": false
  }
}
```

---

### Phase 3：AI 報告生成

```bash
# MockLLM（無需 Ollama，適合驗證流程）
python -m medipose_agent.report_pipeline \
  --analysis output/joint_sequences/session_xxx_analysis.json \
  --corpus data/clinical_guidelines.txt \
  --mock-llm

# 完整模式（需 Ollama 服務已啟動）
python -m medipose_agent.report_pipeline \
  --analysis output/joint_sequences/session_xxx_analysis.json \
  --corpus data/clinical_guidelines.txt
```

**臨床指引語料庫**：純文字 UTF-8，可貼入台灣物理治療學會指引或醫院復健科衛教資料。

**報告輸出範例：**

```
根據臨床復健指引，患者左膝最大屈曲角度為 85.3 度，
尚未達到術後第六週建議目標（120 度以上）。[來源: guideline_a3f2]

左右肢體對稱指數為 +10.6%，在臨床可接受範圍（±15%）內。[來源: guideline_c7d4]

建議：增加股四頭肌離心訓練頻率。[來源: guideline_b8f1]

---
📊 引用可追溯覆蓋率：100.0%　有效引用：3/3
```

---

### Phase 4：Gradio 一體化介面

```bash
# Demo 模式
python -m medipose_agent.app --mode demo

# 分析模式
python -m medipose_agent.app \
  --mode analysis_only \
  --analysis output/joint_sequences/session_xxx_analysis.json

# 指定埠號 / 公開分享連結
python -m medipose_agent.app --mode demo --port 8080 --share
```

**介面佈局：**

```
┌─────────────────────┬──────────────────────────────────────┐
│  📹 感知層輸入       │  📊 力學分析 × AI 決策              │
│                     │                                      │
│  [影片上傳]          │  [Plotly 角度曲線]    [量化指標]     │
│  [analysis.json]    │  左膝 ── 右膝         Max ROM        │
│  [臨床指引文本]      │                       對稱指數       │
│  [LLM Backend]      │                       Jerk Score    │
│  [🚀 開始分析]       │  ─────────────────────────────────  │
│                     │  🤖 AI 物理治療評估報告              │
│  狀態：分析中...     │  （逐字串流輸出）                    │
└─────────────────────┴──────────────────────────────────────┘
```

---

## 🔬 技術選型比較

### 視覺感知

| | YOLOv11-Pose ✅ | MediaPipe Pose | OpenPose |
|---|---|---|---|
| 推論速度 | **極快（>100 FPS）** | 快（~60 FPS） | 慢（~15 FPS） |
| 複雜背景穩定性 | **極高** | 中等 | 高 |
| 底層座標可存取性 | **完整開放** | 封裝，難客製 | 開放 |
| 自研演算法整合 | **易於整合** | 困難 | 可行 |

### 語言模型

| | 在地 Llama-3-Taiwan ✅ | 雲端 GPT-4 API |
|---|---|---|
| 患者隱私 | **100% 安全** | 需上傳第三方 |
| 推論成本 | **零元（本機）** | 依用量計費 |
| 離線可用 | **✓** | ✗ |
| 繁中品質 | 良好（台灣語料微調） | 優秀 |

### Embedding 後端（可無縫切換）

| | TF-IDF（字元 n-gram）✅ | SentenceTransformer |
|---|---|---|
| 部署依賴 | **零（純本機）** | 需下載預訓練模型 |
| 測試可行性 | **完整** | 需網路 |
| 語意理解 | 關鍵詞匹配 | 語意相似度 |

---

## ⚠️ 已知限制

本系統明確揭露以下限制：

| 限制 | 影響 | 緩解策略 |
|---|---|---|
| 視角補償為一階近似，僅適用單軸旋轉 | 複合多軸旋轉時誤差增大 | 超過 60° 偏轉角直接標記不可信 |
| 無法完整還原三維資訊 | 量測角度與真實值存在誤差 | 揭露信賴區間，規劃量角器驗證 |
| 代償偵測僅涵蓋下肢六關節 | 無法偵測軀幹代償 | 標記為「下肢角速度異常旗標」 |
| Gradio generator 為單執行緒 | 多用戶同時使用可能阻塞 | 設計情境為單一患者居家使用 |

---

## 🔭 未來研究方向

1. **縱向追蹤分析**：建立患者資料庫，追蹤 ROM 與 Jerk Score 的週期性進步趨勢
2. **單眼深度估測整合**：整合 MiDaS / Depth Anything 取代一階幾何補償
3. **多模態感測器融合**：整合 IMU 資料解決遮擋問題
4. **聯邦學習**：在保持隱私的前提下，讓多家診所共同優化模型
5. **量角器準確度驗證**：完成 Bland-Altman 分析，提供可引用的準確度指標

---

## 📚 參考文獻

1. Hogan, N., & Sternad, D. (2009). Sensitivity of smoothness measures to movement duration, amplitude, and arrests. *Journal of Motor Behavior*, 41(6), 529–534.
2. Robinson, R. O., Herzog, W., & Nigg, B. M. (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. *Journal of Manipulative and Physiological Therapeutics*, 10(4), 172–176.
3. Winter, D. A. (2009). *Biomechanics and Motor Control of Human Movement* (4th ed.). Wiley.
4. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
5. Jocher, G., et al. (2024). Ultralytics YOLO11. https://github.com/ultralytics/ultralytics

---

## 📄 授權

MIT License

---

## 🏫 專案背景

本專案為台科大醫學工程研究所推甄備審作品，旨在展示資工背景申請者對生醫工程跨域整合的技術深度。歡迎任何學術或研究用途的引用與延伸。

---

<div align="center">

**⚠️ 免責聲明**

本系統為臨床決策支援工具（CDSS），不取代醫師或物理治療師的臨床判斷。
所有 AI 報告均標注引用來源，使用者應結合專業醫療評估做出最終判斷。

</div>
