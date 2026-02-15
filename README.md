# 🔬 Autonomous Research Agent Fleet

**自主探索 → 策略矩陣 → LaTeX 撰寫 → GPU 實作 → Closed-Loop 迭代 → 超越 SOTA**

基於 Claude Code 的 **Agents + Skills** 雙層架構自主學術研究系統。

---

## 📋 系統需求

| 項目 | 需求 |
|------|------|
| Claude Code | ≥ v1.0（[安裝指南](https://code.claude.com/docs/en/overview)） |
| Anthropic API | 需要有效 API key（建議 Opus 額度） |
| Python | ≥ 3.9 + PyTorch ≥ 2.0 |
| LaTeX | texlive 或 MiKTeX |
| Colab sync | Google Drive Desktop 或 rclone |

---

## 🚀 安裝（3 步驟）

```bash
# 1. 取得專案
git clone <repo-url> auto-research-fleet && cd auto-research-fleet

# 2. 執行設定（檢查環境 + 初始化 Drive）
chmod +x scripts/*.sh && ./scripts/setup.sh

# 3. 啟動
claude
```

進入 Claude Code 後，告訴 orchestrator 你的研究方向即可。

---

## 🏗️ 雙層架構：Agents + Skills

### 設計原則

| | Agent (`.claude/agents/`) | Skill (`.claude/skills/`) |
|---|---|---|
| 回答 | **WHO** — 誰來做 | **HOW** — 怎麼做 |
| 內容 | 角色定義 + 決策邏輯 + 交付物 | 可複用知識 + 模板 + 腳本 |
| 大小 | ~50-80 行（精簡） | 依知識量而定 |
| 可複用 | 專案專屬 | 跨專案通用 |

### 7 個 Agent

```
.claude/agents/
├── orchestrator.md          🎯 總指揮 (Opus)
├── literature-explorer.md   📚 文獻探索 (Sonnet) → uses: literature-search
├── strategy-matrix.md       🎲 策略矩陣 (Opus) → uses: game-theory
├── theory-writer.md         ✍️  理論撰寫 (Opus) → uses: ieee-latex
├── implementer.md           ⚙️  程式實作 (Sonnet) → uses: colab-gpu
├── benchmark-comparator.md  📊 基準比較 (Sonnet) → uses: colab-gpu
└── watchdog.md              🛡️  品質閘門 (Opus) → uses: theory-audit, code-audit, results-audit
```

### 7 個 Skill

```
.claude/skills/
├── ieee-latex/              📄 IEEE LaTeX 慣例 + 模板 + 編譯腳本
├── game-theory/             🎲 逆向歸納框架 + 策略卡模板
├── literature-search/       🔍 學術搜索方法論 + 提取模板
├── theory-audit/            ✅ 數學正確性驗證清單
├── code-audit/              ✅ LaTeX↔Code 一致性檢查
├── results-audit/           ✅ 假資料偵測 + 統計檢驗
└── colab-gpu/               ☁️  Colab 同步 + 執行指南
```

### 為什麼這樣拆？

**舊設計的問題**（感謝 Grok 指出）：
- Watchdog 一個 agent 管四件事 → 單點瓶頸
- Agent `.md` 裡塞滿了 IEEE 模板、game tree 公式 → 不可複用、難維護
- 沒用 skills 結構 → 不符合 Claude Code 最佳實踐

**新設計的改進**：
- Watchdog 變成**聚合器** — 呼叫 3 個 audit skills 做細節檢查，自己只做專家判斷
- 知識模板抽到 skills — 任何 agent 都能載入，可獨立測試
- Agent 精簡到 50-80 行 — 減少 context 消耗

---

## 🎲 策略矩陣（逆向歸納）

在寫任何一行程式碼之前：

1. **維度提取**：從文獻中拆出獨立策略維度
2. **組合篩選**：全排列 → 可行 → 有新穎性
3. **逆向歸納**：從「超越 SOTA」反推最佳路徑
4. **敏感度分析**：識別「調錯就失敗」的關鍵參數
5. **輸出**：主策略 + 備選策略 + 預期範圍

---

## 🔄 迭代流程

```
使用者靈感
    ↓
📚 文獻探索 (literature-search skill)
    ↓
🎲 策略矩陣 (game-theory skill) → 主策略 + 備選
    ↓
✍️ 理論 & LaTeX (ieee-latex skill) → CONFIG-SYNC 標記
    ↓
🛡️ 理論審計 (theory-audit skill) ← GATE
    ↓
⚙️ 程式實作 (colab-gpu skill) → Push to Drive → Colab GPU → Pull results
    ↓
🛡️ 程式審計 (code-audit skill) ← GATE
    ↓
📊 Benchmark (colab-gpu skill)
    ↓
🛡️ 結果審計 (results-audit skill) + Reviewer 評分 ← GATE
    ↓
🎯 決策：ACCEPT / TUNE / REVISE / PIVOT / RECOMPUTE
```

---

## ☁️ Colab GPU 整合

```bash
./scripts/colab_sync.sh init     # 首次：建 Drive 資料夾 + 上傳 notebook
./scripts/colab_sync.sh push     # 推 src/ 到 Drive
# → Colab 按 Run All
./scripts/colab_sync.sh watch    # 等完成
./scripts/colab_sync.sh pull     # 拉回 results/
```

---

## 💰 預估成本

| 項目 | 每次迭代 | 完整研究 (5-10 輪) |
|------|---------|-------------------|
| API 費用 | ~$15-40 | ~$100-400 |
| GPU 計算 | 依領域 | 依領域 |

Opus (orchestrator, strategy-matrix, theory-writer, watchdog) 較貴。
Sonnet (literature-explorer, implementer, benchmark-comparator) 較便宜。

---

## ⚠️ 注意事項

1. **Agent Teams 仍是實驗功能** — 系統同時用檔案通訊作為 fallback
2. **GPU 需另外準備** — Colab 免費版有用量限制
3. **人工審查仍必要** — 系統是輔助工具，最終論文需人工確認
4. **Skills 可跨專案** — 複製到 `~/.claude/skills/` 即可全域使用

---

## 📄 License

MIT
# ncea_denoise
