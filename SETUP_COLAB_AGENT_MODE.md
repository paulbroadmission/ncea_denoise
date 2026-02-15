# 🚀 設定 Agent Fleet + Colab GPU 完整指南 (模式 2)

## 必要條件

- ✅ rclone 已安裝 (或執行 `bash scripts/setup.sh`)
- ✅ Google 帳戶 (需要存取 Google Drive)
- ✅ Colab 帳戶 (免費)

---

## Step 1: 配置 rclone 連接 Google Drive

### 方式 A: rclone interactive setup (推薦)

```bash
rclone config
```

**互動步驟：**

```
# 按 n 建立新遠端
n) New remote
name> gdrive

# 選擇 Google Drive
Type of storage> drive

# 選擇默認值（幾乎所有問題都按 Enter）
client_id> [按 Enter，使用預設]
client_secret> [按 Enter，使用預設]
scope> [按 Enter，1 = Full access]
root_folder_id> [按 Enter，留空]
service_account_file> [按 Enter，留空]

# 選擇 "n" (No) 進行自動 OAuth
Use auto config?> n

# 複製你會看到的網址，貼到瀏覽器
https://accounts.google.com/o/oauth2/auth?...

# 登入 Google 帳戶，給予 rclone 權限
# ✅ 確認

# rclone 會顯示認證碼
Go to this URL by opening your browser, and paste the code returned here:
Enter verification code> [粘貼認證碼]

# 完成設定
y) Yes this is OK
Done. Press q to quit.
```

### 方式 B: 自動設定腳本

如果你想跳過互動步驟，我可以幫你寫一個自動配置腳本。

---

## Step 2: 驗證 rclone 配置

```bash
# 列出 Google Drive 的資料夾
rclone lsd gdrive:

# 檢查配置
rclone config show
```

應該會看到：
```
[gdrive]
type = drive
client_id = ...
client_secret = ...
token = {...認證令牌...}
```

---

## Step 3: 設定 orchestrator_state.json (開始追蹤迭代)

```bash
cat workspace/logs/orchestrator_state.json
```

更新為你的項目設定：

```json
{
  "project_name": "Neural Conditional Ensemble Averaging",
  "research_domain": "SSVEP-BCI",
  "user_inspiration": "Improve SSVEP classification with consistency loss",
  "target_venue": "IEEE TMI / NeurIPS",
  "success_criteria": "Exceed Li et al. 2024 by 2%",
  "iteration": 1,
  "max_iterations": 10,
  "phase": "implementation",
  "status": "ready_for_colab",
  "sota_baseline": {
    "method_name": "Li et al. 2024 TRCA+CNN",
    "primary_metric_name": "accuracy",
    "primary_metric_value": 0.92,
    "dataset": "BETA"
  },
  "our_best_result": {
    "iteration": 0,
    "primary_metric_value": null,
    "all_metrics": {}
  },
  "gap_to_sota": null,
  "decision_history": [],
  "guardian_reports": [],
  "reviewer_scores": [],
  "created_at": "2026-02-15",
  "last_updated": "2026-02-15"
}
```

---

## Step 4: 初始化 Colab 專案目錄 (Google Drive 端)

```bash
./scripts/colab_sync.sh init
```

這會：
1. ✅ 在 Google Drive 建立 `research-fleet/` 資料夾
2. ✅ 建立子目錄 (src/, baselines/, results/, logs/)
3. ✅ 驗證 rclone 連接

---

## Step 5: 執行完整 Iteration Cycle

### **Iteration 1 開始**

```bash
# 1️⃣ 本地開發與測試
echo "編輯 workspace/src/*.py 檔案..."
python3 workspace/src/main.py --mode train --dataset synthetic --epochs 50
# 應該看到 Guardian PASS ✅

# 2️⃣ 推送到 Google Drive（給 Colab 用）
./scripts/colab_sync.sh push
```

**輸出應該像這樣：**
```
📤 Pushing to Google Drive...
   Iteration: 1
   Method: rclone (gdrive:research-fleet)
   src/ synced
   logs/ synced
   baselines/ synced

✅ Push complete. Now:
   1. Open Colab: https://colab.research.google.com
   2. Open notebook from GitHub: paulbroadmission/ncea_denoise
   3. Select: colab/COLAB_READY_AGENT_INTEGRATED.ipynb
   4. Enable GPU (Runtime → Change runtime type → GPU)
   5. Run All cells
```

### **在 Colab 執行**

1. 打開 Google Colab: https://colab.research.google.com
2. File → Open notebook from GitHub
3. 搜尋: `paulbroadmission/ncea_denoise`
4. 選擇: `colab/COLAB_READY_AGENT_INTEGRATED.ipynb`
5. **非常重要**: Runtime → Change runtime type → GPU (select T4 or V100)
6. **Run All** (Shift+Enter 逐個執行，或按上面的 "Run All" 按鈕)

**Notebook 會自動：**
- ✅ Mount Google Drive
- ✅ 從 Drive 讀取 src/ (你推送的代碼)
- ✅ 運行 Guardian 驗證
- ✅ 在 GPU 上訓練
- ✅ 保存結果到 Google Drive
- ✅ 寫入 `_colab_complete.json` 完成標記

### **回到本地拉取結果**

```bash
# 3️⃣ 拉取結果
./scripts/colab_sync.sh pull
```

**輸出：**
```
📥 Pulling results from Google Drive...
   Iteration: 1
   src: ....
✅ Results pulled to workspace/results/iteration_001/
   - test_results.json (訓練指標)
   - training_history.json (訓練歷史)
   - best_model.pt (最佳模型)
```

### **Watchdog 自動審計（Close-Loop）**

現在你可以執行 Watchdog 來自動分析結果：

```bash
# 4️⃣ Watchdog 分析並評分
cd workspace && python ../path/to/watchdog.py --iteration 1
```

**Watchdog 會生成：**
```
workspace/logs/
  ├─ results_audit_iteration_001.json (詳細審計)
  └─ watchdog_verdict_iteration_001.json (評分 + 決策)
```

**watchdog_verdict.json 內容例子：**
```json
{
  "iteration": 1,
  "timestamp": "2026-02-15T...",
  "reviewer_scores": {
    "domain_master": 7,
    "dl_master": 8,
    "ieee_reviewer": 7,
    "average": 7.3
  },
  "verdict": "MINOR REVISE",
  "action_items": [
    {
      "priority": "WARNING",
      "description": "Consistency loss weight could be tuned higher"
    }
  ]
}
```

---

## 決策與迭代

根據 Watchdog 的評分決策：

| 評分 | 判定 | 行動 |
|------|------|------|
| 9-10 | PASS | ✅ 完成！論文可以發表 |
| 7-8 | MINOR REVISE | 🔧 修改參數，Iteration 2 |
| 5-6 | MAJOR REVISE | 🔨 重新設計，Iteration 2 |
| 3-4 | PIVOT | 🔄 改變策略，新方向 |
| 1-2 | REJECT | ❌ 停止 |

### 如果得分 7-8 (MINOR REVISE)

```bash
# 根據 action_items 修改代碼
vi workspace/src/config.py
# 例: LAMBDA_CONSISTENCY = 0.2 (原本是 0.1)

# 更新 iteration
# 編輯 workspace/logs/orchestrator_state.json:
# "iteration": 2

# 重複 Iteration Cycle
./scripts/colab_sync.sh push
# → Colab 執行
./scripts/colab_sync.sh pull
# → Watchdog 再評一次
```

---

## 故障排除

### 問題 1: rclone 找不到 `gdrive` 遠端

```bash
# 重新配置
rclone config
# 確保遠端名稱是 "gdrive"

# 驗證
rclone lsd gdrive:
```

### 問題 2: Google Drive 認證過期

```bash
# 重新授權
rclone config reconnect gdrive
```

### 問題 3: Colab 訓練失敗

1. 檢查 Colab 輸出日誌
2. 通常是 Guardian 失敗 (看 Colab 的第一個訓練 cell 的錯誤)
3. 修復本地代碼，重新 push

### 問題 4: `_colab_complete.json` 沒有出現

```bash
# 檢查 Colab 是否真的執行完
./scripts/colab_sync.sh status

# 如果還沒完成，等待 Colab notebook 全部執行完
# 可能需要 5-30 分鐘，取決於訓練大小和 epoch 數
```

---

## 完整命令參考

```bash
# === 初始設定 ===
rclone config                          # 配置 Google Drive
./scripts/colab_sync.sh init           # 初始化 Drive 目錄

# === 每個 Iteration ===
./scripts/colab_sync.sh push           # 推送代碼
# → 打開 Colab, 執行 notebook
./scripts/colab_sync.sh pull           # 拉取結果
./scripts/colab_sync.sh status         # 檢查完成狀態

# === 調試 ===
./scripts/colab_sync.sh watch          # 等待 Colab 完成 (自動輪詢)
```

---

## 下一步

1. ✅ 執行 `rclone config` (只需一次)
2. ✅ 執行 `./scripts/colab_sync.sh init`
3. ✅ 確保本地代碼能運行
4. ✅ 執行 `./scripts/colab_sync.sh push`
5. ✅ 打開 Colab notebook 並運行
6. ✅ 執行 `./scripts/colab_sync.sh pull`
7. ✅ (將來) 執行 Watchdog 並根據評分迭代

**準備好開始？**

現在執行：
```bash
rclone config
```

我會在旁邊幫你。
