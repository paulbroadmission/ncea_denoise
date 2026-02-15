# 🚀 Mode 2 Quick Start (20 分鐘完整設定)

## Step 1: rclone Google Drive 授權 (5 分鐘)

```bash
# 執行 rclone 互動配置
rclone config
```

**看到提示時依序輸入：**

| 提示 | 輸入 |
|------|------|
| `n/s/q>` | `n` |
| `name>` | `gdrive` |
| `Type of storage>` | `drive` |
| `client_id>` | **按 Enter (留空)** |
| `client_secret>` | **按 Enter (留空)** |
| `scope>` | **按 Enter (預設)** |
| `root_folder_id>` | **按 Enter (留空)** |
| `service_account_file>` | **按 Enter (留空)** |
| `Edit advanced config>` | `n` |
| `Use web browser>` | `y` |

✅ **瀏覽器會自動打開** → 點擊 **Allow** → 回到終端完成

---

## Step 2: 驗證 rclone 配置 (1 分鐘)

```bash
# 測試連接
rclone lsd gdrive:
```

✅ 應該看到你 Google Drive 上的資料夾列表

---

## Step 3: 自動完成所有設定 (1 分鐘)

```bash
# 自動初始化所有 Mode 2 設定
./scripts/setup_colab_mode2.sh
```

**這會做：**
- ✅ 確認 rclone 連接
- ✅ 在 Google Drive 建立 `research-fleet/` 資料夾結構
- ✅ 更新 `workspace/logs/orchestrator_state.json` (追蹤迭代號)
- ✅ 設定好 `scripts/colab_sync.sh` (同步命令)

---

## Step 4: 本地測試代碼 (5 分鐘)

```bash
# 測試代碼能否執行 + Guardian 驗證
python3 workspace/src/main.py --mode train --dataset synthetic --epochs 2
```

✅ 應該看到：
```
✓ ALL CHECKS PASSED!  (Guardian)
Epoch 1/2 | Loss: ...
Epoch 2/2 | Loss: ...
✓ Test passed! Best accuracy: ...
```

---

## Step 5: 推送到 Google Drive (2 分鐘)

```bash
# 推送代碼、日誌、配置到 Google Drive (給 Colab 用)
./scripts/colab_sync.sh push
```

✅ 輸出應該像：
```
📤 Pushing to Google Drive...
   Iteration: 1
   Method: rclone (gdrive:research-fleet)
   ✅ src/ synced
   ✅ logs/ synced
   ✅ baselines/ synced

✅ Push complete. Now:
   1. Open Colab: https://colab.research.google.com
   2. Open notebook from GitHub: paulbroadmission/ncea_denoise
   3. Select: colab/COLAB_READY_AGENT_INTEGRATED.ipynb
   4. Runtime → Change runtime type → GPU
   5. Run All
```

---

## Step 6: 在 Colab 執行 (15-30 分鐘)

打開 Google Colab: https://colab.research.google.com

1. **File** → **Open notebook** → **GitHub**
2. 搜尋: `paulbroadmission/ncea_denoise`
3. 打開: `colab/COLAB_READY_AGENT_INTEGRATED.ipynb`
4. ⚠️ **Runtime** → **Change runtime type** → **GPU** (選 T4 或 V100)
5. 按上面的 **Run All** 按鈕執行所有 cell

**Notebook 會自動：**
- ✅ Mount Google Drive
- ✅ 從 Drive 讀取你推送的代碼
- ✅ 執行 Guardian 驗證
- ✅ 在 GPU 上訓練 (合成數據, 500 epochs)
- ✅ 保存結果回 Google Drive
- ✅ 寫入完成標記 `_colab_complete.json`

---

## Step 7: 拉回結果 (1 分鐘)

Colab 執行完成後，執行：

```bash
# 從 Google Drive 拉回結果
./scripts/colab_sync.sh pull
```

✅ 結果會存在：
```
workspace/results/iteration_001/
  ├─ test_results.json      (訓練指標)
  ├─ training_history.json  (訓練歷史)
  └─ best_model.pt          (最佳模型)
```

---

## Step 8: 查看結果 + 決策 (5 分鐘)

```bash
# 查看測試結果
cat workspace/results/iteration_001/test_results.json | jq .test_metrics
```

輸出例子：
```json
{
  "accuracy": 0.9833,
  "f1_score": 0.9834,
  "itr": 204.3,
  ...
}
```

### 根據結果決策：

| 指標 | 判定 | 下一步 |
|------|------|------|
| **Accuracy > 93%** | ✅ PASS | 完成！可以發表 |
| **90-93%** | 🔧 MINOR TUNE | 微調超參數 → Iteration 2 |
| **85-90%** | 🔨 REVISE | 重新設計 → Iteration 2 |
| **< 85%** | ❌ CRITICAL | 改變策略 |

---

## 如果要繼續迭代 (Iteration 2)

```bash
# 1. 修改代碼 (例: 調整 LAMBDA_CONSISTENCY)
vi workspace/src/config.py

# 2. 更新迭代號
vi workspace/logs/orchestrator_state.json
# "iteration": 2

# 3. 重複 Step 4-7
python3 workspace/src/main.py --dataset synthetic --epochs 2
./scripts/colab_sync.sh push
# → Colab 執行
./scripts/colab_sync.sh pull
```

---

## ⚡ 快速命令參考

```bash
# 設定
rclone config                    # 首次授權 (只需一次)
./scripts/setup_colab_mode2.sh   # 初始化 (只需一次)

# 每個 Iteration
python3 workspace/src/main.py --dataset synthetic --epochs 2
./scripts/colab_sync.sh push
# → 手動在 Colab 執行 (15-30 分鐘)
./scripts/colab_sync.sh pull

# 檢查狀態
./scripts/colab_sync.sh status   # 檢查 Colab 是否完成
./scripts/colab_sync.sh watch    # 等待 Colab 完成 (自動輪詢)
```

---

## 🆘 常見問題

### Q: rclone config 後怎麼沒打開瀏覽器？
**A:** 選擇 `y` 給 "Use web browser" 問題，它會打開。如果還是沒有，手動複製網址到瀏覽器。

### Q: Google Drive 顯示 "research-fleet not found"？
**A:** 執行 `./scripts/setup_colab_mode2.sh` 會自動建立。

### Q: Colab 訓練失敗了？
**A:** 檢查 Colab cell 的錯誤訊息。通常是 Guardian 失敗。修改本地代碼，重新 push。

### Q: 怎麼知道 Colab 執行完了？
**A:** 執行 `./scripts/colab_sync.sh status` 或等待 Colab notebook 顯示 "Done"。

---

## 準備開始？

執行第一個命令：

```bash
rclone config
```

我會一直在這裡幫你！
