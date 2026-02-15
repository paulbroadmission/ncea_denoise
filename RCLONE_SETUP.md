# 🔐 rclone Google Drive Authorization (5 分鐘)

## 方式 A: 最簡單 (推薦)

執行一條命令，會自動打開瀏覽器：

```bash
rclone config
```

### 互動步驟（複製粘貼）

當看到提示時，按照以下輸入：

```
No remotes found - make a new one
n) New remote
s) Set configuration password
q) Quit config
n/s/q> n

name> gdrive

Type of storage (enter string value) []: drive

Google Application Client Id
Leave blank normally.
client_id> [按 Enter - 留空]

Google Application Client Secret
Leave blank normally.
client_secret> [按 Enter - 留空]

OAuth Scope
scope> [按 Enter - 預設選項 1]

ID of the root folder
root_folder_id> [按 Enter - 留空]

Service Account Credentials JSON file path
service_account_file> [按 Enter - 留空]

Edit advanced config (y/n)
y) Yes
n) No
y/n> n

Use web browser to automatically authenticate rclone with remote?
y) Yes
n) No
y/n> y
```

此時會自動打開瀏覽器 → 點擊 **Allow** → 回到終端

```
Success! Config saved in: /Users/paul/.config/rclone/rclone.conf
```

✅ 完成！

---

## 方式 B: 如果瀏覽器沒打開 (備方案)

手動配置後會要求輸入認證碼：

```
Go to this URL by opening your browser, and paste the code returned here:
https://accounts.google.com/o/oauth2/auth?access_type=offline&...

Enter verification code>
```

複製網址到瀏覽器 → 點擊 **Allow** → 複製認證碼 → 貼到終端 → Enter

---

## 驗證設定

```bash
# 測試連接
rclone lsd gdrive:

# 應該會看到你 Google Drive 上的資料夾列表
```

✅ 看到資料夾列表 = 成功！

---

## 下一步

認證完成後，執行：

```bash
./scripts/setup_colab_mode2.sh
```

這會自動完成剩下的所有設定。
