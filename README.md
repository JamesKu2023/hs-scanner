# hs-scanner：头肩顶做空扫描（只通知，不下单）

这个仓库在 GitHub Actions 上每 15 分钟扫描一次币安、欧易各 30 个合约的头肩顶形态，扫到「可入场」或「已破位·等回踩」就推送到 Grok Bot 的通知例程。**它不下单，也不需要任何交易所 API Key。**

## 一、你要做的设置（只做一次）

### 1. 加两个加密 Secret
1. 打开本仓库 → 上方 **Settings** → 左栏 **Secrets and variables** → **Actions**。
2. 点绿色 **New repository secret**：
   - Name 填 `HS_WEBHOOK_URL`，Secret 填 Grok Bot 通知例程里的 **Webhook URL**。
   - 再点一次 **New repository secret**，Name 填 `HS_WEBHOOK_KEY`，Secret 填例程里的 **Webhook key**。
3. 保存后值不会再显示，也不会出现在日志里。

没加 Secret 时，扫描和记账照常跑，只是不推送，日志里会写「HS_WEBHOOK_KEY 未配置…跳过推送」，工作流不会报错。

### 2. 确认 Actions 已开启
仓库上方 **Actions** 页签 → 如果提示要启用，点 **I understand my workflows, go ahead and enable them**。
想马上跑一次：Actions → 左栏 **hs-scan** → 右侧 **Run workflow**。

## 二、运行方式
- 定时：每小时第 2、17、32、47 分（GitHub 的定时常会晚 5–15 分钟，偶尔跳过一次；下一次会自动补扫）。
- 每次：扫 15m；有新的 1H 收盘就扫 1H；北京时间 0/4/8/12/16/20 点的 4H 收盘后扫 4H；嵌套小头肩一起扫。
- 夜间（北京时间 00:00–08:59）照常推送，推送里带 `night=true`。
- 状态（已推送去重、纸面账本、熔断状态、最后扫描的K线）存在 `state/` 目录，每次运行后自动提交回仓库。
- 推送过的信号的图表备份在 `charts/日期/`。

## 三、数据源说明（重要）
GitHub 的服务器在美国，币安合约接口（fapi）对美国 IP 返回 451。所以在 GitHub 上：
- 币安的币改用**币安现货K线**（data-api.binance.vision），1000PEPE 这类自动换算成 PEPE 现货价格 ×1000。
- 币安只有合约、没有现货的币（如 EDGE、AKE、ON、PIEVERSE），如果欧易有这个永续，就用**欧易永续K线代替**；欧易池子里已经有的不重复扫。币安现货和欧易都没有的币（目前是 JCT、ARIA）在 GitHub 上扫不到。
- 欧易照常用永续合约K线。
- 推送里每个信号带 `data_source`，用现货时整条推送带 `data_source_note`：「Binance 数据源：现货K线（GitHub 无法访问合约接口）」。现货价格和合约基本一致，但**成交量是现货的**，量能确认会有出入。

## 四、费用
私有仓库的 GitHub Actions 每月有免费额度（Free 计划 $12，约等于 1 核 runner 6000 分钟）。本工作流用最便宜的 `ubuntu-slim`（每分钟 $0.002），每天约 96 次 15m 扫描 + 24 次 1H 扫描，**估计每月会接近或略超免费额度**。超出后默认会停跑（Free 计划默认花费上限是 $0），不会自动扣钱。把仓库改成公开（Public）则 Actions 完全免费，但代码和纸面账本会公开（Secret 仍然保密）。

## 五、目录
- `backtest/gha_runner.py`：GitHub 上每次运行的入口。
- `backtest/scan_runner.py`：扫描、记账、熔断、推送（与本机守护进程同一份逻辑）。
- `backtest/binance_data.py`：币安数据源（合约 → 现货 → 欧易永续代替）。
- `backtest/render_from_payload.py`：在 Grok Bot 电脑上按推送内容重画图表。
- `core_pool.json` / `okx_pool.json`：币安 / 欧易各 30 个币；`perp_symbols.json`：币安 USDT 永续清单（池子过期又连不上合约接口时，用现货成交额重排）。
- `state/`：运行状态，请不要手动改。

## 六、本地测试
```
pip install -r requirements-dev.txt
cd backtest && python -m pytest -q tests
```
