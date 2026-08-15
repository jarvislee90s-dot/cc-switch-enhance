use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use notify_debouncer_mini::notify;
use notify_debouncer_mini::notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};
use serde_json::Value;

use crate::error::AppError;
use crate::provider::Provider;

/// watcher 更新 autoSyncState 后把 provider 配置持久化到 DB 的回调。
pub(crate) type PersistSettingsCallback = Arc<dyn Fn(Value) -> Result<(), String> + Send + Sync>;

/// 监听 settings.json 后决定的当前激活模型窗口
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveModelWindow {
    pub model: String,
    pub window: u64,
}

/// watcher 防循环用的文件快照：model 或 ACW/MAX 任一变化都视为需要处理。
#[derive(Debug, Clone, PartialEq, Eq)]
struct WatcherSnapshot {
    model: Option<String>,
    acw: Option<String>,
    max: Option<String>,
}

/// 根据 settings.json 顶层 model 字段和 provider 配置，
/// 决定要写入 ACW/MAX 的窗口值。
///
/// 返回 None 表示"不写"（model 字段无效 / 角色无可用窗口 / 无固定兜底窗口）。
pub(crate) fn resolve_active_model_window(
    settings: &Value,
    provider: &Provider,
) -> Option<ActiveModelWindow> {
    // 1. 读顶层 model 字段
    let model = settings.get("model").and_then(Value::as_str)?;
    // 2. 映射到 env 字段名
    let env_key = match model {
        "sonnet" => "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "opus" => "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "fable" => "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "haiku" => "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "subagent" => "CLAUDE_CODE_SUBAGENT_MODEL",
        _ => return None,
    };
    // 3. 优先读取 contextWindows 显式窗口，其次回退到 env 模型名后缀；
    // 两者都没有时，保留 Codex OAuth / Kimi 的固定兜底窗口。
    let window =
        crate::claude_desktop_config::resolve_context_window(&provider.settings_config, env_key)
            .or_else(|| {
                crate::services::provider::static_context_window_fallback(provider)
                    .and_then(|(acw, _max)| acw.parse().ok())
            });
    window.map(|w| ActiveModelWindow {
        model: model.to_string(),
        window: w,
    })
}

/// 读取 provider 的自动压缩比例，缺失或非法时默认 0.95。
pub(crate) fn provider_compact_ratio(provider: &Provider) -> f64 {
    provider
        .settings_config
        .get("autoSyncCompactRatio")
        .and_then(Value::as_f64)
        .filter(|ratio| ratio.is_finite() && (0.2..=0.95).contains(ratio))
        .unwrap_or(0.95)
}

/// 根据窗口值和压缩比例生成要写入 settings.json.env 的两个 env 项。
/// ACW = 窗口 × ratio（向下取整），MAX = 窗口本身。
pub(crate) fn build_env_writes(window: u64, ratio: f64) -> Vec<(&'static str, String)> {
    let ratio = if ratio.is_finite() && (0.2..=0.95).contains(&ratio) {
        ratio
    } else {
        0.95
    };
    let acw = ((window as f64) * ratio).floor() as u64;
    vec![
        ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", acw.to_string()),
        ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", window.to_string()),
    ]
}

/// 检查新事件的 model + ACW/MAX 是否与上次快照不同。
/// 这里只声明候选，不修改 state；只有写入/持久化成功路径才提交快照。
fn should_process(
    state: &Mutex<Option<WatcherSnapshot>>,
    new_model: Option<&str>,
    new_acw: Option<&str>,
    new_max: Option<&str>,
) -> bool {
    let guard = state.lock().expect("settings watcher mutex poisoned");
    let next = WatcherSnapshot {
        model: new_model.map(|s| s.to_string()),
        acw: new_acw.map(|s| s.to_string()),
        max: new_max.map(|s| s.to_string()),
    };
    *guard != Some(next)
}

fn record_processed_state(
    state: &Mutex<Option<WatcherSnapshot>>,
    model: Option<&str>,
    acw: Option<&str>,
    max: Option<&str>,
) {
    *state.lock().expect("settings watcher mutex poisoned") = Some(WatcherSnapshot {
        model: model.map(|s| s.to_string()),
        acw: acw.map(|s| s.to_string()),
        max: max.map(|s| s.to_string()),
    });
}

/// Claude Code settings.json 监听器
///
/// 后台线程监听文件变化，根据顶层 model 字段值变化自动同步 ACW/MAX。
pub struct ClaudeSettingsWatcher {
    /// 防循环用的"上次见到的 model + ACW/MAX 快照"
    #[allow(dead_code)]
    state: Arc<Mutex<Option<WatcherSnapshot>>>,
    /// 关闭信号
    shutdown: Arc<AtomicBool>,
    /// notify debouncer handle（Drop 时自动停止监听）
    _debouncer: Option<Debouncer<notify::RecommendedWatcher>>,
    /// 测试专用：保留传入的 provider 快照，便于断言内部同步字段。
    #[cfg(test)]
    provider: Arc<Mutex<Provider>>,
}

impl Drop for ClaudeSettingsWatcher {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

/// 启动 settings.json 监听器
///
/// 返回的 watcher 在 Drop 时自动停止监听。
pub(crate) fn spawn_claude_settings_watcher(
    settings_path: PathBuf,
    provider: Arc<Mutex<Provider>>,
    persist: PersistSettingsCallback,
) -> Result<ClaudeSettingsWatcher, AppError> {
    let state = Arc::new(Mutex::new(None));
    let shutdown = Arc::new(AtomicBool::new(false));

    // 启动时读一次 settings.json 初始化 state
    if let Ok(content) = std::fs::read_to_string(&settings_path) {
        if let Ok(v) = serde_json::from_str::<Value>(&content) {
            *state.lock().unwrap() = Some(WatcherSnapshot {
                model: v.get("model").and_then(Value::as_str).map(String::from),
                acw: v
                    .pointer("/env/CLAUDE_CODE_AUTO_COMPACT_WINDOW")
                    .and_then(Value::as_str)
                    .map(String::from),
                max: v
                    .pointer("/env/CLAUDE_CODE_MAX_CONTEXT_TOKENS")
                    .and_then(Value::as_str)
                    .map(String::from),
            });
        }
    }

    let state_clone = state.clone();
    let shutdown_clone = shutdown.clone();
    let provider_clone = provider.clone();
    let path_clone = settings_path.clone();
    let persist_clone = persist.clone();

    let mut debouncer = new_debouncer(
        Duration::from_millis(200),
        move |result: DebounceEventResult| {
            if shutdown_clone.load(Ordering::SeqCst) {
                return;
            }
            let events = match result {
                Ok(events) => events,
                Err(errors) => {
                    log::warn!("[ClaudeSettingsWatcher] debounce error: {errors}");
                    return;
                }
            };
            // watch 父目录会收到目录下所有文件事件。任意事件都读一次
            // settings.json 检查 model 字段，should_process 会跳过 model
            // 没变的情况，避免不必要的写入。不过滤 event.path 是因为某些
            // 平台（如 macOS FSEvents）可能报告目录级别事件，file_name
            // 过滤会漏掉。
            for _event in events {
                handle_settings_change(
                    &path_clone,
                    provider_clone.as_ref(),
                    &state_clone,
                    &persist_clone,
                );
            }
        },
    )
    .map_err(|e| AppError::Message(format!("failed to create settings watcher: {e}")))?;

    // watch 父目录而不是文件本身：atomic_write 用 rename 覆盖文件，
    // 在 inotify（Linux）上文件 watch 附加旧 inode，rename 后失效；
    // watch 父目录能持续观察文件替换 + 创建（fresh 安装时文件还不存在）。
    let watch_dir = settings_path
        .parent()
        .ok_or_else(|| AppError::Message("settings.json has no parent directory".to_string()))?;
    debouncer
        .watcher()
        .watch(watch_dir, RecursiveMode::NonRecursive)
        .map_err(|e| AppError::Message(format!("failed to watch settings.json dir: {e}")))?;

    Ok(ClaudeSettingsWatcher {
        state,
        shutdown,
        _debouncer: Some(debouncer),
        #[cfg(test)]
        provider,
    })
}

/// 进程级单例 slot，持有当前存活的 watcher。
///
/// production 路径下 spawn 出来的 watcher 必须存到这里，否则函数返回时
/// 返回值被 Drop，notify 监听线程随之退出--这正是 dev 测试里 /model 切换
/// 不更新 ACW/MAX 的根因。
static WATCHER_SLOT: OnceLock<Mutex<Option<ClaudeSettingsWatcher>>> = OnceLock::new();

fn watcher_slot() -> &'static Mutex<Option<ClaudeSettingsWatcher>> {
    WATCHER_SLOT.get_or_init(|| Mutex::new(None))
}

/// 把新 watcher 存进进程级单例，旧的自动 Drop（停止监听）。
///
/// 调用方不需要持有返回的 watcher--这正是 production 路径需要的语义：
/// spawn_claude_settings_watcher 的 Ok 返回值交给 replace_watcher，
/// 由静态 slot 接管所有权，watcher 才能存活到进程退出或下次替换。
pub fn replace_watcher(new: ClaudeSettingsWatcher) {
    let mut guard = watcher_slot().lock().expect("watcher slot mutex poisoned");
    // 旧 watcher 在赋值时自动 Drop：Drop 设 shutdown=true 并 drop debouncer，
    // notify 监听线程随之停止。新 watcher 接管监听。
    *guard = Some(new);
}

#[cfg(test)]
pub(crate) fn clear_watcher_slot_for_tests() {
    *watcher_slot().lock().expect("watcher slot mutex poisoned") = None;
}

#[cfg(test)]
pub(crate) fn watcher_slot_is_empty_for_tests() -> bool {
    watcher_slot()
        .lock()
        .expect("watcher slot mutex poisoned")
        .is_none()
}

#[cfg(test)]
pub(crate) fn watcher_provider_settings_config_for_tests() -> Option<Value> {
    let slot = watcher_slot().lock().expect("watcher slot mutex poisoned");
    let watcher = slot.as_ref()?;
    let settings_config = watcher
        .provider
        .lock()
        .expect("watcher provider mutex poisoned")
        .settings_config
        .clone();
    Some(settings_config)
}

/// 处理一次 settings.json 变化
fn handle_settings_change(
    path: &std::path::Path,
    provider: &Mutex<Provider>,
    state: &Mutex<Option<WatcherSnapshot>>,
    persist: &PersistSettingsCallback,
) {
    // 1. 读最新内容
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            log::debug!("[ClaudeSettingsWatcher] read failed: {e}");
            return;
        }
    };
    let v: Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("[ClaudeSettingsWatcher] invalid JSON: {e}");
            return;
        }
    };

    // 只有路由（代理接管）模式才由 watcher 自动同步 ACW/MAX：直连状态下
    // live settings.json 不含接管占位符，此时即使 DB 开关仍为 true（例如用户
    // 曾路由模式开开关后转直连），watcher 也绝不回写 ACW/MAX。
    if !crate::services::proxy::ProxyService::is_claude_live_taken_over(&v) {
        log::debug!(
            "[ClaudeSettingsWatcher] live settings not in takeover state, skip ACW/MAX write"
        );
        return;
    }

    let new_model = v.get("model").and_then(Value::as_str);
    let new_acw = v
        .pointer("/env/CLAUDE_CODE_AUTO_COMPACT_WINDOW")
        .and_then(Value::as_str);
    let new_max = v
        .pointer("/env/CLAUDE_CODE_MAX_CONTEXT_TOKENS")
        .and_then(Value::as_str);

    // 3. 检查 model / ACW / MAX 任一变化（防循环）
    if !should_process(state, new_model, new_acw, new_max) {
        return;
    }

    let mut provider = provider
        .lock()
        .expect("settings watcher provider mutex poisoned");

    // 4. 检查 provider 的 autoSyncContextWindow 开关（字段缺失即关闭，spec：缺失该开关时默认关闭）
    if !effective_auto_sync_enabled(&provider) {
        log::debug!("[ClaudeSettingsWatcher] auto-sync disabled for provider, skip");
        return;
    }

    // 5. 只有静态注入实际生效的路径需要跳过：Kimi 固定注入，或 Codex OAuth 的
    // env 模型全部指向 gpt-5.6。其他 Codex OAuth 配置仍按 contextWindows 增强。
    let static_injection_active =
        crate::services::provider::static_context_window_fallback(&provider).is_some();
    if static_injection_active {
        log::debug!(
            "[ClaudeSettingsWatcher] static ACW/MAX for {}, skip watcher rewrite",
            provider.name
        );
        return;
    }

    // 6. 决定要写的窗口值
    let active = match resolve_active_model_window(&v, &provider) {
        Some(a) => a,
        None => {
            log::debug!("[ClaudeSettingsWatcher] no active model window to write");
            return;
        }
    };

    // 6. 生成 ACW/MAX 并做写前二次校验，避免覆盖并发修改。
    let writes = build_env_writes(active.window, provider_compact_ratio(&provider));
    let new_content = match update_env_fields(&content, &writes) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("[ClaudeSettingsWatcher] update failed: {e}");
            return;
        }
    };
    if let Err(e) = verify_file_unchanged(path, &content) {
        log::warn!("[ClaudeSettingsWatcher] concurrent change, skip write: {e}");
        return;
    }

    // 使用原子写入（临时文件 + rename），避免 Claude Code 在写入期间
    // 读到截断后的空文件或残缺 JSON；写入成功后才更新 lastWritten。
    if let Err(e) = crate::config::atomic_write(path, new_content.as_bytes()) {
        log::warn!("[ClaudeSettingsWatcher] write failed: {e}");
    } else {
        record_last_written(&mut provider, &writes[0].1, &writes[1].1);
        record_processed_state(
            state,
            Some(active.model.as_str()),
            Some(&writes[0].1),
            Some(&writes[1].1),
        );
        // live 已写入后若 DB 持久化失败，不能回滚到与文件不一致的账本/state；
        // 保留新值等下次变化时重试，避免 rename 事件把自动写入误判成用户手改。
        if persist_settings(persist, &provider) {
            log::info!(
                "[ClaudeSettingsWatcher] wrote ACW/MAX for model={} window={}",
                active.model,
                active.window
            );
        } else {
            log::warn!(
                "[ClaudeSettingsWatcher] persist failed after live write; keeping ACW/MAX for retry"
            );
        }
    }
}

fn persist_settings(persist: &PersistSettingsCallback, provider: &Provider) -> bool {
    match persist(provider.settings_config.clone()) {
        Ok(()) => true,
        Err(e) => {
            log::warn!("[ClaudeSettingsWatcher] failed to persist autoSyncState: {e}");
            false
        }
    }
}

fn record_last_written(provider: &mut Provider, acw: &str, max: &str) {
    if let Some(state) =
        crate::services::provider::auto_sync_state_mut(&mut provider.settings_config)
    {
        state.insert(
            "lastWritten".to_string(),
            serde_json::json!({ "ACW": acw, "MAX": max }),
        );
    }
}

/// autoSyncContextWindow 开关有效值：显式字段优先，缺失即关闭（spec：缺失该开关时默认关闭）。
pub(crate) fn effective_auto_sync_enabled(provider: &Provider) -> bool {
    provider
        .settings_config
        .get("autoSyncContextWindow")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn verify_file_unchanged(path: &std::path::Path, expected: &str) -> Result<(), String> {
    let current = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    if current == expected {
        Ok(())
    } else {
        Err("settings.json changed between read and write".to_string())
    }
}

/// 原子地更新 settings.json 中 env 子对象的指定字段，其他字段全部保留
fn update_env_fields(content: &str, writes: &[(&'static str, String)]) -> Result<String, String> {
    let mut v: Value = serde_json::from_str(content).map_err(|e| e.to_string())?;
    if !v.is_object() {
        return Err("top-level not object".to_string());
    }
    // cc-switch 内部字段（contextWindows / autoSyncState 等）不能写回 settings.json。
    v = crate::services::provider::sanitize_claude_settings_for_live(&v);
    let obj = v.as_object_mut().unwrap();
    let env = obj
        .entry("env".to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !env.is_object() {
        return Err("env not object".to_string());
    }
    let env_obj = env.as_object_mut().unwrap();
    for (key, value) in writes {
        env_obj.insert((*key).to_string(), Value::String(value.clone()));
    }
    serde_json::to_string_pretty(&v).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use serial_test::serial;

    fn make_provider(env: Value) -> Provider {
        Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({ "env": env }),
            None,
        )
    }

    fn noop_persist() -> Arc<dyn Fn(Value) -> Result<(), String> + Send + Sync> {
        Arc::new(|_| Ok(()))
    }

    fn recording_persist(
        calls: Arc<std::sync::atomic::AtomicUsize>,
        captured: Arc<Mutex<Option<Value>>>,
    ) -> Arc<dyn Fn(Value) -> Result<(), String> + Send + Sync> {
        Arc::new(move |settings| {
            calls.fetch_add(1, Ordering::SeqCst);
            *captured.lock().unwrap() = Some(settings);
            Ok(())
        })
    }

    fn failing_persist() -> Arc<dyn Fn(Value) -> Result<(), String> + Send + Sync> {
        Arc::new(|_| Err("db unavailable".to_string()))
    }

    fn fail_once_then_succeed_persist() -> Arc<dyn Fn(Value) -> Result<(), String> + Send + Sync> {
        let failed = Arc::new(AtomicBool::new(false));
        Arc::new(move |_| {
            if failed.swap(true, Ordering::SeqCst) {
                Ok(())
            } else {
                Err("db unavailable".to_string())
            }
        })
    }

    /// 给 live settings.json 注入接管占位符，模拟路由（代理接管）状态。
    /// I1 守卫：直连（无占位符）时 watcher 不写 ACW/MAX，所有断言"写成功"的
    /// 测试都必须用本 helper 生成接管态 live 文件。
    fn taken_over_live(mut live: Value) -> Value {
        let obj = live
            .as_object_mut()
            .expect("live settings must be a JSON object");
        let env = obj
            .entry("env".to_string())
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        env.as_object_mut()
            .expect("live env must be a JSON object")
            .insert(
                "ANTHROPIC_AUTH_TOKEN".to_string(),
                Value::String("PROXY_MANAGED".to_string()),
            );
        live
    }

    // ========== Task 2: 角色映射测试 ==========

    #[test]
    fn resolve_maps_haiku_to_anthropic_default_haiku_model() {
        let settings = json!({ "model": "haiku" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]"
        }));
        let result = resolve_active_model_window(&settings, &provider).unwrap();
        assert_eq!(result.model, "haiku");
        assert_eq!(result.window, 30000);
    }

    #[test]
    fn resolve_maps_sonnet_to_anthropic_default_sonnet_model() {
        let settings = json!({ "model": "sonnet" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]","ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
        }));
        let result = resolve_active_model_window(&settings, &provider).unwrap();
        assert_eq!(result.model, "sonnet");
        assert_eq!(result.window, 1000000);
    }

    #[test]
    fn resolve_maps_opus_to_anthropic_default_opus_model() {
        let settings = json!({ "model": "opus" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro[1000000]"
        }));
        let result = resolve_active_model_window(&settings, &provider).unwrap();
        assert_eq!(result.model, "opus");
        assert_eq!(result.window, 1000000);
    }

    #[test]
    fn resolve_maps_fable_to_anthropic_default_fable_model() {
        let settings = json!({ "model": "fable" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_FABLE_MODEL": "GLM-5.2[200k]"
        }));
        let result = resolve_active_model_window(&settings, &provider).unwrap();
        assert_eq!(result.model, "fable");
        assert_eq!(result.window, 200000);
    }

    #[test]
    fn resolve_maps_subagent_to_claude_code_subagent_model() {
        let settings = json!({ "model": "subagent" });
        let provider = make_provider(json!({
            "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-v4-flash"
        }));
        // subagent 没后缀 → 期望 None（无窗口可写）
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_maps_codex_oauth_gpt56_without_suffix_to_372k() {
        let settings = json!({ "model": "sonnet" });
        let mut provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.6",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.6-luna"
        }));
        provider.meta = Some(crate::provider::ProviderMeta {
            provider_type: Some("codex_oauth".to_string()),
            ..Default::default()
        });
        let result = resolve_active_model_window(&settings, &provider).unwrap();
        assert_eq!(result.model, "sonnet");
        assert_eq!(result.window, 372000);
    }

    #[test]
    fn resolve_maps_kimi_without_suffix_to_262k() {
        let settings = json!({ "model": "sonnet" });
        let provider = make_provider(json!({
            "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-for-coding"
        }));
        let result = resolve_active_model_window(&settings, &provider).unwrap();
        assert_eq!(result.model, "sonnet");
        assert_eq!(result.window, 262144);
    }

    #[test]
    fn resolve_active_model_window_uses_context_windows_without_env_suffix() {
        let settings = json!({ "model": "sonnet" });
        let provider = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet-model" },
                "contextWindows": { "ANTHROPIC_DEFAULT_SONNET_MODEL": 200000 }
            }),
            None,
        );
        let result = resolve_active_model_window(&settings, &provider).unwrap();
        assert_eq!(result.model, "sonnet");
        assert_eq!(result.window, 200000);
    }

    #[test]
    fn resolve_active_model_window_context_windows_beats_legacy_suffix() {
        let settings = json!({ "model": "sonnet" });
        let provider = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "model[200k]" },
                "contextWindows": { "ANTHROPIC_DEFAULT_SONNET_MODEL": 500000 }
            }),
            None,
        );
        let result = resolve_active_model_window(&settings, &provider).unwrap();
        assert_eq!(result.model, "sonnet");
        assert_eq!(result.window, 500000);
    }

    // ========== Task 3: build_env_writes 测试 ==========

    #[test]
    fn build_writes_for_30k_window() {
        let writes = build_env_writes(30000, 0.8);
        assert_eq!(
            writes,
            vec![
                ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "24000".to_string()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "30000".to_string()),
            ]
        );
    }

    #[test]
    fn build_writes_for_1m_window() {
        let writes = build_env_writes(1000000, 0.8);
        assert_eq!(
            writes,
            vec![
                ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "800000".to_string()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "1000000".to_string()),
            ]
        );
    }

    #[test]
    fn build_writes_for_200k_window() {
        let writes = build_env_writes(200000, 0.8);
        assert_eq!(
            writes,
            vec![
                ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "160000".to_string()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "200000".to_string()),
            ]
        );
    }

    #[test]
    fn build_writes_for_1_token_boundary() {
        // 最小边界：1 token → ACW=0（×0.8 = 0.8，向下取整 = 0）
        let writes = build_env_writes(1, 0.8);
        assert_eq!(
            writes,
            vec![
                ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "0".to_string()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "1".to_string()),
            ]
        );
    }

    #[test]
    fn compact_ratio_defaults_to_095_when_missing_or_invalid() {
        let missing =
            Provider::with_id("p".to_string(), "P".to_string(), json!({ "env": {} }), None);
        assert_eq!(provider_compact_ratio(&missing), 0.95);

        let too_low = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({ "autoSyncCompactRatio": 0.1 }),
            None,
        );
        assert_eq!(provider_compact_ratio(&too_low), 0.95);

        let too_high = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({ "autoSyncCompactRatio": 1.5 }),
            None,
        );
        assert_eq!(provider_compact_ratio(&too_high), 0.95);
    }

    #[test]
    fn compact_ratio_rejects_values_above_095() {
        let explicit_one = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({ "autoSyncCompactRatio": 1.0 }),
            None,
        );
        assert_eq!(provider_compact_ratio(&explicit_one), 0.95);

        let too_high = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({ "autoSyncCompactRatio": 0.96 }),
            None,
        );
        assert_eq!(provider_compact_ratio(&too_high), 0.95);
    }

    #[test]
    fn compact_ratio_accepts_upper_bound() {
        let provider = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({ "autoSyncCompactRatio": 0.95 }),
            None,
        );
        assert_eq!(provider_compact_ratio(&provider), 0.95);
    }

    #[test]
    fn compact_ratio_accepts_valid_range() {
        let provider = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({ "autoSyncCompactRatio": 0.6 }),
            None,
        );
        assert_eq!(provider_compact_ratio(&provider), 0.6);
    }

    #[test]
    fn build_writes_use_custom_ratio() {
        let writes = build_env_writes(30000, 0.5);
        assert_eq!(
            writes,
            vec![
                ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "15000".to_string()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "30000".to_string()),
            ]
        );
    }

    #[test]
    fn build_writes_use_custom_ratio_upper_bound() {
        let writes = build_env_writes(30000, 0.95);
        assert_eq!(
            writes,
            vec![
                ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "28500".to_string()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "30000".to_string()),
            ]
        );
    }

    #[test]
    fn build_writes_fallback_to_095_for_ratio_above_095() {
        for ratio in [0.96, 1.0] {
            let writes = build_env_writes(30000, ratio);
            assert_eq!(
                writes,
                vec![
                    ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "28500".to_string()),
                    ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "30000".to_string()),
                ]
            );
        }
    }

    #[test]
    fn build_writes_fallback_to_one_for_invalid_ratio() {
        let writes = build_env_writes(30000, 2.0);
        assert_eq!(
            writes,
            vec![
                ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "28500".to_string()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "30000".to_string()),
            ]
        );
    }

    #[test]
    fn compact_ratio_defaults_to_095_when_missing() {
        // D1：autoSyncCompactRatio 缺失时默认 0.95，开箱行为 ACW = 窗口 × 0.95。
        let missing =
            Provider::with_id("p".to_string(), "P".to_string(), json!({ "env": {} }), None);
        assert_eq!(provider_compact_ratio(&missing), 0.95);
        let writes = build_env_writes(30000, provider_compact_ratio(&missing));
        assert_eq!(
            writes,
            vec![
                ("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "28500".to_string()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "30000".to_string()),
            ]
        );
    }
    // ========== Task 4: 无效输入处理测试 ==========

    #[test]
    fn resolve_returns_none_when_model_field_missing() {
        let settings = json!({});
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi[30k]"
        }));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_model_value_unknown() {
        let settings = json!({ "model": "custom-alias" });
        let provider = make_provider(json!({}));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_model_is_not_string() {
        let settings = json!({ "model": 123 });
        let provider = make_provider(json!({}));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_model_is_null() {
        let settings = json!({ "model": null });
        let provider = make_provider(json!({}));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_role_env_field_missing() {
        let settings = json!({ "model": "haiku" });
        let provider = make_provider(json!({}));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_env_value_not_string() {
        let settings = json!({ "model": "haiku" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": { "name": "weird" }
        }));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_suffix_invalid() {
        let settings = json!({ "model": "haiku" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "model[invalid]"
        }));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_suffix_zero() {
        let settings = json!({ "model": "haiku" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "model[0]"
        }));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_suffix_unsupported_unit() {
        let settings = json!({ "model": "haiku" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "model[1G]"
        }));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_suffix_decimal() {
        let settings = json!({ "model": "haiku" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "model[1.5m]"
        }));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    #[test]
    fn resolve_returns_none_when_no_suffix_at_all() {
        let settings = json!({ "model": "haiku" });
        let provider = make_provider(json!({
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.6"
        }));
        assert!(resolve_active_model_window(&settings, &provider).is_none());
    }

    // ========== Task 5: 防循环测试 ==========

    #[test]
    fn loop_same_model_consecutive_triggers() {
        let state = Mutex::new(None);
        assert!(should_process(&state, Some("haiku"), None, None));
        // 尚未成功提交前，相同事件仍是候选，后续失败/早退路径可以重试。
        assert!(should_process(&state, Some("haiku"), None, None));
        record_processed_state(&state, Some("haiku"), None, None);
        assert!(!should_process(&state, Some("haiku"), None, None));
        assert!(!should_process(&state, Some("haiku"), None, None));
    }

    #[test]
    fn loop_two_models_alternating() {
        let state = Mutex::new(None);
        assert!(should_process(&state, Some("haiku"), None, None));
        assert!(should_process(&state, Some("sonnet"), None, None));
        assert!(should_process(&state, Some("haiku"), None, None));
        assert!(should_process(&state, Some("sonnet"), None, None));
        record_processed_state(&state, Some("sonnet"), None, None);
        assert!(!should_process(&state, Some("sonnet"), None, None));
    }

    #[test]
    fn loop_model_to_none_to_same() {
        let state = Mutex::new(None);
        assert!(should_process(&state, Some("haiku"), None, None));
        record_processed_state(&state, Some("haiku"), None, None);
        assert!(!should_process(&state, Some("haiku"), None, None));
        assert!(should_process(&state, None, None, None)); // model 被删
        record_processed_state(&state, None, None, None);
        assert!(should_process(&state, Some("haiku"), None, None)); // 重新出现
    }

    #[test]
    fn loop_model_to_none_stays_none() {
        let state = Mutex::new(None);
        assert!(should_process(&state, Some("haiku"), None, None));
        assert!(should_process(&state, None, None, None));
        record_processed_state(&state, None, None, None);
        // 后续 None 事件都算"无变化" → 跳过
        assert!(!should_process(&state, None, None, None));
        assert!(!should_process(&state, None, None, None));
    }

    #[test]
    fn loop_initial_state_with_existing_model() {
        // 启动时 model 已经是 "haiku"（如上次会话留下的）
        let state = Mutex::new(Some(WatcherSnapshot {
            model: Some("haiku".to_string()),
            acw: None,
            max: None,
        }));
        // 第一次触发就是 haiku → 跳过（不算变化）
        assert!(!should_process(&state, Some("haiku"), None, None));
        // 切到别的 → 处理
        assert!(should_process(&state, Some("sonnet"), None, None));
        assert!(should_process(&state, Some("sonnet"), None, None));
        record_processed_state(&state, Some("sonnet"), None, None);
        assert!(!should_process(&state, Some("sonnet"), None, None));
    }

    // ========== Task 6: 文件系统集成测试 ==========

    #[test]
    fn fs_update_env_fields_writes_only_env_keys() {
        let original = r#"{"model":"sonnet","effortLevel":"xhigh","env":{"ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]"}}"#;
        let writes = build_env_writes(1000000, 0.8);
        let result = update_env_fields(original, &writes).unwrap();
        let v: Value = serde_json::from_str(&result).unwrap();
        // 顶层字段不动
        assert_eq!(v["model"], "sonnet");
        assert_eq!(v["effortLevel"], "xhigh");
        // env 只加了 ACW/MAX
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "800000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "1000000");
        // 原有 env 字段保留
        assert_eq!(v["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"], "MiniMax-M3[1M]");
    }

    #[test]
    fn fs_update_env_fields_creates_env_if_missing() {
        let original = r#"{"model":"haiku","effortLevel":"max"}"#;
        let writes = build_env_writes(30000, 0.8);
        let result = update_env_fields(original, &writes).unwrap();
        let v: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["model"], "haiku");
        assert_eq!(v["effortLevel"], "max");
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "24000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");
    }

    #[test]
    fn fs_update_env_fields_preserves_existing_env_fields() {
        let original = r#"{"model":"haiku","env":{"ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi[30k]","CLAUDE_CODE_SUBAGENT_MODEL":"deepseek"}}"#;
        let writes = build_env_writes(30000, 0.8);
        let result = update_env_fields(original, &writes).unwrap();
        let v: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "Kimi[30k]");
        assert_eq!(v["env"]["CLAUDE_CODE_SUBAGENT_MODEL"], "deepseek");
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "24000");
    }

    #[test]
    fn fs_update_env_fields_overwrites_existing_acw_max() {
        let original = r#"{"model":"haiku","env":{"CLAUDE_CODE_AUTO_COMPACT_WINDOW":"999","CLAUDE_CODE_MAX_CONTEXT_TOKENS":"888"}}"#;
        let writes = build_env_writes(30000, 0.8);
        let result = update_env_fields(original, &writes).unwrap();
        let v: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "24000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");
    }

    #[test]
    fn update_env_fields_strips_internal_fields_before_writing() {
        let original = r#"{"model":"sonnet","contextWindows":{"ANTHROPIC_DEFAULT_SONNET_MODEL":200000},"autoSyncContextWindow":true,"autoSyncCompactRatio":0.8,"autoSyncState":{"lastWritten":{"ACW":"160000","MAX":"200000"}},"env":{}}"#;
        let writes = build_env_writes(200000, 0.8);
        let result = update_env_fields(original, &writes).unwrap();
        let v: Value = serde_json::from_str(&result).unwrap();
        for key in [
            "contextWindows",
            "autoSyncContextWindow",
            "autoSyncCompactRatio",
            "autoSyncState",
        ] {
            assert!(v.get(key).is_none(), "{key} leaked into live settings");
        }
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "160000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "200000");
    }

    /// 用 tempfile 创建临时目录，验证真实 fs 事件的 watcher 行为
    #[test]
    fn fs_real_watcher_external_model_change() {
        use std::fs;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]","ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Arc::new(Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]","ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
                },
                "autoSyncContextWindow":true
            }),
            None,
        )));

        let watcher =
            spawn_claude_settings_watcher(path.clone(), provider, noop_persist()).unwrap();

        // 模拟外部程序修改 model 字段
        let new_content = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]",
                "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]","ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(new_content).to_string()).unwrap();

        // 等待 debouncer + 文件写入生效
        thread::sleep(Duration::from_millis(800));

        // 验证 ACW/MAX 已被写入
        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["model"], "haiku");
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "28500");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");

        drop(watcher);
    }

    /// provider 配置 autoSyncCompactRatio 时，watcher 按该比例写 ACW。
    #[test]
    #[serial]
    fn fs_real_watcher_uses_provider_compact_ratio() {
        use std::fs;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Arc::new(Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3[1M]",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]"
                },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.5
            }),
            None,
        )));

        let watcher =
            spawn_claude_settings_watcher(path.clone(), provider, noop_persist()).unwrap();

        let new_content = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(new_content).to_string()).unwrap();

        thread::sleep(Duration::from_millis(800));

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "15000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");

        drop(watcher);
    }
    /// 只改 effortLevel 不应该触发 ACW/MAX 写入

    /// Kimi/Codex OAuth 的静态 ACW/MAX 写入 live 后，watcher 不应再按压缩比例重写。
    #[test]
    fn handle_settings_change_keeps_kimi_static_acw_max() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-for-coding",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "262144",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "262144"
            },
            "autoSyncState": { "staticInjected": { "ACW": "262144", "MAX": "262144" } }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "kimi".to_string(),
            "Kimi For Coding".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-for-coding"
                },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8,
                "autoSyncState": { "staticInjected": { "ACW": "262144", "MAX": "262144" } }
            }),
            None,
        ));

        let state = Mutex::new(None);
        handle_settings_change(&path, &provider, &state, &noop_persist());

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "262144");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "262144");
    }

    #[test]
    fn handle_settings_change_keeps_codex_oauth_static_acw_max() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.6",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "372000",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "372000"
            },
            "autoSyncState": { "staticInjected": { "ACW": "372000", "MAX": "372000" } }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "codex-oauth".to_string(),
            "Codex".to_string(),
            json!({
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.6" },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8,
                "autoSyncState": { "staticInjected": { "ACW": "372000", "MAX": "372000" } }
            }),
            None,
        ));
        provider.lock().unwrap().meta = Some(crate::provider::ProviderMeta {
            provider_type: Some("codex_oauth".to_string()),
            ..Default::default()
        });

        let state = Mutex::new(None);
        handle_settings_change(&path, &provider, &state, &noop_persist());

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "372000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "372000");
    }

    #[test]
    fn handle_settings_change_syncs_non_gpt56_codex_oauth_from_context_windows() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.5-mini"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "codex-oauth-non-gpt56".to_string(),
            "Codex Non-GPT56".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.5",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.5-mini"
                },
                "contextWindows": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": 432000 },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8
            }),
            None,
        ));
        provider.lock().unwrap().meta = Some(crate::provider::ProviderMeta {
            provider_type: Some("codex_oauth".to_string()),
            ..Default::default()
        });

        // 模拟 model 从 sonnet 切到 haiku，且 provider 不触发 gpt-5.6 静态注入。
        let state = Mutex::new(Some(WatcherSnapshot {
            model: Some("sonnet".to_string()),
            acw: None,
            max: None,
        }));
        handle_settings_change(&path, &provider, &state, &noop_persist());

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "345600");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "432000");
    }
    #[test]
    #[serial]
    fn fs_real_watcher_effort_change_no_trigger() {
        use std::fs;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        // spawn 时文件尚不存在 → watcher 初始快照为空
        let provider = Arc::new(Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
                },
                // 开关开启 + 带窗口模型：让"effort-only 不触发"真正走到写入判定路径
                "autoSyncContextWindow": true
            }),
            None,
        )));

        let watcher =
            spawn_claude_settings_watcher(path.clone(), provider, noop_persist()).unwrap();

        // 首次写：触发 watcher 建立快照并写入 ACW/MAX（1M 窗口）
        fs::write(
            &path,
            taken_over_live(json!({
                "model": "sonnet",
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
                }
            }))
            .to_string(),
        )
        .unwrap();
        thread::sleep(Duration::from_millis(800));

        // 只改 effortLevel（model / ACW / MAX 均未变化）→ 不应重写
        fs::write(
            &path,
            taken_over_live(json!({
                "model": "sonnet",
                "effortLevel": "max",
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
                }
            }))
            .to_string(),
        )
        .unwrap();
        thread::sleep(Duration::from_millis(800));

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["effortLevel"], "max");
        // 首次写入的 ACW/MAX 保持，effort-only 不重写
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "950000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "1000000");

        drop(watcher);
    }

    /// 设置不存在时的行为
    #[test]
    fn fs_real_watcher_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nonexistent.json");

        let provider = Arc::new(Mutex::new(make_provider(json!({
            "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]","ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
        }))));

        let result = spawn_claude_settings_watcher(path, provider, noop_persist());
        // 文件不存在 → 应该出错（watch 失败）
        assert!(
            result.is_ok(),
            "父目录存在时 spawn 应成功，能检测后续文件创建"
        );
    }
    /// 回归测试：production 路径下 spawn 的 watcher 必须靠 replace_watcher 存活，
    /// 不能因为返回值没绑定到局部变量就被 Drop。
    ///
    /// 修复前（直接 if-let-Err 丢弃 Ok 返回值）：watcher 构造完立即 Drop，
    /// notify 线程退出，改文件后 ACW/MAX 不会被写入。
    /// 修复后（spawn 的 Ok 交给 replace_watcher 存进进程单例）：watcher 存活，
    /// 改 model 字段后 ACW/MAX 正确写入。
    #[test]
    #[serial]
    fn fs_watcher_survives_via_replace_watcher_without_local_binding() {
        use std::fs;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Arc::new(Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
                },
                "autoSyncContextWindow":true
            }),
            None,
        )));

        // 模拟 production 调用：spawn 后立即存进进程单例，不保留局部绑定。
        // spawned 在这里 move 进 replace_watcher，没有局部变量持有 watcher。
        let spawned =
            spawn_claude_settings_watcher(path.clone(), provider, noop_persist()).unwrap();
        replace_watcher(spawned);

        // 改 model 字段，模拟 Claude Code /model 切换 sonnet -> haiku
        let new_content = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(new_content).to_string()).unwrap();

        thread::sleep(Duration::from_millis(800));

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["model"], "haiku");
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "28500");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");
    }

    /// 回归测试 #4：autoSyncContextWindow=false 时，model 字段变化不写 ACW/MAX。
    /// 验证开关关闭后终端切模型不会同步（开关行为链路：toggle OFF -> save ->
    /// update -> write_live -> replace_watcher(新 provider 快照) -> watcher 读到 false -> skip）。
    #[test]
    #[serial]
    fn fs_watcher_auto_sync_disabled_skips_writes() {
        use std::fs;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        // provider 显式关闭 autoSyncContextWindow
        let provider = Arc::new(Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
                },
                "autoSyncContextWindow": false
            }),
            None,
        )));

        let spawned =
            spawn_claude_settings_watcher(path.clone(), provider, noop_persist()).unwrap();
        replace_watcher(spawned);

        // 改 model 字段 sonnet -> haiku
        let new_content = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(new_content).to_string()).unwrap();

        thread::sleep(Duration::from_millis(800));

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        // model 字段确实变了（说明事件被收到），但 ACW/MAX 不应该被写
        assert_eq!(v["model"], "haiku");
        assert!(
            v["env"].get("CLAUDE_CODE_AUTO_COMPACT_WINDOW").is_none(),
            "autoSync OFF 时不应写 ACW，但实际写入了: {:?}",
            v["env"].get("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
        );
        assert!(
            v["env"].get("CLAUDE_CODE_MAX_CONTEXT_TOKENS").is_none(),
            "autoSync OFF 时不应写 MAX，但实际写入了: {:?}",
            v["env"].get("CLAUDE_CODE_MAX_CONTEXT_TOKENS")
        );
    }

    /// 回归测试 P1#1：watch 父目录后，atomic_write（rename 覆盖）仍能被观察到。
    /// write_live_snapshot 用 atomic_write 写 settings.json，在 inotify（Linux）
    /// 上 watch 文件会因 inode 替换失效；watch 父目录能持续观察文件替换。
    #[test]
    #[serial]
    fn fs_watcher_observes_atomic_write_replacement() {
        use std::fs;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Arc::new(Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
                },
                "autoSyncContextWindow":true
            }),
            None,
        )));

        let spawned =
            spawn_claude_settings_watcher(path.clone(), provider, noop_persist()).unwrap();
        replace_watcher(spawned);

        // 用 atomic_write 覆盖 settings.json（模拟 write_live_snapshot）
        let new_content = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL":"MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL":"Kimi-K2.7-Code[30k]"
            }
        });
        crate::config::atomic_write(&path, taken_over_live(new_content).to_string().as_bytes())
            .unwrap();

        thread::sleep(Duration::from_millis(800));

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["model"], "haiku");
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "28500");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");
    }

    // ========== Task 6: pretty、记账、用户显式、二次校验 ==========

    #[test]
    fn watcher_writes_use_pretty_json_and_update_last_written_on_success() {
        let content = r#"{"model":"sonnet","env":{}}"#;
        let writes = build_env_writes(200000, 0.8);
        let next = update_env_fields(content, &writes).unwrap();
        assert!(next.contains('\n'), "expected pretty JSON");
        let v: Value = serde_json::from_str(&next).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "160000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "200000");
    }

    #[test]
    fn should_process_declares_candidate_without_committing_state() {
        let state = Mutex::new(None);
        assert!(should_process(
            &state,
            Some("sonnet"),
            Some("160000"),
            Some("200000")
        ));
        // 失败/早退路径必须保留旧快照，后续同一事件才能重试。
        assert!(should_process(
            &state,
            Some("sonnet"),
            Some("160000"),
            Some("200000")
        ));
        assert_eq!(*state.lock().unwrap(), None);

        record_processed_state(&state, Some("sonnet"), Some("160000"), Some("200000"));
        assert!(!should_process(
            &state,
            Some("sonnet"),
            Some("160000"),
            Some("200000")
        ));
    }

    #[test]
    fn should_process_tracks_acw_max_changes() {
        let state = Mutex::new(None);
        assert!(should_process(
            &state,
            Some("sonnet"),
            Some("160000"),
            Some("200000")
        ));
        assert!(should_process(
            &state,
            Some("sonnet"),
            Some("160000"),
            Some("200000")
        ));
        record_processed_state(&state, Some("sonnet"), Some("160000"), Some("200000"));
        assert!(!should_process(
            &state,
            Some("sonnet"),
            Some("160000"),
            Some("200000")
        ));
        assert!(should_process(
            &state,
            Some("sonnet"),
            Some("250000"),
            Some("250000")
        ));
        assert!(should_process(
            &state,
            Some("sonnet"),
            Some("250000"),
            Some("250000")
        ));
        record_processed_state(&state, Some("sonnet"), Some("250000"), Some("250000"));
        assert!(!should_process(
            &state,
            Some("sonnet"),
            Some("250000"),
            Some("250000")
        ));
    }

    #[test]
    fn handle_settings_change_records_last_written_on_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2" }
        });
        std::fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2" },
                "contextWindows": { "ANTHROPIC_DEFAULT_SONNET_MODEL": 200000 },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8,
                "autoSyncState": { "lastWritten": {} }
            }),
            None,
        ));
        let state = Mutex::new(None);

        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let captured = Arc::new(Mutex::new(None));
        let persist = recording_persist(calls.clone(), captured.clone());
        handle_settings_change(&path, &provider, &state, &persist);

        let content = std::fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "160000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "200000");
        let provider_guard = provider.lock().unwrap();
        assert_eq!(
            provider_guard.settings_config["autoSyncState"]["lastWritten"],
            json!({ "ACW": "160000", "MAX": "200000" })
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let captured_guard = captured.lock().unwrap();
        assert_eq!(
            captured_guard.as_ref().unwrap()["autoSyncState"]["lastWritten"],
            json!({ "ACW": "160000", "MAX": "200000" })
        );
    }

    #[test]
    fn handle_settings_change_keeps_ledger_and_state_when_persist_fails_after_write() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "haiku",
            "env": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]" }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]" },
                "contextWindows": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": 30000 },
                "autoSyncContextWindow": true,
                "autoSyncState": { "lastWritten": { "ACW": "1", "MAX": "2" } }
            }),
            None,
        ));
        let previous = Some(WatcherSnapshot {
            model: Some("sonnet".to_string()),
            acw: None,
            max: None,
        });
        let state = Mutex::new(previous.clone());

        handle_settings_change(&path, &provider, &state, &failing_persist());

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "28500");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");

        let provider_guard = provider.lock().unwrap();
        assert_eq!(
            provider_guard.settings_config["autoSyncState"]["lastWritten"],
            json!({ "ACW": "28500", "MAX": "30000" })
        );
        drop(provider_guard);
        assert_eq!(
            *state.lock().unwrap(),
            Some(WatcherSnapshot {
                model: Some("haiku".to_string()),
                acw: Some("28500".to_string()),
                max: Some("30000".to_string()),
            })
        );
        assert!(!should_process(
            &state,
            Some("haiku"),
            Some("28500"),
            Some("30000")
        ));
    }

    #[test]
    fn failing_persist_after_write_keeps_auto_sync_flowing() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        fs::write(
            &path,
            taken_over_live(json!({
                "model": "haiku",
                "env": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]" }
            }))
            .to_string(),
        )
        .unwrap();

        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2"
                },
                "contextWindows": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": 30000,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": 200000
                },
                "autoSyncContextWindow": true,
                "autoSyncState": { "lastWritten": { "ACW": "1", "MAX": "2" } }
            }),
            None,
        ));
        let state = Mutex::new(None);

        // 第一次写 live 成功但 DB 持久化失败，必须保留新 live 值/账本/state，
        // 否则 atomic_write 触发的同名事件会被误判成用户手改。
        let persist = fail_once_then_succeed_persist();
        handle_settings_change(&path, &provider, &state, &persist);
        handle_settings_change(&path, &provider, &state, &persist);

        // 自动同步仍可继续：后续真实切换 model 仍会写入新窗口，而不是永久停同步。
        fs::write(
            &path,
            taken_over_live(json!({
                "model": "sonnet",
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2" }
            }))
            .to_string(),
        )
        .unwrap();
        handle_settings_change(&path, &provider, &state, &persist);

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "190000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "200000");
    }

    #[test]
    fn handle_settings_change_rebuilt_watcher_with_stale_ledger_keeps_auto_sync() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku-model",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet-model",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "24000",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "30000"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        // DB 账本还是旧值，live 文件已包含上一次自动写入的 ACW/MAX；
        // watcher 重建时 state 从空快照开始，不能用旧 lastWritten 判成用户手改。
        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku-model",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet-model"
                },
                "contextWindows": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": 30000,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": 200000
                },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8,
                "autoSyncState": { "lastWritten": { "ACW": "1", "MAX": "2" } }
            }),
            None,
        ));
        let state = Mutex::new(None);

        handle_settings_change(&path, &provider, &state, &noop_persist());

        // 自动同步不停止：下一次真实切换仍按目标写入。
        fs::write(
            &path,
            taken_over_live(json!({
                "model": "sonnet",
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet-model" }
            }))
            .to_string(),
        )
        .unwrap();
        handle_settings_change(&path, &provider, &state, &noop_persist());

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "160000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "200000");
    }

    #[test]
    fn handle_settings_change_calibrates_against_all_configured_role_targets() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku-model",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet-model",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "160000",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        // live 保留 sonnet 的自动写入值，但 DB lastWritten 缺失、当前 model 已是 haiku：
        // watcher 应直接按当前 model（haiku）的目标窗口重写，而非保留旧值。
        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku-model",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "sonnet-model"
                },
                "contextWindows": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": 30000,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": 200000
                },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8,
                "autoSyncState": { "lastWritten": {} }
            }),
            None,
        ));
        let state = Mutex::new(None);

        handle_settings_change(&path, &provider, &state, &noop_persist());

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "24000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");

        let provider_guard = provider.lock().unwrap();
        assert_eq!(
            provider_guard.settings_config["autoSyncState"]["lastWritten"],
            json!({ "ACW": "24000", "MAX": "30000" })
        );
    }

    #[test]
    fn handle_settings_change_kimi_raw_static_target_without_ledger_is_auto() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-for-coding",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-for-coding",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "262144",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "262144"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "kimi".to_string(),
            "Kimi For Coding".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-for-coding",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-for-coding"
                },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8,
                "autoSyncState": { "lastWritten": {} }
            }),
            None,
        ));
        let state = Mutex::new(Some(WatcherSnapshot {
            model: Some("sonnet".to_string()),
            acw: None,
            max: None,
        }));

        handle_settings_change(&path, &provider, &state, &noop_persist());

        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "262144");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "262144");
    }

    #[test]
    fn handle_settings_change_codex_oauth_raw_static_target_without_ledger_is_auto() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "haiku",
            "env": {
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.6-mini",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.6",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "372000",
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "372000"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "codex-oauth".to_string(),
            "Codex".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.6-mini",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.6"
                },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8,
                "autoSyncState": { "lastWritten": {} }
            }),
            None,
        ));
        provider.lock().unwrap().meta = Some(crate::provider::ProviderMeta {
            provider_type: Some("codex_oauth".to_string()),
            ..Default::default()
        });
        let state = Mutex::new(Some(WatcherSnapshot {
            model: Some("sonnet".to_string()),
            acw: None,
            max: None,
        }));

        handle_settings_change(&path, &provider, &state, &noop_persist());

        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "372000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "372000");
    }

    #[test]
    fn failing_persist_retry_after_state_reset_does_not_mark_auto_target_explicit() {
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "haiku",
            "env": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi[30k]" }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi[30k]",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM[200k]"
                },
                "autoSyncContextWindow": true,
                "autoSyncCompactRatio": 0.8,
                "autoSyncState": { "lastWritten": { "ACW": "1", "MAX": "2" } }
            }),
            None,
        ));
        let state = Mutex::new(None);

        // 第一次 live 写入成功但 DB 持久化失败；第二次相同事件不再被 should_process 跳过，
        // 并且 DB 账本仍是旧值，必须按当前 model 的自动目标校准重写。
        let persist = fail_once_then_succeed_persist();
        handle_settings_change(&path, &provider, &state, &persist);
        provider.lock().unwrap().settings_config["autoSyncState"]["lastWritten"] =
            json!({ "ACW": "1", "MAX": "2" });
        *state.lock().unwrap() = None;
        handle_settings_change(&path, &provider, &state, &persist);

        fs::write(
            &path,
            taken_over_live(json!({
                "model": "sonnet",
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM[200k]" }
            }))
            .to_string(),
        )
        .unwrap();
        handle_settings_change(&path, &provider, &state, &persist);

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "160000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "200000");
    }

    #[cfg(unix)]
    #[test]
    fn handle_settings_change_keeps_last_written_when_atomic_write_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2" }
        });
        std::fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2" },
                "contextWindows": { "ANTHROPIC_DEFAULT_SONNET_MODEL": 200000 },
                "autoSyncContextWindow": true,
                "autoSyncState": { "lastWritten": { "ACW": "1", "MAX": "2" } }
            }),
            None,
        ));
        let state = Mutex::new(None);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(dir.path()).unwrap().permissions();
            permissions.set_mode(0o555);
            std::fs::set_permissions(dir.path(), permissions).unwrap();
        }

        handle_settings_change(&path, &provider, &state, &noop_persist());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = std::fs::metadata(dir.path()).unwrap().permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(dir.path(), permissions).unwrap();
        }

        let provider_guard = provider.lock().unwrap();
        assert_eq!(
            provider_guard.settings_config["autoSyncState"]["lastWritten"],
            json!({ "ACW": "1", "MAX": "2" })
        );
    }

    #[cfg(unix)]
    #[test]
    fn handle_settings_change_retries_same_event_after_atomic_write_fails() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "haiku",
            "env": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku-model" }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": "haiku-model" },
                "contextWindows": { "ANTHROPIC_DEFAULT_HAIKU_MODEL": 30000 },
                "autoSyncContextWindow": true,
                "autoSyncState": { "lastWritten": { "ACW": "1", "MAX": "2" } }
            }),
            None,
        ));
        let previous = Some(WatcherSnapshot {
            model: Some("sonnet".to_string()),
            acw: None,
            max: None,
        });
        let state = Mutex::new(previous.clone());

        let mut permissions = fs::metadata(dir.path()).unwrap().permissions();
        permissions.set_mode(0o555);
        fs::set_permissions(dir.path(), permissions).unwrap();
        handle_settings_change(&path, &provider, &state, &noop_persist());

        // 失败路径不得提交新快照，因此恢复目录权限后同一事件仍会重试写入。
        assert_eq!(*state.lock().unwrap(), previous);
        let provider_guard = provider.lock().unwrap();
        assert_eq!(
            provider_guard.settings_config["autoSyncState"]["lastWritten"],
            json!({ "ACW": "1", "MAX": "2" })
        );
        drop(provider_guard);

        let mut permissions = fs::metadata(dir.path()).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(dir.path(), permissions).unwrap();
        handle_settings_change(&path, &provider, &state, &noop_persist());

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "28500");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "30000");
        assert_eq!(
            *state.lock().unwrap(),
            Some(WatcherSnapshot {
                model: Some("haiku".to_string()),
                acw: Some("28500".to_string()),
                max: Some("30000".to_string()),
            })
        );
    }

    #[test]
    fn verify_file_unchanged_detects_concurrent_modification() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, "first").unwrap();
        assert!(verify_file_unchanged(&path, "first").is_ok());
        std::fs::write(&path, "second").unwrap();
        assert!(verify_file_unchanged(&path, "first").is_err());
    }

    #[test]
    fn effective_auto_sync_enabled_false_without_field_or_ledger() {
        let provider =
            Provider::with_id("p".to_string(), "P".to_string(), json!({ "env": {} }), None);
        assert!(!effective_auto_sync_enabled(&provider));
    }

    #[test]
    fn effective_auto_sync_enabled_respects_explicit_field() {
        // 显式字段优先：即使账本有记录，显式 false 仍为关闭
        let provider = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {},
                "autoSyncContextWindow": false,
                "autoSyncState": { "staticInjected": { "ACW": "262144", "MAX": "262144" } }
            }),
            None,
        );
        assert!(!effective_auto_sync_enabled(&provider));

        let provider = Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({ "env": {}, "autoSyncContextWindow": true }),
            None,
        );
        assert!(effective_auto_sync_enabled(&provider));
    }

    #[test]
    #[serial]
    fn fs_real_watcher_missing_field_does_not_auto_sync() {
        // 方案 A：autoSyncContextWindow 字段缺失即关闭（即使账本有记录），
        // watcher 不写 ACW/MAX。
        use std::fs;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let initial = json!({
            "model": "sonnet",
            "env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3[1M]",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]"
            }
        });
        fs::write(&path, taken_over_live(initial).to_string()).unwrap();

        let provider = Arc::new(Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": {
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3[1M]",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]"
                },
                "autoSyncState": { "lastWritten": { "ACW": "1", "MAX": "2" } }
            }),
            None,
        )));

        let watcher =
            spawn_claude_settings_watcher(path.clone(), provider, noop_persist()).unwrap();

        // 模拟外部程序修改 model 字段
        fs::write(
            &path,
            taken_over_live(json!({
                "model": "haiku",
                "env": {
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "Kimi-K2.7-Code[30k]",
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M3[1M]"
                }
            }))
            .to_string(),
        )
        .unwrap();

        thread::sleep(Duration::from_millis(800));

        let content = fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["model"], "haiku");
        assert!(
            v["env"].get("CLAUDE_CODE_AUTO_COMPACT_WINDOW").is_none(),
            "字段缺失时 watcher 不应写 ACW"
        );
        assert!(
            v["env"].get("CLAUDE_CODE_MAX_CONTEXT_TOKENS").is_none(),
            "字段缺失时 watcher 不应写 MAX"
        );

        drop(watcher);
    }

    #[test]
    fn handle_settings_change_route_mode_gate_controls_acw_max_writes() {
        // I1：直连（live 无接管占位符）时即使 autoSyncContextWindow=true，
        // watcher 也不写 ACW/MAX；接管态（占位符）则按激活角色写入。
        use std::fs;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");

        let provider = Mutex::new(Provider::with_id(
            "p".to_string(),
            "P".to_string(),
            json!({
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2" },
                "contextWindows": { "ANTHROPIC_DEFAULT_SONNET_MODEL": 200000 },
                "autoSyncContextWindow": true
            }),
            None,
        ));
        let state = Mutex::new(None);

        // 直连：live 无接管占位符 → 门早退，不写 ACW/MAX。
        fs::write(
            &path,
            json!({
                "model": "sonnet",
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2" }
            })
            .to_string(),
        )
        .unwrap();
        handle_settings_change(&path, &provider, &state, &noop_persist());

        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert!(v["env"].get("CLAUDE_CODE_AUTO_COMPACT_WINDOW").is_none());
        assert!(v["env"].get("CLAUDE_CODE_MAX_CONTEXT_TOKENS").is_none());

        // 接管态：live 含 ANTHROPIC_AUTH_TOKEN=PROXY_MANAGED → 按 sonnet 窗口写入。
        fs::write(
            &path,
            taken_over_live(json!({
                "model": "sonnet",
                "env": { "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2" }
            }))
            .to_string(),
        )
        .unwrap();
        handle_settings_change(&path, &provider, &state, &noop_persist());

        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "190000");
        assert_eq!(v["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "200000");
    }
}
