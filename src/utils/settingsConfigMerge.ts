const CLAUDE_CONTEXT_WINDOW_ENV_KEYS = [
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
  "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
] as const;

const CLAUDE_CONTEXT_WINDOW_STATE_KEYS = {
  CLAUDE_CODE_AUTO_COMPACT_WINDOW: "ACW",
  CLAUDE_CODE_MAX_CONTEXT_TOKENS: "MAX",
} as const;

const AUTO_SYNC_COMPACT_RATIO_MIN = 0.2;
const AUTO_SYNC_COMPACT_RATIO_MAX = 0.95;

/**
 * 解析 autoSyncContextWindow 有效值：显式字段优先；字段缺失时以
 * autoSyncState 账本为主信号（lastWritten/staticInjected 有 ACW/MAX 记录 →
 * 视为开启），辅以 env ACW/MAX 命中 contextWindows 推导目标进一步确认，
 * 衔接用户更新前的状态，避免升级后行为突变。
 */
export function resolveAutoSyncContextWindow(config: string): boolean {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(config || "{}") as Record<string, unknown>;
  } catch {
    return false;
  }
  if (typeof parsed.autoSyncContextWindow === "boolean") {
    return parsed.autoSyncContextWindow;
  }
  const state = parsed.autoSyncState;
  if (isRecord(state)) {
    for (const source of ["lastWritten", "staticInjected"] as const) {
      const src = state[source];
      if (
        isRecord(src) &&
        ["ACW", "MAX"].some(
          (shortKey) =>
            src[shortKey] !== undefined &&
            src[shortKey] !== null &&
            src[shortKey] !== false,
        )
      ) {
        return true;
      }
    }
  }
  const env = parsed.env;
  const contextWindows = parsed.contextWindows;
  if (isRecord(env) && isRecord(contextWindows)) {
    const ratio =
      typeof parsed.autoSyncCompactRatio === "number" &&
      Number.isFinite(parsed.autoSyncCompactRatio) &&
      parsed.autoSyncCompactRatio >= AUTO_SYNC_COMPACT_RATIO_MIN &&
      parsed.autoSyncCompactRatio <= AUTO_SYNC_COMPACT_RATIO_MAX
        ? parsed.autoSyncCompactRatio
        : 0.95;
    const targets = new Set<string>();
    for (const value of Object.values(contextWindows)) {
      if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
        continue;
      }
      targets.add(String(Math.floor(value * ratio)));
      targets.add(String(value));
    }
    for (const envKey of CLAUDE_CONTEXT_WINDOW_ENV_KEYS) {
      const liveValue = env[envKey];
      if (typeof liveValue === "string" && targets.has(liveValue)) {
        return true;
      }
    }
  }
  return false;
}

function removeClaudeContextWindowEnvFields(config: Record<string, unknown>) {
  const env = config.env;
  if (!env || typeof env !== "object" || Array.isArray(env)) return;

  const envRecord = env as Record<string, unknown>;
  for (const key of CLAUDE_CONTEXT_WINDOW_ENV_KEYS) {
    delete envRecord[key];
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function matchesAutoSyncSource(
  state: Record<string, unknown>,
  shortKey: string,
  liveValue: unknown,
): boolean {
  return ["lastWritten", "staticInjected"].some((sourceName) => {
    const source = state[sourceName];
    if (!isRecord(source)) return false;
    const sourceValue = source[shortKey];
    return (
      sourceValue !== undefined &&
      sourceValue !== null &&
      sourceValue !== false &&
      String(sourceValue) === String(liveValue)
    );
  });
}

function hasUserExplicitValue(
  state: Record<string, unknown>,
  shortKey: string,
): boolean {
  const userExplicit = state.userExplicit;
  if (!isRecord(userExplicit)) return false;

  const shortValue = userExplicit[shortKey];
  return shortValue !== undefined && shortValue !== null;
}

function restorableLedgerValue(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return String(value);
  }
  return undefined;
}

function restoreAutoSyncContextWindowValue(
  state: Record<string, unknown>,
  shortKey: string,
): string | undefined {
  const userExplicit = state.userExplicit;
  if (isRecord(userExplicit)) {
    const shortValue = restorableLedgerValue(userExplicit[shortKey]);
    if (shortValue !== undefined) return shortValue;
  }
  for (const sourceName of ["lastWritten", "staticInjected"] as const) {
    const source = state[sourceName];
    if (!isRecord(source)) continue;
    const value = restorableLedgerValue(source[shortKey]);
    if (value !== undefined) return value;
  }
  return undefined;
}

function writeUserExplicitState(
  state: Record<string, unknown>,
  shortKey: string,
  liveValue: unknown,
) {
  if (!isRecord(state.userExplicit)) {
    state.userExplicit = {};
  }
  const userExplicit = state.userExplicit as Record<string, unknown>;
  userExplicit[shortKey] = liveValue;
}

/**
 * 写入自动同步开关状态。
 *
 * 关闭时按 autoSyncState 清理自动写入的 ACW/MAX；live 中非自动来源的值
 * 视为关闭瞬间用户写入，保留并记入 userExplicit。无账本时沿用旧行为删除。
 * 开启时优先从 userExplicit 恢复 ACW/MAX；没有显式值时从
 * lastWritten/staticInjected 恢复；都没有则不写回。
 */
export function applyAutoSyncContextWindowSetting(
  config: string,
  enabled: boolean,
): string {
  try {
    const parsed = JSON.parse(config || "{}") as Record<string, unknown>;
    parsed.autoSyncContextWindow = enabled;
    if (enabled) {
      const state = parsed.autoSyncState;
      if (isRecord(state)) {
        const writes: Array<[string, string]> = [];
        for (const envKey of CLAUDE_CONTEXT_WINDOW_ENV_KEYS) {
          const shortKey = CLAUDE_CONTEXT_WINDOW_STATE_KEYS[envKey];
          const value = restoreAutoSyncContextWindowValue(state, shortKey);
          if (value !== undefined) writes.push([envKey, value]);
        }
        if (writes.length > 0) {
          if (!isRecord(parsed.env)) parsed.env = {};
          const env = parsed.env as Record<string, unknown>;
          for (const [envKey, value] of writes) env[envKey] = value;
        }
      }
      return JSON.stringify(parsed, null, 2);
    }

    const state = parsed.autoSyncState;
    if (!isRecord(state)) {
      removeClaudeContextWindowEnvFields(parsed);
      return JSON.stringify(parsed, null, 2);
    }

    const env = parsed.env;
    if (isRecord(env)) {
      for (const envKey of CLAUDE_CONTEXT_WINDOW_ENV_KEYS) {
        const liveValue = env[envKey];
        if (liveValue === undefined) continue;
        const shortKey = CLAUDE_CONTEXT_WINDOW_STATE_KEYS[envKey];
        if (!hasUserExplicitValue(state, shortKey)) {
          if (matchesAutoSyncSource(state, shortKey, liveValue)) {
            delete env[envKey];
            continue;
          }
        }
        writeUserExplicitState(state, shortKey, liveValue);
      }
    }
    return JSON.stringify(parsed, null, 2);
  } catch {
    return config;
  }
}

/**
 * 写入或清空自动压缩比例。
 *
 * null 表示用户留空，不持久化字段，watcher 按 1 处理；
 * 数字必须在 0.2 ~ 0.95 之间，调用方应已校验。
 */
export function applyAutoSyncCompactRatioSetting(
  config: string,
  ratio: number | null,
): string {
  try {
    const parsed = JSON.parse(config || "{}") as Record<string, unknown>;
    if (ratio === null) {
      delete parsed.autoSyncCompactRatio;
    } else {
      parsed.autoSyncCompactRatio = ratio;
    }
    return JSON.stringify(parsed, null, 2);
  } catch {
    return config;
  }
}

/**
 * 合并 settings.json 配置字符串时，保留当前表单中的 autoSyncContextWindow
 * 和 autoSyncCompactRatio。
 *
 * ClaudeFormFields 的开关、比例和模型/API Key/Base URL 等输入都会整包更新
 * settingsConfig。react-hook-form 的 setValue 默认不触发重渲染，后续输入
 * 可能基于旧配置重建 JSON，把刚改的自动同步状态盖掉。这里以当前表单值为准，
 * 只要当前有显式值，就把它强制写回新配置；当前缺失时自动写入不得补写该字段。
 * ACW/MAX 的清理只发生在开关关闭动作中，这里不重复删除。
 */
export function mergeSettingsConfigPreservingAutoSync(
  currentConfig: string,
  nextConfig: string,
): string {
  try {
    const current = JSON.parse(currentConfig || "{}") as Record<
      string,
      unknown
    >;
    const next = JSON.parse(nextConfig || "{}") as Record<string, unknown>;

    if (typeof current.autoSyncContextWindow === "boolean") {
      next.autoSyncContextWindow = current.autoSyncContextWindow;
    } else {
      delete next.autoSyncContextWindow;
    }

    if (
      typeof current.autoSyncCompactRatio === "number" &&
      Number.isFinite(current.autoSyncCompactRatio) &&
      current.autoSyncCompactRatio >= AUTO_SYNC_COMPACT_RATIO_MIN &&
      current.autoSyncCompactRatio <= AUTO_SYNC_COMPACT_RATIO_MAX
    ) {
      next.autoSyncCompactRatio = current.autoSyncCompactRatio;
    } else {
      delete next.autoSyncCompactRatio;
    }

    return JSON.stringify(next, null, 2);
  } catch {
    return nextConfig;
  }
}
