const CLAUDE_CONTEXT_WINDOW_ENV_KEYS = [
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
  "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
] as const;

const CLAUDE_CONTEXT_WINDOW_STATE_KEYS = {
  CLAUDE_CODE_AUTO_COMPACT_WINDOW: "ACW",
  CLAUDE_CODE_MAX_CONTEXT_TOKENS: "MAX",
} as const;

const AUTO_SYNC_COMPACT_RATIO_MIN = 0.2;
const AUTO_SYNC_COMPACT_RATIO_MAX = 1;

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
  envKey: string,
  liveValue: unknown,
): boolean {
  return ["lastWritten", "staticInjected"].some((sourceName) => {
    const source = state[sourceName];
    if (!isRecord(source)) return false;
    return [shortKey, envKey].some((key) => {
      const sourceValue = source[key];
      return (
        sourceValue !== undefined &&
        sourceValue !== null &&
        sourceValue !== false &&
        String(sourceValue) === String(liveValue)
      );
    });
  });
}

function hasUserExplicitValue(
  state: Record<string, unknown>,
  shortKey: string,
  envKey: string,
): boolean {
  const userExplicit = state.userExplicit;
  if (!isRecord(userExplicit)) return false;

  const shortValue = userExplicit[shortKey];
  if (shortValue !== undefined && shortValue !== null) return true;

  const legacyValue = userExplicit[envKey];
  return (
    legacyValue !== undefined && legacyValue !== null && legacyValue !== false
  );
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
 * 开启时只写开关状态，不自动补回字段，等 watcher 在终端切换模型时再写。
 */
export function applyAutoSyncContextWindowSetting(
  config: string,
  enabled: boolean,
): string {
  try {
    const parsed = JSON.parse(config || "{}") as Record<string, unknown>;
    parsed.autoSyncContextWindow = enabled;
    if (enabled) {
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
        if (!hasUserExplicitValue(state, shortKey, envKey)) {
          if (matchesAutoSyncSource(state, shortKey, envKey, liveValue)) {
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
 * 数字必须在 0.2 ~ 1 之间，调用方应已校验。
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
