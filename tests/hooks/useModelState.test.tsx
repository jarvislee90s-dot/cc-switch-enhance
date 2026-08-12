import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  parseModelSuffix,
  hasClaudeOneMMarker,
  setClaudeOneMMarker,
  stripClaudeOneMMarker,
  setModelSuffix,
  reapplySuffix,
  readContextWindows,
  writeContextWindow,
  stripModelSuffix,
  useModelState,
  parseWindowToken,
} from "@/components/providers/forms/hooks/useModelState";

describe("useModelState", () => {
  it("hydrates role models and display names from Claude Code env", () => {
    const settingsConfig = JSON.stringify({
      env: {
        ANTHROPIC_MODEL: "fallback-model",
        ANTHROPIC_SMALL_FAST_MODEL: "legacy-small",
        ANTHROPIC_DEFAULT_SONNET_MODEL: "deepseek-v4-pro",
        ANTHROPIC_DEFAULT_SONNET_MODEL_NAME: "DeepSeek V4 Pro",
        ANTHROPIC_DEFAULT_OPUS_MODEL: "kimi-k2",
        ANTHROPIC_DEFAULT_OPUS_MODEL_NAME: "Kimi K2",
        CLAUDE_CODE_SUBAGENT_MODEL: "subagent-model[1M]",
      },
    });

    const { result } = renderHook(() =>
      useModelState({
        settingsConfig,
        onConfigChange: vi.fn(),
      }),
    );

    expect(result.current.claudeModel).toBe("fallback-model");
    expect(result.current.defaultSonnetModel).toBe("deepseek-v4-pro");
    expect(result.current.defaultSonnetModelName).toBe("DeepSeek V4 Pro");
    expect(result.current.defaultOpusModel).toBe("kimi-k2");
    expect(result.current.defaultOpusModelName).toBe("Kimi K2");
    expect(result.current.defaultHaikuModel).toBe("legacy-small");
    expect(result.current.defaultHaikuModelName).toBe("legacy-small");
    expect(result.current.subagentModel).toBe("subagent-model[1M]");
  });

  it("writes and clears role display-name env fields without changing model mapping", () => {
    let latestConfig = JSON.stringify({
      env: {
        ANTHROPIC_DEFAULT_SONNET_MODEL: "deepseek-v4-pro",
      },
    });
    const onConfigChange = vi.fn((config: string) => {
      latestConfig = config;
    });

    const { result } = renderHook(() =>
      useModelState({
        settingsConfig: latestConfig,
        onConfigChange,
      }),
    );

    act(() => {
      result.current.handleModelChange(
        "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
        "DeepSeek V4 Pro",
      );
    });

    let env = JSON.parse(latestConfig).env;
    expect(env.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe("deepseek-v4-pro");
    expect(env.ANTHROPIC_DEFAULT_SONNET_MODEL_NAME).toBe("DeepSeek V4 Pro");

    act(() => {
      result.current.handleModelChange(
        "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
        "",
      );
    });

    env = JSON.parse(latestConfig).env;
    expect(env.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe("deepseek-v4-pro");
    expect(env.ANTHROPIC_DEFAULT_SONNET_MODEL_NAME).toBeUndefined();
  });

  it("keeps the 1M marker on request models but strips it from fallback display names", () => {
    const settingsConfig = JSON.stringify({
      env: {
        ANTHROPIC_DEFAULT_SONNET_MODEL: "deepseek-v4-pro[1M]",
      },
    });

    const { result } = renderHook(() =>
      useModelState({
        settingsConfig,
        onConfigChange: vi.fn(),
      }),
    );

    expect(result.current.defaultSonnetModel).toBe("deepseek-v4-pro[1M]");
    expect(result.current.defaultSonnetModelName).toBe("deepseek-v4-pro");
  });

  it("writes clean Claude Code subagent model env and strips legacy suffix", () => {
    let latestConfig = JSON.stringify({
      env: {
        ANTHROPIC_MODEL: "fallback-model",
      },
    });
    const onConfigChange = vi.fn((config: string) => {
      latestConfig = config;
    });

    const { result } = renderHook(() =>
      useModelState({
        settingsConfig: latestConfig,
        onConfigChange,
      }),
    );

    act(() => {
      result.current.handleModelChange(
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "subagent-model[1M]",
      );
    });

    let env = JSON.parse(latestConfig).env;
    expect(env.ANTHROPIC_MODEL).toBe("fallback-model");
    expect(env.CLAUDE_CODE_SUBAGENT_MODEL).toBe("subagent-model");

    act(() => {
      result.current.handleModelChange("CLAUDE_CODE_SUBAGENT_MODEL", "");
    });

    env = JSON.parse(latestConfig).env;
    expect(env.CLAUDE_CODE_SUBAGENT_MODEL).toBeUndefined();
  });

  it("normalizes Claude Code 1M markers for UI toggles", () => {
    expect(hasClaudeOneMMarker("deepseek-v4-pro[1m]")).toBe(true);
    expect(hasClaudeOneMMarker("deepseek-v4-pro [1M]  ")).toBe(true);
    expect(stripClaudeOneMMarker("deepseek-v4-pro [1M]  ")).toBe(
      "deepseek-v4-pro",
    );
    expect(setClaudeOneMMarker("deepseek-v4-pro [1M]", false)).toBe(
      "deepseek-v4-pro",
    );
    expect(setClaudeOneMMarker("deepseek-v4-pro", true)).toBe(
      "deepseek-v4-pro[1M]",
    );
  });
});

describe("parseWindowToken", () => {
  it.each([
    ["200K", 200000],
    ["200k", 200000],
    ["1M", 1000000],
    ["1m", 1000000],
    ["128000", 128000],
  ])("接受 %s", (input, expected) => {
    expect(parseWindowToken(input)).toBe(expected);
  });

  it.each([
    "1e3",
    "1foo2",
    "12abc34",
    "1.5M",
    "1,000,000",
    "1_000_000",
    "1 000 000",
    "0",
    "1G",
  ])("拒绝 %s", (input) => {
    expect(parseWindowToken(input)).toBeUndefined();
  });
});

describe("parseModelSuffix", () => {
  it("parses [1m] suffix", () => {
    expect(parseModelSuffix("deepseek-v4-pro[1m]")).toEqual({
      slug: "deepseek-v4-pro",
      window: 1000000,
    });
  });

  it("parses [200k] suffix", () => {
    expect(parseModelSuffix("glm-5.2[200k]")).toEqual({
      slug: "glm-5.2",
      window: 200000,
    });
  });

  it("parses uppercase [500K]", () => {
    expect(parseModelSuffix("model[500K]")).toEqual({
      slug: "model",
      window: 500000,
    });
  });

  it("parses pure number [1000000]", () => {
    expect(parseModelSuffix("model[1000000]")).toEqual({
      slug: "model",
      window: 1000000,
    });
  });

  it("returns undefined window for no suffix", () => {
    expect(parseModelSuffix("model")).toEqual({
      slug: "model",
      window: undefined,
    });
  });

  it("does not strip invalid suffix", () => {
    expect(parseModelSuffix("model[invalid]")).toEqual({
      slug: "model[invalid]",
      window: undefined,
    });
  });

  it("parses lowercase [1m]", () => {
    expect(parseModelSuffix("model[1m]")).toEqual({
      slug: "model",
      window: 1000000,
    });
  });

  it("parses [128k] suffix", () => {
    expect(parseModelSuffix("model[128k]")).toEqual({
      slug: "model",
      window: 128000,
    });
  });

  it("rejects [0] as invalid window", () => {
    expect(parseModelSuffix("model[0]")).toEqual({
      slug: "model[0]",
      window: undefined,
    });
  });

  it("rejects [1.5m] decimal", () => {
    expect(parseModelSuffix("model[1.5m]")).toEqual({
      slug: "model[1.5m]",
      window: undefined,
    });
  });

  it.each([
    "1e3",
    "1foo2",
    "12abc34",
    "1.5M",
    "1,000,000",
    "1_000_000",
    "1 000 000",
    "0",
    "1G",
  ])("rejects [%s] as invalid suffix", (input) => {
    expect(parseModelSuffix(`model[${input}]`)).toEqual({
      slug: `model[${input}]`,
      window: undefined,
    });
  });
});

describe("setModelSuffix", () => {
  it("appends lowercase suffix", () => {
    expect(setModelSuffix("model", "1M")).toBe("model[1m]");
  });

  it("clears suffix when empty", () => {
    expect(setModelSuffix("model[1m]", "")).toBe("model");
  });

  it("replaces existing suffix", () => {
    expect(setModelSuffix("model[1m]", "200K")).toBe("model[200k]");
  });

  it("writes lowercase [200k] from 200K", () => {
    expect(setModelSuffix("model", "200K")).toBe("model[200k]");
  });

  it("writes lowercase [128k] from 128K", () => {
    expect(setModelSuffix("model", "128K")).toBe("model[128k]");
  });

  it("writes pure number [1000000]", () => {
    expect(setModelSuffix("model", "1000000")).toBe("model[1000000]");
  });

  it("returns empty string for empty base", () => {
    expect(setModelSuffix("", "1M")).toBe("");
  });

  it("returns base unchanged for invalid input abc", () => {
    expect(setModelSuffix("model", "abc")).toBe("model");
  });

  it("returns base unchanged for unsupported unit 1G", () => {
    expect(setModelSuffix("model", "1G")).toBe("model");
  });

  it.each([
    "1e3",
    "1foo2",
    "12abc34",
    "1.5M",
    "1,000,000",
    "1_000_000",
    "1 000 000",
    "0",
    "1G",
  ])("returns base unchanged for invalid input %s", (input) => {
    expect(setModelSuffix("model", input)).toBe("model");
  });
});

describe("setModelSuffix - 多元化输入", () => {
  it.each([
    "[30k]",
    "[30",
    "30k]",
    "1,000,000",
    "1_000_000",
    "1 000 000",
    "1.5M",
    "0.5M",
    "[1,000k]",
  ])("rejects %s without suffix", (input) => {
    expect(setModelSuffix("model", input)).toBe("model");
  });
});

describe("stripModelSuffix", () => {
  it("strips [200k]", () => {
    expect(stripModelSuffix("model[200k]")).toBe("model");
  });
});

describe("reapplySuffix", () => {
  it("preserves suffix when changing model name", () => {
    expect(reapplySuffix("deepseek-v4-pro[200k]", "glm-5.2")).toBe(
      "glm-5.2[200k]",
    );
  });

  it("returns base unchanged when old model has no suffix", () => {
    expect(reapplySuffix("deepseek-v4-pro", "glm-5.2")).toBe("glm-5.2");
  });

  it("returns empty string when new input is empty", () => {
    expect(reapplySuffix("deepseek-v4-pro[200k]", "")).toBe("");
  });

  it("old suffix wins when new input also has a suffix", () => {
    expect(reapplySuffix("deepseek-v4-pro[200k]", "glm-5.2[100k]")).toBe(
      "glm-5.2[200k]",
    );
  });
});

describe("contextWindows", () => {
  it("reads contextWindows from settings config", () => {
    const config = JSON.stringify({
      env: { ANTHROPIC_DEFAULT_SONNET_MODEL: "glm-5.2" },
      contextWindows: { ANTHROPIC_DEFAULT_SONNET_MODEL: 200000 },
    });

    expect(readContextWindows(config)).toEqual({
      ANTHROPIC_DEFAULT_SONNET_MODEL: 200000,
    });
  });

  it("returns empty object for missing or invalid config", () => {
    expect(readContextWindows("")).toEqual({});
    expect(readContextWindows("{invalid")).toEqual({});
    expect(readContextWindows(JSON.stringify({ env: {} }))).toEqual({});
  });

  it("returns empty object for non-object contextWindows", () => {
    expect(readContextWindows(JSON.stringify({ contextWindows: [] }))).toEqual(
      {},
    );
    expect(
      readContextWindows(JSON.stringify({ contextWindows: "bad" })),
    ).toEqual({});
  });

  it("filters non-finite and non-positive contextWindows values", () => {
    const config = JSON.stringify({
      contextWindows: {
        ANTHROPIC_MODEL: 200000,
        ANTHROPIC_DEFAULT_HAIKU_MODEL: 0,
        ANTHROPIC_DEFAULT_SONNET_MODEL: -1,
        ANTHROPIC_DEFAULT_OPUS_MODEL: Number.POSITIVE_INFINITY,
        ANTHROPIC_DEFAULT_FABLE_MODEL: Number.NaN,
        CLAUDE_CODE_SUBAGENT_MODEL: "1000000",
        ANTHROPIC_DEFAULT_FABLE_MODEL_NAME: null,
      },
    });

    expect(readContextWindows(config)).toEqual({ ANTHROPIC_MODEL: 200000 });
  });

  it("窗口写入 contextWindows 且模型名不带后缀", () => {
    const config = JSON.stringify({
      env: { ANTHROPIC_DEFAULT_SONNET_MODEL: "glm-5.2" },
    });
    const next = writeContextWindow(
      config,
      "ANTHROPIC_DEFAULT_SONNET_MODEL",
      200000,
    );
    const parsed = JSON.parse(next);
    expect(parsed.env.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe("glm-5.2");
    expect(parsed.contextWindows.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe(200000);
  });

  it("写入窗口时剥离 env 中的旧后缀", () => {
    const config = JSON.stringify({
      env: { ANTHROPIC_DEFAULT_SONNET_MODEL: "model[1M]" },
    });
    const next = writeContextWindow(
      config,
      "ANTHROPIC_DEFAULT_SONNET_MODEL",
      200000,
    );
    const parsed = JSON.parse(next);
    expect(parsed.env.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe("model");
    expect(parsed.contextWindows.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe(200000);
  });

  it("清空窗口时不写 contextWindows 也不写后缀", () => {
    const config = JSON.stringify({
      env: { ANTHROPIC_DEFAULT_SONNET_MODEL: "glm-5.2[1M]" },
      contextWindows: { ANTHROPIC_DEFAULT_SONNET_MODEL: 1000000 },
    });
    const next = writeContextWindow(
      config,
      "ANTHROPIC_DEFAULT_SONNET_MODEL",
      null,
    );
    const parsed = JSON.parse(next);
    expect(
      parsed.contextWindows.ANTHROPIC_DEFAULT_SONNET_MODEL,
    ).toBeUndefined();
    expect(parsed.env.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe("glm-5.2");
  });

  it("writeContextWindow 对无效 JSON 不抛错并返回原 config", () => {
    const invalid = "{invalid";
    expect(() =>
      writeContextWindow(invalid, "ANTHROPIC_DEFAULT_SONNET_MODEL", 200000),
    ).not.toThrow();
    expect(
      writeContextWindow(invalid, "ANTHROPIC_DEFAULT_SONNET_MODEL", 200000),
    ).toBe(invalid);
  });

  it("writeContextWindow 会替换非对象 contextWindows", () => {
    const next = writeContextWindow(
      JSON.stringify({ contextWindows: [] }),
      "ANTHROPIC_DEFAULT_SONNET_MODEL",
      200000,
    );
    const parsed = JSON.parse(next);
    expect(Array.isArray(parsed.contextWindows)).toBe(false);
    expect(parsed.contextWindows.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe(200000);
  });
});
