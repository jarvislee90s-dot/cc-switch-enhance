import ReactDOM from "react-dom";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Provider } from "@/types";

const apiMocks = vi.hoisted(() => ({
  getCurrent: vi.fn(),
  getLiveProviderSettings: vi.fn(),
  getOpenClawLiveProvider: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  providersApi: {
    getCurrent: apiMocks.getCurrent,
  },
  vscodeApi: {
    getLiveProviderSettings: apiMocks.getLiveProviderSettings,
  },
  openclawApi: {
    getLiveProvider: apiMocks.getOpenClawLiveProvider,
  },
}));

vi.mock("@/components/common/FullScreenPanel", () => ({
  FullScreenPanel: ({
    isOpen,
    children,
    footer,
  }: {
    isOpen: boolean;
    children: React.ReactNode;
    footer?: React.ReactNode;
  }) =>
    isOpen ? (
      <div>
        <div>{children}</div>
        <div>{footer}</div>
      </div>
    ) : null,
}));

vi.mock("@/components/providers/forms/ProviderForm", () => ({
  ProviderForm: ({
    initialData,
    onSubmit,
    isProxyTakeover,
  }: {
    initialData: {
      name?: string;
      websiteUrl?: string;
      notes?: string;
      settingsConfig?: Record<string, unknown>;
      meta?: Record<string, unknown>;
      icon?: string;
      iconColor?: string;
    };
    onSubmit: (values: {
      name: string;
      websiteUrl: string;
      notes?: string;
      settingsConfig: string;
      meta?: Record<string, unknown>;
      icon?: string;
      iconColor?: string;
    }) => void;
    isProxyTakeover?: boolean;
  }) => (
    <form
      id="provider-form"
      data-testid="provider-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit({
          name: initialData.name ?? "",
          websiteUrl: initialData.websiteUrl ?? "",
          notes: initialData.notes,
          settingsConfig: JSON.stringify(initialData.settingsConfig ?? {}),
          meta: initialData.meta,
          icon: initialData.icon,
          iconColor: initialData.iconColor,
        });
      }}
    >
      <output data-testid="settings-config">
        {JSON.stringify(initialData.settingsConfig ?? {})}
      </output>
      <output data-testid="is-proxy-takeover">
        {isProxyTakeover ? "true" : "false"}
      </output>
    </form>
  ),
}));

import { EditProviderDialog } from "@/components/providers/EditProviderDialog";

describe("EditProviderDialog", () => {
  beforeEach(() => {
    apiMocks.getCurrent.mockReset();
    apiMocks.getLiveProviderSettings.mockReset();
    apiMocks.getOpenClawLiveProvider.mockReset();
  });

  it("保留 Codex 数据库中的 modelCatalog，避免 live 配置缺字段时清空模型映射", async () => {
    const dbModelCatalog = {
      models: [
        {
          model: "deepseek-v4-flash",
          displayName: "DeepSeek V4 Flash",
          contextWindow: 1000000,
        },
      ],
    };
    const provider: Provider = {
      id: "deepseek",
      name: "DeepSeek",
      category: "aggregator",
      settingsConfig: {
        auth: {
          OPENAI_API_KEY: "db-key",
        },
        config: 'model_provider = "custom"\nmodel = "deepseek-v4-flash"\n',
        modelCatalog: dbModelCatalog,
      },
    };
    const liveSettings = {
      auth: {
        OPENAI_API_KEY: "live-key",
      },
      config: 'model_provider = "custom"\nmodel = "deepseek-v4-pro"\n',
    };
    const handleSubmit = vi.fn().mockResolvedValue(undefined);

    apiMocks.getCurrent.mockResolvedValue(provider.id);
    apiMocks.getLiveProviderSettings.mockResolvedValue(liveSettings);

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={handleSubmit}
        appId="codex"
      />,
    );

    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("settings-config").textContent ?? "{}"),
      ).toEqual({
        ...liveSettings,
        modelCatalog: dbModelCatalog,
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "common.save" }));

    await waitFor(() => expect(handleSubmit).toHaveBeenCalledTimes(1));
    expect(handleSubmit.mock.calls[0][0].provider.settingsConfig).toEqual({
      ...liveSettings,
      modelCatalog: dbModelCatalog,
    });
  });

  it("代理接管中编辑 Codex 供应商时展示数据库配置而不是读取 live 代理配置", async () => {
    const provider: Provider = {
      id: "deepseek",
      name: "DeepSeek",
      category: "custom",
      settingsConfig: {
        auth: {
          OPENAI_API_KEY: "db-key",
        },
        config:
          'model_provider = "custom"\n[model_providers.custom]\nbase_url = "https://api.deepseek.com/v1"\n',
      },
    };

    apiMocks.getCurrent.mockResolvedValue(provider.id);
    apiMocks.getLiveProviderSettings.mockResolvedValue({
      auth: {
        OPENAI_API_KEY: "PROXY_MANAGED",
      },
      config:
        'model_provider = "custom"\n[model_providers.custom]\nbase_url = "http://127.0.0.1:15721/v1"\nexperimental_bearer_token = "PROXY_MANAGED"\n',
    });

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
        isProxyTakeover
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("is-proxy-takeover").textContent).toBe("true");
    });

    expect(apiMocks.getLiveProviderSettings).not.toHaveBeenCalled();
    expect(
      JSON.parse(screen.getByTestId("settings-config").textContent ?? "{}"),
    ).toEqual(provider.settingsConfig);
  });

  it("Claude live 配置缺少 contextWindows/autoSyncState 时保留数据库账本", async () => {
    const dbContextWindows = {
      ANTHROPIC_MODEL: 200000,
      CLAUDE_CODE_SUBAGENT_MODEL: 1000000,
    };
    const dbAutoSyncState = {
      lastWritten: { ACW: "160000", MAX: "200000" },
      userExplicit: { MAX: "250000" },
    };
    const provider: Provider = {
      id: "claude-db",
      name: "Claude DB",
      category: "aggregator",
      settingsConfig: {
        env: { ANTHROPIC_MODEL: "deepseek-v3" },
        contextWindows: dbContextWindows,
        autoSyncState: dbAutoSyncState,
      },
    };
    const liveSettings = { env: { ANTHROPIC_MODEL: "deepseek-v3" } };
    const handleSubmit = vi.fn().mockResolvedValue(undefined);

    apiMocks.getCurrent.mockResolvedValue(provider.id);
    apiMocks.getLiveProviderSettings.mockResolvedValue(liveSettings);

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={handleSubmit}
        appId="claude"
      />,
    );

    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("settings-config").textContent ?? "{}"),
      ).toEqual({
        ...liveSettings,
        contextWindows: dbContextWindows,
        autoSyncState: dbAutoSyncState,
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "common.save" }));

    await waitFor(() => expect(handleSubmit).toHaveBeenCalledTimes(1));
    expect(handleSubmit.mock.calls[0][0].provider.settingsConfig).toEqual({
      ...liveSettings,
      contextWindows: dbContextWindows,
      autoSyncState: dbAutoSyncState,
    });
  });

  it("Claude live 配置缺少 autoSyncContextWindow 时保留数据库开关状态", async () => {
    const provider: Provider = {
      id: "claude-proxy",
      name: "Claude Proxy",
      category: "aggregator",
      settingsConfig: {
        env: {
          ANTHROPIC_MODEL: "fallback-model[1M]",
        },
        autoSyncContextWindow: false,
        autoSyncCompactRatio: 0.8,
      },
    };
    const liveSettings = {
      env: {
        ANTHROPIC_MODEL: "fallback-model[1M]",
      },
    };
    const handleSubmit = vi.fn().mockResolvedValue(undefined);

    apiMocks.getCurrent.mockResolvedValue(provider.id);
    apiMocks.getLiveProviderSettings.mockResolvedValue(liveSettings);

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={handleSubmit}
        appId="claude"
      />,
    );

    await waitFor(() => {
      expect(apiMocks.getLiveProviderSettings).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("settings-config").textContent ?? "{}"),
      ).toEqual({
        ...liveSettings,
        autoSyncContextWindow: false,
        autoSyncCompactRatio: 0.8,
      });
    });
  });

  it("live 配置加载完成前不渲染表单，加载完成后出现", async () => {
    const provider: Provider = {
      id: "deepseek",
      name: "DeepSeek",
      category: "aggregator",
      settingsConfig: {
        auth: { OPENAI_API_KEY: "db-key" },
      },
    };
    let resolveLive: (value: Record<string, unknown>) => void = () => {};
    const livePromise = new Promise<Record<string, unknown>>((resolve) => {
      resolveLive = resolve;
    });

    apiMocks.getCurrent.mockResolvedValue(provider.id);
    apiMocks.getLiveProviderSettings.mockReturnValue(livePromise);

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
      />,
    );

    expect(screen.queryByTestId("provider-form")).not.toBeInTheDocument();

    resolveLive({});
    await waitFor(() =>
      expect(screen.getByTestId("provider-form")).toBeInTheDocument(),
    );
  });

  it("live 配置加载失败时禁用保存且不渲染表单", async () => {
    const provider: Provider = {
      id: "deepseek",
      name: "DeepSeek",
      category: "aggregator",
      settingsConfig: {
        auth: { OPENAI_API_KEY: "db-key" },
      },
    };

    apiMocks.getCurrent.mockResolvedValue(provider.id);
    apiMocks.getLiveProviderSettings.mockRejectedValue(new Error("boom"));

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
      />,
    );

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "common.save" }),
      ).toBeDisabled(),
    );
    expect(screen.queryByTestId("provider-form")).not.toBeInTheDocument();
    expect(screen.getByTestId("live-load-error")).toHaveTextContent(
      /读取实时配置失败/,
    );
  });

  it("OpenClaw live 配置加载失败时禁用保存且不渲染表单", async () => {
    const provider: Provider = {
      id: "openclaw-provider",
      name: "OpenClaw Provider",
      category: "custom",
      settingsConfig: { base_url: "https://db.example.com" },
    };

    apiMocks.getOpenClawLiveProvider.mockRejectedValue(new Error("boom"));

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="openclaw"
      />,
    );

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "common.save" }),
      ).toBeDisabled(),
    );
    expect(screen.queryByTestId("provider-form")).not.toBeInTheDocument();
    expect(screen.getByTestId("live-load-error")).toHaveTextContent(
      /读取实时配置失败/,
    );
  });

  it("providersApi.getCurrent 失败时禁用保存并显示 live 读取失败", async () => {
    const provider: Provider = {
      id: "deepseek",
      name: "DeepSeek",
      category: "aggregator",
      settingsConfig: {
        auth: { OPENAI_API_KEY: "db-key" },
      },
    };

    apiMocks.getCurrent.mockRejectedValue(new Error("boom"));

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("live-load-error")).toHaveTextContent(
        /读取实时配置失败/,
      );
    });
    expect(screen.getByRole("button", { name: "common.save" })).toBeDisabled();
    expect(screen.queryByTestId("provider-form")).not.toBeInTheDocument();
    expect(apiMocks.getLiveProviderSettings).not.toHaveBeenCalled();
  });

  it("切换 provider 后的首个 commit 不渲染旧 live 表单", async () => {
    const providerA: Provider = {
      id: "deepseek",
      name: "DeepSeek",
      category: "aggregator",
      settingsConfig: {
        auth: { OPENAI_API_KEY: "db-key-a" },
      },
    };
    const providerB: Provider = {
      id: "kimi",
      name: "Kimi",
      category: "aggregator",
      settingsConfig: {
        auth: { OPENAI_API_KEY: "db-key-b" },
      },
    };
    const liveA = { auth: { OPENAI_API_KEY: "live-key-a" } };
    const liveB = { auth: { OPENAI_API_KEY: "live-key-b" } };
    let resolveA: (value: Record<string, unknown>) => void = () => {};
    const livePromiseA = new Promise<Record<string, unknown>>((resolve) => {
      resolveA = resolve;
    });

    apiMocks.getCurrent
      .mockResolvedValueOnce(providerA.id)
      .mockResolvedValueOnce(providerB.id);
    apiMocks.getLiveProviderSettings
      .mockReturnValueOnce(livePromiseA)
      .mockReturnValueOnce(liveB);

    const view = render(
      <EditProviderDialog
        open
        provider={providerA}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
      />,
      { legacyRoot: true },
    );

    resolveA(liveA);
    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("settings-config").textContent ?? "{}"),
      ).toEqual(liveA);
    });

    // 直接 ReactDOM.render 绕开 RTL 的 act，让断言落在切换后的首个 commit 上
    ReactDOM.render(
      <EditProviderDialog
        open
        provider={providerB}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
      />,
      view.container,
    );

    expect(apiMocks.getCurrent).toHaveBeenCalledTimes(1);
    expect(screen.getByText("加载中...")).toBeInTheDocument();
    expect(screen.queryByTestId("provider-form")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "common.save" })).toBeDisabled();
  });

  it("同一会话切换 provider 时重新进入加载 gate，避免短暂用旧 live 配置渲染", async () => {
    const providerA: Provider = {
      id: "deepseek",
      name: "DeepSeek",
      category: "aggregator",
      settingsConfig: {
        auth: { OPENAI_API_KEY: "db-key-a" },
      },
    };
    const providerB: Provider = {
      id: "kimi",
      name: "Kimi",
      category: "aggregator",
      settingsConfig: {
        auth: { OPENAI_API_KEY: "db-key-b" },
      },
    };
    let resolveA: (value: Record<string, unknown>) => void = () => {};
    let resolveB: (value: Record<string, unknown>) => void = () => {};
    const liveA = new Promise<Record<string, unknown>>((resolve) => {
      resolveA = resolve;
    });
    const liveB = new Promise<Record<string, unknown>>((resolve) => {
      resolveB = resolve;
    });

    apiMocks.getCurrent
      .mockResolvedValueOnce(providerA.id)
      .mockResolvedValueOnce(providerB.id);
    apiMocks.getLiveProviderSettings
      .mockReturnValueOnce(liveA)
      .mockReturnValueOnce(liveB);

    const view = render(
      <EditProviderDialog
        open
        provider={providerA}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
      />,
    );

    resolveA({ auth: { OPENAI_API_KEY: "live-key-a" } });
    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("settings-config").textContent ?? "{}"),
      ).toEqual({ auth: { OPENAI_API_KEY: "live-key-a" } });
    });

    view.rerender(
      <EditProviderDialog
        open
        provider={providerB}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
      />,
    );
    expect(screen.getByText("加载中...")).toBeInTheDocument();
    expect(screen.queryByTestId("provider-form")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "common.save" })).toBeDisabled();

    resolveB({ auth: { OPENAI_API_KEY: "live-key-b" } });
    await waitFor(() => {
      expect(
        JSON.parse(screen.getByTestId("settings-config").textContent ?? "{}"),
      ).toEqual({ auth: { OPENAI_API_KEY: "live-key-b" } });
    });
  });

  it("live 配置不存在时仍使用数据库快照渲染并允许保存", async () => {
    const provider: Provider = {
      id: "deepseek",
      name: "DeepSeek",
      category: "aggregator",
      settingsConfig: {
        auth: { OPENAI_API_KEY: "db-key" },
      },
    };

    apiMocks.getCurrent.mockResolvedValue(provider.id);
    apiMocks.getLiveProviderSettings.mockResolvedValue(null);

    render(
      <EditProviderDialog
        open
        provider={provider}
        onOpenChange={vi.fn()}
        onSubmit={vi.fn()}
        appId="codex"
      />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("provider-form")).toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: "common.save" })).toBeEnabled();
    expect(
      JSON.parse(screen.getByTestId("settings-config").textContent ?? "{}"),
    ).toEqual(provider.settingsConfig);
  });
});
