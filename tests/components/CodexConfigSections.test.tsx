import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CodexConfigSection } from "@/components/providers/forms/CodexConfigSections";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        "codexConfig.contextWindow": "上下文大小",
        "codexConfig.autoCompactLimit": "自动压缩阈值",
      })[key] ?? key,
  }),
}));

vi.mock("@/components/JsonEditor", () => ({
  default: () => null,
}));

function renderSection(value = "", onChange = vi.fn()) {
  render(
    <CodexConfigSection
      value={value}
      onChange={onChange}
      useCommonConfig={false}
      onCommonConfigToggle={vi.fn()}
      onEditCommonConfig={vi.fn()}
    />,
  );
  return onChange;
}

describe("CodexConfigSection top-level integer fields", () => {
  it.each([
    {
      label: "上下文大小",
      field: "model_context_window",
    },
    {
      label: "自动压缩阈值",
      field: "model_auto_compact_token_limit",
    },
  ])("填写$label后写入$field", ({ label, field }) => {
    const onChange = renderSection();

    fireEvent.change(screen.getByLabelText(label), {
      target: { value: "200000" },
    });

    expect(onChange).toHaveBeenCalledWith(
      expect.stringContaining(`${field} = 200000`),
    );
  });

  it.each([
    {
      label: "上下文大小",
    },
    {
      label: "自动压缩阈值",
    },
  ])("$label输入框只接受数字", ({ label }) => {
    renderSection();

    const input = screen.getByLabelText(label) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "12abc" } });

    // 不再硬剥非数字：非法 token 视为无效输入，字段被移除。
    expect(input).toHaveValue("");
  });

  it.each([
    {
      label: "上下文大小",
      field: "model_context_window",
    },
    {
      label: "自动压缩阈值",
      field: "model_auto_compact_token_limit",
    },
  ])("清空$label后移除$field", ({ label, field }) => {
    const onChange = renderSection(`${field} = 200000\n`);

    fireEvent.change(screen.getByLabelText(label), { target: { value: "" } });

    const updated = onChange.mock.calls[0][0] as string;
    expect(updated).not.toMatch(new RegExp(`^${field}\\s*=`, "m"));
  });

  it("输入 500k 写入 model_context_window = 500000", () => {
    const onChange = renderSection();

    fireEvent.change(screen.getByLabelText("上下文大小"), {
      target: { value: "500k" },
    });

    expect(onChange).toHaveBeenCalledWith(
      expect.stringContaining("model_context_window = 500000"),
    );
  });

  it("输入 2m 写入 model_auto_compact_token_limit = 2000000", () => {
    const onChange = renderSection();

    fireEvent.change(screen.getByLabelText("自动压缩阈值"), {
      target: { value: "2m" },
    });

    expect(onChange).toHaveBeenCalledWith(
      expect.stringContaining("model_auto_compact_token_limit = 2000000"),
    );
  });
});
