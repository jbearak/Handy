import { test, expect, type Page } from "@playwright/test";

// Stand-in for the Tauri IPC the overlay webview uses. `get_app_settings`
// never resolves, so the overlay can only appear if showing it doesn't wait
// on Rust.
async function mockOverlayIpc(page: Page) {
  await page.addInitScript(() => {
    const callbacks = new Map<number, (message: unknown) => void>();
    const listeners = new Map<string, number[]>();
    let nextId = 1;
    const w = window as unknown as Record<string, unknown>;
    w.__TAURI_OS_PLUGIN_INTERNALS__ = { platform: "macos" };
    w.__TAURI_EVENT_PLUGIN_INTERNALS__ = { unregisterListener: () => {} };
    w.__TAURI_INTERNALS__ = {
      transformCallback: (cb: (message: unknown) => void) => {
        const id = nextId++;
        callbacks.set(id, cb);
        return id;
      },
      unregisterCallback: (id: number) => callbacks.delete(id),
      invoke: (cmd: string, args: Record<string, unknown>) => {
        if (cmd === "plugin:event|listen") {
          const event = args.event as string;
          const handler = args.handler as number;
          listeners.set(event, [...(listeners.get(event) ?? []), handler]);
          return Promise.resolve(handler);
        }
        if (cmd === "get_app_settings") return new Promise(() => {});
        return Promise.resolve(null);
      },
    };
    w.__emit = (event: string, payload: unknown) => {
      for (const id of listeners.get(event) ?? []) {
        callbacks.get(id)?.({ event, id, payload });
      }
    };
    w.__listening = (event: string) => (listeners.get(event) ?? []).length > 0;
  });
}

// Opacity of the card, including its fading ancestors.
const cardOpacity = (page: Page) =>
  page.evaluate(() => {
    const card = document.querySelector<HTMLElement>(".scard");
    if (!card) return 0;
    let opacity = 1;
    for (let el: HTMLElement | null = card; el; el = el.parentElement) {
      opacity *= parseFloat(getComputedStyle(el).opacity);
    }
    return opacity;
  });

test.describe("Recording overlay", () => {
  for (const state of ["recording", "streaming"]) {
    test(`shows ${state} without waiting on settings`, async ({ page }) => {
      await mockOverlayIpc(page);
      await page.goto("/src/overlay/index.html");
      // Listeners register one after another; the last one means all are in.
      await page.waitForFunction(() =>
        (
          window as unknown as { __listening: (e: string) => boolean }
        ).__listening("model-state-changed"),
      );

      await page.evaluate(
        (s) =>
          (
            window as unknown as { __emit: (e: string, p: unknown) => void }
          ).__emit("show-overlay", { state: s, position: "top" }),
        state,
      );

      await expect(page.locator(".ov-stage.top .scard")).toBeVisible();
      await expect.poll(() => cardOpacity(page)).toBeGreaterThanOrEqual(0.9);
    });
  }
});
