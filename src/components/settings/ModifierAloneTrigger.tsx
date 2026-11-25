import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";
import { useOsType } from "../../hooks/useOsType";
import {
  MODIFIERS,
  bindingKeys,
  formatKeyCombination,
} from "../../lib/utils/keyboard";

interface ModifierAloneTriggerProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

const isModifier = (key: string) =>
  MODIFIERS.includes(key.toLowerCase().replace(/_(left|right)$/, ""));

/**
 * The transcribe shortcut's modifiers on their own (Option for Option+Space)
 * also dictate. Only handy-keys on macOS supports it, and only for a shortcut
 * that has both modifiers and a key.
 */
export const ModifierAloneTrigger: React.FC<ModifierAloneTriggerProps> =
  React.memo(({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const osType = useOsType();

    const keys = bindingKeys(
      getSetting("bindings")?.transcribe?.current_binding,
    );
    const modifiers = keys.filter(isModifier);
    if (
      osType !== "macos" ||
      getSetting("keyboard_implementation") !== "handy_keys" ||
      modifiers.length === 0 ||
      modifiers.length === keys.length
    ) {
      return null;
    }
    const label = formatKeyCombination(modifiers.join("+"), osType);

    return (
      <ToggleSwitch
        checked={getSetting("modifier_alone_trigger") ?? false}
        onChange={(enabled) => updateSetting("modifier_alone_trigger", enabled)}
        isUpdating={isUpdating("modifier_alone_trigger")}
        label={t("settings.general.modifierAlone.label", { modifiers: label })}
        description={t("settings.general.modifierAlone.description", {
          modifiers: label,
        })}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  });
