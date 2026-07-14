import { useState, useEffect, useCallback } from "react";

export function useSessionState<T>(key: string, defaultValue: T): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [state, setState] = useState<T>(() => {
    try {
      const stored = sessionStorage.getItem(key);
      return stored ? (JSON.parse(stored) as T) : defaultValue;
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      if (state === null || state === undefined) {
        sessionStorage.removeItem(key);
      } else {
        sessionStorage.setItem(key, JSON.stringify(state));
      }
    } catch {
      // storage full or unavailable — silently degrade
    }
  }, [key, state]);

  const clearState = useCallback(() => {
    try {
      sessionStorage.removeItem(key);
    } catch {}
    setState(defaultValue);
  }, [key, defaultValue]);

  return [state, setState];
}

export function clearSessionKey(key: string) {
  try {
    sessionStorage.removeItem(key);
  } catch {}
}
