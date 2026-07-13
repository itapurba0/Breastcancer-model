const CLASSIFIER_API = import.meta.env.VITE_CLASSIFIER_API || "";
const CHAT_API = import.meta.env.VITE_CHAT_API || "";

export function classifierApi(path: string, options?: RequestInit) {
  return fetch(`${CLASSIFIER_API}${path}`, options);
}

export function chatApi(path: string, options?: RequestInit) {
  return fetch(`${CHAT_API}${path}`, options);
}
