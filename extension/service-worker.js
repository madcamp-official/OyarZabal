const MLB_ORIGIN = "https://www.mlb.com";

function isGameday(url) {
  try {
    const parsed = new URL(url);
    return parsed.origin === MLB_ORIGIN && parsed.pathname.startsWith("/gameday/");
  } catch {
    return false;
  }
}

async function configurePanel(tabId, url) {
  await chrome.sidePanel.setOptions({
    tabId,
    path: "sidepanel.html",
    enabled: isGameday(url),
  });
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch(() => undefined);
});

chrome.runtime.onStartup.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch(() => undefined);
});

chrome.tabs.onUpdated.addListener((tabId, change, tab) => {
  const url = change.url ?? tab.url;
  if (url) configurePanel(tabId, url).catch(() => undefined);
});
